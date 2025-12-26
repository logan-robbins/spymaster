import { Injectable, computed, effect, inject, signal, untracked } from '@angular/core';
import { DataStreamService, LevelSignal, LevelsPayload } from './data-stream.service';
import { ViewportSelectionService } from './viewport-selection.service';

type Direction = 'UP' | 'DOWN';
type SignalBias = 'BREAK' | 'BOUNCE' | 'NEUTRAL';

export interface ForceComponents {
  barrier: number;
  tape: number;
  fuel: number;
  approach: number;
  confluence: number;
  total: number;
}

export interface ForceVector {
  break: ForceComponents;
  bounce: ForceComponents;
  net: number; // break.total - bounce.total (Positive = Break bias)
}

export interface DerivedLevel {
  id: string;
  price: number;
  kind: string;
  direction: Direction;
  distance: number;
  breakStrength: number;
  bounceStrength: number;
  bias: SignalBias;
  confidence: LevelSignal['confidence'];
  signal: LevelSignal['signal'];
  barrier: {
    state: LevelSignal['barrier_state'];
    deltaLiq: number;
    wallRatio: number;
    replenishmentRatio: number;
  };
  tape: {
    imbalance: number;
    velocity: number;
    buyVol: number;
    sellVol: number;
    sweepDetected: boolean;
  };
  fuel: {
    effect: LevelSignal['fuel_effect'];
    gammaExposure: number;
    gammaVelocity: number;
  };
  approach: {
    velocity: number;
    bars: number;
    distance: number;
    priorTouches: number;
    barsSinceOpen: number;
    isFirst15m: boolean;
  };
  forces: ForceVector; // Tug-of-War Vector
  
  // Backend-computed confluence features (Phase 3)
  confluence: {
    count: number;           // Number of nearby key levels
    pressure: number;        // Weighted pressure (0-1)
    alignment: number;       // -1=OPPOSED, 0=NEUTRAL, 1=ALIGNED
    level: number;           // 0-10 hierarchical quality scale
    levelName: string;       // ULTRA_PREMIUM, PREMIUM, STRONG, etc.
  };
  
  // ML predictions from viewport (if available)
  ml?: {
    available: boolean;
    p_tradeable: number;        // P(tradeable)
    p_break: number;            // P(break | tradeable)
    p_bounce: number;           // P(bounce | tradeable)
    strength_signed: number;    // Predicted signed strength
    utility_score: number;      // Overall utility score
    stage: string;              // "stage_a" or "stage_b"
    time_to_threshold: {
      t1_60: number;            // P(hit t1 within 60s)
      t1_120: number;           // P(hit t1 within 120s)
      t2_60: number;            // P(hit t2 within 60s)
      t2_120: number;           // P(hit t2 within 120s)
    };
    retrieval_similarity: number; // kNN similarity score
  };
  
  confluenceId?: string; // Legacy: for UI grouping
}

export interface ConfluenceGroup {
  id: string;
  centerPrice: number;
  levels: DerivedLevel[];
  strength: number;
  bias: SignalBias;
  score: number;
}

const LEVEL_WEIGHTS: Record<string, number> = {
  PM_HIGH: 1.0,
  PM_LOW: 1.0,
  OR_HIGH: 1.0,
  OR_LOW: 1.0,
  SESSION_HIGH: 0.9,
  SESSION_LOW: 0.9,
  SMA_200: 0.7,
  SMA_400: 0.7,
  CALL_WALL: 0.7,
  PUT_WALL: 0.7,
  VWAP: 0.5,
  ROUND: 0.4,
  STRIKE: 0.4
};

const BARRIER_BREAK: Record<string, number> = {
  VACUUM: 1.0,
  WEAK: 0.7,
  CONSUMED: 0.6,
  NEUTRAL: 0.4,
  ABSORPTION: 0.2,
  WALL: 0.1
};

const BARRIER_BOUNCE: Record<string, number> = {
  WALL: 1.0,
  ABSORPTION: 0.8,
  CONSUMED: 0.5,
  NEUTRAL: 0.4,
  WEAK: 0.2,
  VACUUM: 0.1
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function toPercent(value: number): number {
  return Math.round(clamp(value, 0, 1) * 100);
}

function assertFinite(value: number, label: string) {
  if (!Number.isFinite(value)) {
    throw new Error(`Invalid level field: ${label}`);
  }
}

function validateLevel(level: LevelSignal): boolean {
  // Fail-soft: if any critical numeric field is missing/invalid, skip this level.
  const values = [
    level.level_price,
    level.distance,
    level.tape_velocity,
    level.gamma_exposure,
    level.barrier_delta_liq
  ];
  return values.every((v) => typeof v === 'number' && Number.isFinite(v));
}

function normalizeVelocity(value: number, scale: number): number {
  return clamp(Math.abs(value) / scale, 0, 1);
}

function computeBarrierScores(level: LevelSignal) {
  const wallNorm = clamp((level.wall_ratio - 1) / 1.5, 0, 1);
  const deltaNorm = clamp(Math.abs(level.barrier_delta_liq) / 200, 0, 1);
  const deltaBreak = level.barrier_delta_liq < 0 ? deltaNorm : deltaNorm * 0.2;
  const deltaBounce = level.barrier_delta_liq > 0 ? deltaNorm : deltaNorm * 0.2;
  const stateBreak = BARRIER_BREAK[level.barrier_state] ?? 0.4;
  const stateBounce = BARRIER_BOUNCE[level.barrier_state] ?? 0.4;

  const breakScore = clamp(stateBreak * 0.6 + (1 - wallNorm) * 0.2 + deltaBreak * 0.2, 0, 1);
  const bounceScore = clamp(stateBounce * 0.6 + wallNorm * 0.2 + deltaBounce * 0.2, 0, 1);

  return { breakScore, bounceScore };
}

function computeTapeScores(level: LevelSignal) {
  const velocityNorm = normalizeVelocity(level.tape_velocity, 60);
  const imbalance = clamp(level.tape_imbalance, -1, 1);
  const isUp = level.direction === 'UP';
  const imbalanceBreak = isUp ? (imbalance + 1) / 2 : (1 - imbalance) / 2;
  const imbalanceBounce = isUp ? (1 - imbalance) / 2 : (imbalance + 1) / 2;
  const sweepBoost = level.sweep_detected ? 0.15 : 0;

  const breakScore = clamp(imbalanceBreak * 0.7 + velocityNorm * 0.3 + sweepBoost, 0, 1);
  const bounceScore = clamp(imbalanceBounce * 0.7 + velocityNorm * 0.3, 0, 1);

  return { breakScore, bounceScore };
}

function computeFuelScores(level: LevelSignal) {
  const gammaNorm = normalizeVelocity(level.gamma_exposure, 50000);
  const effectBreak = level.fuel_effect === 'AMPLIFY' ? 1 : level.fuel_effect === 'NEUTRAL' ? 0.5 : 0.2;
  const effectBounce = level.fuel_effect === 'DAMPEN' ? 1 : level.fuel_effect === 'NEUTRAL' ? 0.5 : 0.2;
  const signBreak = level.gamma_exposure < 0 ? 1 : 0.3;
  const signBounce = level.gamma_exposure > 0 ? 1 : 0.3;

  const breakScore = clamp(effectBreak * 0.6 + gammaNorm * 0.3 + signBreak * 0.1, 0, 1);
  const bounceScore = clamp(effectBounce * 0.6 + gammaNorm * 0.3 + signBounce * 0.1, 0, 1);

  return { breakScore, bounceScore };
}

function computeApproachScores(level: LevelSignal) {
  const speedNorm = normalizeVelocity(level.approach_velocity, 0.5);
  const distanceNorm = normalizeVelocity(level.approach_distance, 2);
  const touchNorm = normalizeVelocity(level.prior_touches, 5);
  const base = clamp((speedNorm + distanceNorm) / 2, 0, 1);
  const breakScore = clamp(base + touchNorm * 0.2, 0, 1);
  const bounceScore = clamp(1 - base * 0.7, 0, 1);

  return { breakScore, bounceScore };
}

function computeConfluenceGroups(levels: LevelSignal[], band: number): Array<{ id: string; center: number; score: number; levelIds: Set<string> }> {
  const sorted = [...levels].sort((a, b) => b.level_price - a.level_price);
  const groups: Array<{ id: string; center: number; score: number; levelIds: Set<string> }> = [];

  let current: { levels: LevelSignal[]; center: number; score: number } | null = null;

  for (const level of sorted) {
    const weight = LEVEL_WEIGHTS[level.level_kind_name] ?? 0.3;
    if (!current) {
      current = { levels: [level], center: level.level_price, score: weight };
      continue;
    }

    const delta = Math.abs(level.level_price - current.center);
    if (delta <= band) {
      current.levels.push(level);
      current.score += weight;
      current.center = current.levels.reduce((sum, l) => sum + l.level_price, 0) / current.levels.length;
    } else {
      const id = `conf-${current.center.toFixed(2)}`;
      groups.push({ id, center: current.center, score: current.score, levelIds: new Set(current.levels.map((l) => l.id)) });
      current = { levels: [level], center: level.level_price, score: weight };
    }
  }

  if (current) {
    const id = `conf-${current.center.toFixed(2)}`;
    groups.push({ id, center: current.center, score: current.score, levelIds: new Set(current.levels.map((l) => l.id)) });
  }

  return groups;
}

@Injectable({ providedIn: 'root' })
export class LevelDerivedService {
  private dataStream = inject(DataStreamService);
  private confluenceBand = signal(0.15);

  private gammaVelocity = signal<Record<string, number>>({});

  constructor() {
    effect(() => {
      const payload = this.dataStream.levelsData();
      if (!payload) return;
      this.updateGammaVelocity(payload);
    });
  }

  private updateGammaVelocity(payload: LevelsPayload) {
    const velocities: Record<string, number> = { ...untracked(() => this.gammaVelocity()) };
    const now = payload.ts;

    const levels = Array.isArray((payload as any).levels) ? (payload as any).levels as LevelSignal[] : [];
    for (const level of levels) {
      const key = level.id;
      const prev = velocities[key];
      const previousEntry = this.lastGamma.get(key);
      if (previousEntry && now > previousEntry.ts) {
        const dt = (now - previousEntry.ts) / 1000;
        const velocity = dt > 0 ? (level.gamma_exposure - previousEntry.gamma) / dt : 0;
        velocities[key] = velocity;
      } else if (prev === undefined) {
        velocities[key] = 0;
      }
      this.lastGamma.set(key, { ts: now, gamma: level.gamma_exposure });
    }

    this.gammaVelocity.set(velocities);
  }

  private lastGamma = new Map<string, { ts: number; gamma: number }>();

  public spy = computed(() => this.dataStream.levelsData()?.spy ?? null);

  public levels = computed(() => {
    const payload = this.dataStream.levelsData();
    if (!payload) return [] as DerivedLevel[];

    const rawLevels = Array.isArray((payload as any).levels) ? (payload as any).levels as LevelSignal[] : [];
    const validLevels = rawLevels.filter(validateLevel);
    if (!validLevels.length) return [] as DerivedLevel[];

    const confluenceGroups = computeConfluenceGroups(validLevels, this.confluenceBand());
    const confluenceLookup = new Map<string, { id: string; strength: number }>();

    for (const group of confluenceGroups) {
      const strength = clamp(group.score / 3, 0, 1);
      for (const levelId of group.levelIds) {
        confluenceLookup.set(levelId, { id: group.id, strength });
      }
    }

    return validLevels.map((level) => {
      const barrier = computeBarrierScores(level);
      const tape = computeTapeScores(level);
      const fuel = computeFuelScores(level);
      const approach = computeApproachScores(level);
      const confluence = confluenceLookup.get(level.id);
      const confluenceStrength = confluence?.strength ?? 0;

      const weights = { barrier: 0.3, tape: 0.25, fuel: 0.25, approach: 0.1, confluence: 0.1 };

      const breakTotal = 
        barrier.breakScore * weights.barrier +
        tape.breakScore * weights.tape +
        fuel.breakScore * weights.fuel +
        approach.breakScore * weights.approach +
        confluenceStrength * weights.confluence;

      const bounceTotal = 
        barrier.bounceScore * weights.barrier +
        tape.bounceScore * weights.tape +
        fuel.bounceScore * weights.fuel +
        approach.bounceScore * weights.approach +
        confluenceStrength * weights.confluence;

      const breakValue = toPercent(breakTotal);
      const bounceValue = toPercent(bounceTotal);

      // Force Vector (Tug-of-War): per-component contributions are WEIGHTED percent-points (0..100).
      // These are designed to stack cleanly into break.total / bounce.total.
      const forces: ForceVector = {
        break: {
          barrier: barrier.breakScore * weights.barrier * 100,
          tape: tape.breakScore * weights.tape * 100,
          fuel: fuel.breakScore * weights.fuel * 100,
          approach: approach.breakScore * weights.approach * 100,
          confluence: confluenceStrength * weights.confluence * 100,
          total: breakValue
        },
        bounce: {
          barrier: barrier.bounceScore * weights.barrier * 100,
          tape: tape.bounceScore * weights.tape * 100,
          fuel: fuel.bounceScore * weights.fuel * 100,
          approach: approach.bounceScore * weights.approach * 100,
          confluence: confluenceStrength * weights.confluence * 100,
          total: bounceValue
        },
        net: breakValue - bounceValue
      };

      const bias: SignalBias = breakValue > bounceValue ? 'BREAK' : bounceValue > breakValue ? 'BOUNCE' : 'NEUTRAL';

      // Map backend confluence features
      const backendConfluence = {
        count: level.confluence_count ?? 0,
        pressure: level.confluence_pressure ?? 0,
        alignment: level.confluence_alignment ?? 0,
        level: level.confluence_level ?? 0,
        levelName: level.confluence_level_name ?? 'UNDEFINED'
      };

      // Map ML predictions if available
      const mlPredictions = level.ml_predictions ? {
        available: true,
        p_tradeable: level.ml_predictions.p_tradeable_2 ?? 0,
        p_break: level.ml_predictions.p_break ?? 0,
        p_bounce: level.ml_predictions.p_bounce ?? 0,
        strength_signed: level.ml_predictions.strength_signed ?? 0,
        utility_score: level.ml_predictions.utility_score ?? 0,
        stage: level.ml_predictions.stage ?? 'stage_a',
        time_to_threshold: {
          t1_60: level.ml_predictions.time_to_threshold?.['t1']?.['60'] ?? 0,
          t1_120: level.ml_predictions.time_to_threshold?.['t1']?.['120'] ?? 0,
          t2_60: level.ml_predictions.time_to_threshold?.['t2']?.['60'] ?? 0,
          t2_120: level.ml_predictions.time_to_threshold?.['t2']?.['120'] ?? 0
        },
        retrieval_similarity: level.ml_predictions.retrieval?.['similarity'] ?? 0
      } : undefined;

      return {
        id: level.id,
        price: level.level_price,
        kind: level.level_kind_name,
        direction: level.direction,
        distance: level.distance,
        breakStrength: breakValue,
        bounceStrength: bounceValue,
        bias,
        confidence: level.confidence,
        signal: level.signal,
        barrier: {
          state: level.barrier_state,
          deltaLiq: level.barrier_delta_liq,
          wallRatio: level.wall_ratio,
          replenishmentRatio: level.barrier_replenishment_ratio
        },
        tape: {
          imbalance: level.tape_imbalance,
          velocity: level.tape_velocity,
          buyVol: level.tape_buy_vol,
          sellVol: level.tape_sell_vol,
          sweepDetected: level.sweep_detected
        },
        fuel: {
          effect: level.fuel_effect,
          gammaExposure: level.gamma_exposure,
          gammaVelocity: this.gammaVelocity()[level.id] ?? 0
        },
        approach: {
          velocity: level.approach_velocity,
          bars: level.approach_bars,
          distance: level.approach_distance,
          priorTouches: level.prior_touches,
          barsSinceOpen: level.bars_since_open,
          isFirst15m: level.is_first_15m
        },
        forces,
        confluence: backendConfluence,
        ml: mlPredictions,
        confluenceId: confluence?.id  // Keep legacy for UI grouping
      };
    }).sort((a, b) => Math.abs(a.distance) - Math.abs(b.distance));
  });

  public confluenceGroups = computed(() => {
    const payload = this.dataStream.levelsData();
    if (!payload) return [] as ConfluenceGroup[];

    const groups = computeConfluenceGroups(payload.levels, this.confluenceBand());
    const derived = this.levels();

    return groups.map((group) => {
      const levels = derived.filter((level) => group.levelIds.has(level.id));
      const avgBreak = levels.reduce((sum, level) => sum + level.breakStrength, 0) / Math.max(1, levels.length);
      const avgBounce = levels.reduce((sum, level) => sum + level.bounceStrength, 0) / Math.max(1, levels.length);
      const bias: SignalBias = avgBreak > avgBounce ? 'BREAK' : avgBounce > avgBreak ? 'BOUNCE' : 'NEUTRAL';
      return {
        id: group.id,
        centerPrice: group.center,
        levels,
        score: group.score,
        strength: toPercent(clamp(group.score / 3, 0, 1)),
        bias
      };
    }).sort((a, b) => b.strength - a.strength);
  });

  /**
   * Primary level for cockpit display
   * 
   * NEW: Integrates with viewport selection
   * - If viewport target selected: show that level's physics
   * - Otherwise: show closest level (legacy behavior)
   */
  public primaryLevel = computed(() => {
    const levels = this.levels();
    if (levels.length === 0) return null;
    
    // Check if viewport selection service is available and has a selection
    try {
      const viewportService = inject(ViewportSelectionService);
      const selectedTarget = viewportService.selectedTarget();
      
      if (selectedTarget) {
        // Find the physics level matching the viewport target
        const matchingLevel = levels.find(l => l.id === selectedTarget.level_id);
        if (matchingLevel) return matchingLevel;
      }
    } catch {
      // ViewportSelectionService not available, fall back to closest
    }
    
    // Default: closest level
    return levels[0];
  });

  public setConfluenceBand(value: number) {
    this.confluenceBand.set(clamp(value, 0.05, 0.5));
  }

  public getConfluenceBand() {
    return this.confluenceBand.asReadonly();
  }
}
