import { Injectable, computed, inject, Signal } from '@angular/core';
import { DataStreamService, ViewportTarget } from './data-stream.service';
import { DerivedLevel } from './level-derived.service';

/**
 * Tradeability assessment based on ML model confidence
 */
export type TradeabilitySignal = 'GO' | 'WAIT' | 'NO-GO';

/**
 * Integrated assessment combining physics + ML
 */
export interface MLEnhancedLevel extends DerivedLevel {
    ml: {
        available: boolean;
        tradeability: TradeabilitySignal;
        p_tradeable: number;
        p_break_ml: number;
        p_bounce_ml: number;
        strength_ml: number;
        expected_time_60s: number;  // Probability of hitting target in 60s
        expected_time_120s: number; // Probability of hitting target in 120s
        confidence_boost: number;    // ML agreement with physics (-1 to +1)
        retrieval_similarity: number; // How similar to historical patterns
    };
}

/**
 * ML Integration Service
 * 
 * Purpose: Bridge ML predictions (viewport) with physics-based level analysis.
 * 
 * Philosophy:
 * - Physics tells us WHY (barrier/tape/fuel mechanics)
 * - ML tells us WHAT (historical pattern outcomes)
 * - Together they tell us IF (should we trade this)
 * 
 * For SPY 0DTE traders:
 * - Tradeability > 60%: GO (high confidence, clear direction)
 * - Tradeability 40-60%: WAIT (choppy, unclear)
 * - Tradeability < 40%: NO-GO (avoid, likely to whipsaw)
 */
@Injectable({
    providedIn: 'root'
})
export class MLIntegrationService {
    private dataStream = inject(DataStreamService);
    
    /**
     * Map of level_id -> viewport target for fast lookup
     */
    public viewportByLevel: Signal<Map<string, ViewportTarget>> = computed(() => {
        const viewport = this.dataStream.viewportData();
        if (!viewport) return new Map();
        
        const map = new Map<string, ViewportTarget>();
        for (const target of viewport.targets) {
            map.set(target.level_id, target);
        }
        return map;
    });
    
    /**
     * Check if ML predictions are available
     */
    public mlAvailable: Signal<boolean> = computed(() => {
        const viewport = this.dataStream.viewportData();
        return viewport !== null && viewport.targets.length > 0;
    });
    
    /**
     * Get ML prediction for a specific level
     */
    public getMLForLevel(levelId: string): ViewportTarget | null {
        return this.viewportByLevel().get(levelId) ?? null;
    }
    
    /**
     * Enhance a derived level with ML predictions
     */
    public enhanceLevel(level: DerivedLevel): MLEnhancedLevel {
        const mlData = this.getMLForLevel(level.id);
        
        if (!mlData) {
            // No ML data available - return level with placeholder ML
            return {
                ...level,
                ml: {
                    available: false,
                    tradeability: 'WAIT',
                    p_tradeable: 0,
                    p_break_ml: 0,
                    p_bounce_ml: 0,
                    strength_ml: 0,
                    expected_time_60s: 0,
                    expected_time_120s: 0,
                    confidence_boost: 0,
                    retrieval_similarity: 0
                }
            };
        }
        
        // ML data available - compute integrated metrics
        const tradeability = this.computeTradeability(mlData);
        const confidenceBoost = this.computeConfidenceBoost(level, mlData);
        const expectedTime60 = mlData.time_to_threshold.t2['60'] ?? 0;
        const expectedTime120 = mlData.time_to_threshold.t2['120'] ?? 0;
        
        return {
            ...level,
            ml: {
                available: true,
                tradeability: tradeability,
                p_tradeable: mlData.p_tradeable_2,
                p_break_ml: mlData.p_break,
                p_bounce_ml: mlData.p_bounce,
                strength_ml: Math.abs(mlData.strength_signed),
                expected_time_60s: expectedTime60,
                expected_time_120s: expectedTime120,
                confidence_boost: confidenceBoost,
                retrieval_similarity: mlData.retrieval.similarity
            }
        };
    }
    
    /**
     * Compute tradeability signal (GO/WAIT/NO-GO)
     * 
     * Logic:
     * - GO: High probability of clean move (p_tradeable > 0.60)
     * - WAIT: Uncertain or mixed signals (0.40 <= p_tradeable <= 0.60)
     * - NO-GO: High chop risk (p_tradeable < 0.40)
     */
    private computeTradeability(ml: ViewportTarget): TradeabilitySignal {
        const p_tradeable = ml.p_tradeable_2;
        
        // Factor in directional confidence
        const direction_confidence = Math.max(ml.p_break, ml.p_bounce);
        
        // Combined score: need both tradeability AND directional confidence
        const combined = p_tradeable * direction_confidence;
        
        if (combined >= 0.50 && p_tradeable >= 0.60) {
            return 'GO';
        } else if (combined >= 0.30 && p_tradeable >= 0.40) {
            return 'WAIT';
        } else {
            return 'NO-GO';
        }
    }
    
    /**
     * Compute confidence boost from ML/physics agreement
     * 
     * Returns:
     * - +1.0: ML strongly agrees with physics
     * - 0.0: ML neutral or mixed
     * - -1.0: ML contradicts physics
     * 
     * This helps traders understand when physics and ML align (high confidence)
     * vs when they disagree (lower confidence, proceed with caution)
     */
    private computeConfidenceBoost(level: DerivedLevel, ml: ViewportTarget): number {
        // Physics bias
        const physics_break_strength = level.breakStrength / 100;
        const physics_bounce_strength = level.bounceStrength / 100;
        const physics_net = physics_break_strength - physics_bounce_strength;
        
        // ML bias
        const ml_net = ml.p_break - ml.p_bounce;
        
        // Agreement: both point same direction
        const agreement = physics_net * ml_net;
        
        // Normalize to [-1, +1] range
        return Math.tanh(agreement * 3);
    }
    
    /**
     * Get tradeability description for UI
     */
    public getTradeabilityDescription(signal: TradeabilitySignal): string {
        switch (signal) {
            case 'GO':
                return 'High confidence setup - clear direction expected';
            case 'WAIT':
                return 'Mixed signals - wait for clearer setup';
            case 'NO-GO':
                return 'High chop risk - avoid or use tight stops';
        }
    }
    
    /**
     * Get confidence boost description for UI
     */
    public getConfidenceDescription(boost: number): string {
        if (boost > 0.5) {
            return 'Physics & ML strongly agree';
        } else if (boost > 0.2) {
            return 'Physics & ML moderately agree';
        } else if (boost > -0.2) {
            return 'Physics & ML neutral';
        } else if (boost > -0.5) {
            return 'Physics & ML moderately disagree';
        } else {
            return 'Physics & ML strongly disagree - caution';
        }
    }
    
    /**
     * Format time to threshold for display
     */
    public formatTimeHorizon(prob_60s: number, prob_120s: number): string {
        if (prob_120s > 0.60) {
            return 'Fast move expected (<2 min)';
        } else if (prob_120s > 0.30) {
            return 'Moderate pace (2-5 min)';
        } else {
            return 'Slow grind or no move';
        }
    }
}

