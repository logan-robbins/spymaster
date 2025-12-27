import { Component, inject, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ViewportSelectionService } from '../viewport-selection.service';
import { LevelDerivedService } from '../level-derived.service';

@Component({
  selector: 'app-level-detail-panel',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="detail-panel">
      <header class="panel-header">
        <h3>LEVEL DETAILS</h3>
        <span class="badge" [ngClass]="{'auto': !isPinned(), 'pinned': isPinned()}">
          {{ isPinned() ? 'PINNED' : 'AUTO' }}
        </span>
      </header>
      
      @if (!selectedLevel()) {
        <div class="empty-state">
          <div class="empty-icon">🎯</div>
          <div class="empty-text">No level selected</div>
          <div class="empty-hint">Click a level on the ladder or chart to view details</div>
        </div>
      } @else {
        <div class="level-content">
          <!-- Level Header -->
          <div class="level-card">
            <div class="level-header">
              <div class="level-main">
                <span class="level-kind">{{ selectedLevel()!.kind }}</span>
                @if (selectedLevel()!.ml) {
                  <span class="status-badge" [ngClass]="getTradeabilityClass()">
                    {{ getTradeabilityLabel() }}
                  </span>
                }
              </div>
              <div class="level-price">\${{ selectedLevel()!.price | number:'1.2-2' }}</div>
            </div>
            
            <div class="level-meta">
              <div class="meta-item">
                <span class="label">Direction</span>
                <span class="value direction" [ngClass]="selectedLevel()!.direction">
                  {{ selectedLevel()!.direction }}
                </span>
              </div>
              <div class="meta-item">
                <span class="label">Distance</span>
                <span class="value">\${{ selectedLevel()!.distance | number:'1.2-2' }}</span>
              </div>
              <div class="meta-item">
                <span class="label">Barrier</span>
                <span class="value barrier" [ngClass]="selectedLevel()!.barrier.state">
                  {{ selectedLevel()!.barrier.state }}
                </span>
              </div>
            </div>
          </div>

          <!-- ML Predictions (if available) -->
          @if (selectedLevel()!.ml) {
            <div class="ml-section">
              <h4 class="section-title">ML Predictions</h4>
              
              <div class="prediction-grid">
                <div class="prediction-card">
                  <div class="pred-label">Tradeable</div>
                  <div class="pred-value">{{ (selectedLevel()!.ml!.p_tradeable * 100) | number:'1.0-0' }}%</div>
                </div>
                <div class="prediction-card">
                  <div class="pred-label">Utility</div>
                  <div class="pred-value">{{ (selectedLevel()!.ml!.utility_score * 100) | number:'1.0-0' }}%</div>
                </div>
              </div>
              
              <div class="direction-probs">
                <div class="prob-row">
                  <span class="prob-label">Break</span>
                  <div class="prob-bar">
                    <div class="prob-fill break" [style.width.%]="selectedLevel()!.ml!.p_break * 100"></div>
                  </div>
                  <span class="prob-value">{{ (selectedLevel()!.ml!.p_break * 100) | number:'1.0-0' }}%</span>
                </div>
                <div class="prob-row">
                  <span class="prob-label">Bounce</span>
                  <div class="prob-bar">
                    <div class="prob-fill bounce" [style.width.%]="selectedLevel()!.ml!.p_bounce * 100"></div>
                  </div>
                  <span class="prob-value">{{ (selectedLevel()!.ml!.p_bounce * 100) | number:'1.0-0' }}%</span>
                </div>
              </div>
              
              <div class="timing-section">
                <div class="timing-title">Time to Threshold</div>
                <div class="timing-grid">
                  <div class="timing-item">
                    <span class="timing-label">T1 (60s)</span>
                    <span class="timing-value">{{ (selectedLevel()!.ml!.time_to_threshold.t1_60 * 100) | number:'1.0-0' }}%</span>
                  </div>
                  <div class="timing-item">
                    <span class="timing-label">T1 (120s)</span>
                    <span class="timing-value">{{ (selectedLevel()!.ml!.time_to_threshold.t1_120 * 100) | number:'1.0-0' }}%</span>
                  </div>
                  <div class="timing-item">
                    <span class="timing-label">T2 (60s)</span>
                    <span class="timing-value">{{ (selectedLevel()!.ml!.time_to_threshold.t2_60 * 100) | number:'1.0-0' }}%</span>
                  </div>
                  <div class="timing-item">
                    <span class="timing-label">T2 (120s)</span>
                    <span class="timing-value">{{ (selectedLevel()!.ml!.time_to_threshold.t2_120 * 100) | number:'1.0-0' }}%</span>
                  </div>
                </div>
              </div>
            </div>
          }

          <!-- Physics Attribution -->
          <div class="physics-section">
            <h4 class="section-title">Physics Forces</h4>
            
            <div class="force-breakdown">
              <div class="force-row">
                <span class="force-label">Barrier</span>
                <div class="force-bar-container">
                  <div class="force-bar break" [style.width.%]="getForcePercent(selectedLevel()!.forces.break.barrier)"></div>
                  <div class="force-bar bounce" [style.width.%]="getForcePercent(selectedLevel()!.forces.bounce.barrier)"></div>
                </div>
              </div>
              <div class="force-row">
                <span class="force-label">Tape</span>
                <div class="force-bar-container">
                  <div class="force-bar break" [style.width.%]="getForcePercent(selectedLevel()!.forces.break.tape)"></div>
                  <div class="force-bar bounce" [style.width.%]="getForcePercent(selectedLevel()!.forces.bounce.tape)"></div>
                </div>
              </div>
              <div class="force-row">
                <span class="force-label">Fuel</span>
                <div class="force-bar-container">
                  <div class="force-bar break" [style.width.%]="getForcePercent(selectedLevel()!.forces.break.fuel)"></div>
                  <div class="force-bar bounce" [style.width.%]="getForcePercent(selectedLevel()!.forces.bounce.fuel)"></div>
                </div>
              </div>
            </div>
          </div>

          <!-- Confluence Quality -->
          @if (selectedLevel()!.confluence.level > 0) {
            <div class="confluence-section">
              <h4 class="section-title">Setup Quality</h4>
              <div class="confluence-card" [attr.data-quality]="getConfluenceQuality()">
                <div class="confluence-name">{{ selectedLevel()!.confluence.levelName }}</div>
                <div class="confluence-level">Level {{ selectedLevel()!.confluence.level }}/10</div>
                <div class="confluence-pressure">Pressure: {{ (selectedLevel()!.confluence.pressure * 100) | number:'1.0-0' }}%</div>
              </div>
            </div>
          }
        </div>
      }
    </div>
  `,
  styles: [`
    .detail-panel {
      height: 100%;
      background: rgba(15, 23, 42, 0.6);
      padding: 1rem;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    .detail-panel::-webkit-scrollbar {
      width: 4px;
    }

    .detail-panel::-webkit-scrollbar-track {
      background: rgba(148, 163, 184, 0.1);
    }

    .detail-panel::-webkit-scrollbar-thumb {
      background: rgba(148, 163, 184, 0.3);
      border-radius: 2px;
    }

    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    h3 {
      font-size: 0.8rem;
      font-weight: 700;
      letter-spacing: 0.1em;
      color: #e2e8f0;
      font-family: 'Space Grotesk', sans-serif;
      text-transform: uppercase;
    }

    .badge {
      font-size: 0.65rem;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-family: 'IBM Plex Mono', monospace;
      letter-spacing: 0.1em;
      font-weight: 600;
    }

    .badge.auto {
      background: rgba(56, 189, 248, 0.15);
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.3);
    }

    .badge.pinned {
      background: rgba(168, 85, 247, 0.15);
      color: #a855f7;
      border: 1px solid rgba(168, 85, 247, 0.3);
    }

    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.5rem;
      padding: 3rem 1rem;
      text-align: center;
    }

    .empty-icon {
      font-size: 3rem;
      opacity: 0.3;
    }

    .empty-text {
      font-size: 0.9rem;
      color: #94a3b8;
      font-weight: 600;
    }

    .empty-hint {
      font-size: 0.7rem;
      color: #64748b;
      font-style: italic;
    }

    .level-content {
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    .level-card {
      background: rgba(30, 41, 59, 0.6);
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 12px;
      padding: 1rem;
    }

    .level-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 0.75rem;
    }

    .level-main {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .level-kind {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.7rem;
      color: #94a3b8;
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }

    .status-badge {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.65rem;
      padding: 0.2rem 0.4rem;
      border-radius: 4px;
      font-weight: 700;
      letter-spacing: 0.1em;
      width: fit-content;
    }

    .status-badge.go {
      background: rgba(34, 197, 94, 0.2);
      color: #86efac;
      border: 1px solid rgba(34, 197, 94, 0.4);
    }

    .status-badge.wait {
      background: rgba(251, 191, 36, 0.2);
      color: #fbbf24;
      border: 1px solid rgba(251, 191, 36, 0.4);
    }

    .status-badge.no {
      background: rgba(248, 113, 113, 0.2);
      color: #fca5a5;
      border: 1px solid rgba(248, 113, 113, 0.4);
    }

    .level-price {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 1.8rem;
      font-weight: 700;
      color: #f8fafc;
    }

    .level-meta {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0.5rem;
      padding-top: 0.75rem;
      border-top: 1px solid rgba(148, 163, 184, 0.1);
    }

    .meta-item {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }

    .meta-item .label {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.6rem;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }

    .meta-item .value {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.75rem;
      color: #e2e8f0;
      font-weight: 600;
    }

    .value.direction.UP {
      color: #fca5a5;
    }

    .value.direction.DOWN {
      color: #86efac;
    }

    .value.barrier.WALL {
      color: #f8fafc;
    }

    .value.barrier.VACUUM {
      color: #fbbf24;
    }

    .section-title {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.7rem;
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-bottom: 0.5rem;
    }

    .ml-section, .physics-section, .confluence-section {
      background: rgba(15, 23, 42, 0.4);
      border: 1px solid rgba(148, 163, 184, 0.15);
      border-radius: 10px;
      padding: 0.75rem;
    }

    .prediction-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.5rem;
      margin-bottom: 0.75rem;
    }

    .prediction-card {
      background: rgba(30, 41, 59, 0.5);
      border: 1px solid rgba(148, 163, 184, 0.1);
      border-radius: 6px;
      padding: 0.5rem;
      text-align: center;
    }

    .pred-label {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.6rem;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-bottom: 0.25rem;
    }

    .pred-value {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 1.2rem;
      font-weight: 700;
      color: #38bdf8;
    }

    .direction-probs {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      margin-bottom: 0.75rem;
    }

    .prob-row {
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 0.5rem;
      align-items: center;
    }

    .prob-label {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.65rem;
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      width: 60px;
    }

    .prob-bar {
      height: 6px;
      background: rgba(148, 163, 184, 0.2);
      border-radius: 999px;
      overflow: hidden;
    }

    .prob-fill {
      height: 100%;
      border-radius: 999px;
    }

    .prob-fill.break {
      background: linear-gradient(90deg, rgba(248, 113, 113, 0.5), #f87171);
    }

    .prob-fill.bounce {
      background: linear-gradient(90deg, rgba(34, 197, 94, 0.5), #22c55e);
    }

    .prob-value {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.7rem;
      color: #f8fafc;
      font-weight: 600;
      width: 45px;
      text-align: right;
    }

    .timing-section {
      border-top: 1px solid rgba(148, 163, 184, 0.1);
      padding-top: 0.75rem;
    }

    .timing-title {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.65rem;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-bottom: 0.5rem;
    }

    .timing-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.4rem;
    }

    .timing-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.3rem 0.5rem;
      background: rgba(30, 41, 59, 0.3);
      border-radius: 4px;
    }

    .timing-label {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.6rem;
      color: #94a3b8;
    }

    .timing-value {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.65rem;
      color: #cbd5e1;
      font-weight: 600;
    }

    .force-breakdown {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .force-row {
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 0.5rem;
      align-items: center;
    }

    .force-label {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.65rem;
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      width: 60px;
    }

    .force-bar-container {
      display: flex;
      height: 6px;
      background: rgba(148, 163, 184, 0.1);
      border-radius: 999px;
      overflow: hidden;
    }

    .force-bar {
      height: 100%;
    }

    .force-bar.break {
      background: rgba(248, 113, 113, 0.6);
    }

    .force-bar.bounce {
      background: rgba(34, 197, 94, 0.6);
    }

    .confluence-card {
      padding: 0.75rem;
      border-radius: 8px;
      text-align: center;
    }

    .confluence-card[data-quality="premium"] {
      background: rgba(34, 197, 94, 0.15);
      border: 1px solid rgba(34, 197, 94, 0.3);
    }

    .confluence-card[data-quality="strong"] {
      background: rgba(251, 191, 36, 0.15);
      border: 1px solid rgba(251, 191, 36, 0.3);
    }

    .confluence-card[data-quality="moderate"] {
      background: rgba(148, 163, 184, 0.15);
      border: 1px solid rgba(148, 163, 184, 0.3);
    }

    .confluence-name {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.85rem;
      font-weight: 700;
      color: #f8fafc;
      margin-bottom: 0.25rem;
    }

    .confluence-level {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.7rem;
      color: #94a3b8;
      margin-bottom: 0.5rem;
    }

    .confluence-pressure {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.7rem;
      color: #cbd5e1;
    }
  `]
})
export class LevelDetailPanelComponent {
  private viewportService = inject(ViewportSelectionService);
  private derivedService = inject(LevelDerivedService);

  public selectedTarget = this.viewportService.selectedTarget;
  public isPinned = this.viewportService.isPinned;
  
  // Get the full level data from derived service
  public selectedLevel = computed(() => {
    const target = this.selectedTarget();
    if (!target) return null;
    
    const levels = this.derivedService.levels();
    return levels.find(l => l.id === target.level_id) || null;
  });

  public getTradeabilityLabel(): string {
    const level = this.selectedLevel();
    if (!level?.ml) return 'N/A';
    
    const p_tradeable = level.ml.p_tradeable;
    const direction_confidence = Math.max(level.ml.p_break, level.ml.p_bounce);
    const combined = p_tradeable * direction_confidence;
    
    if (combined >= 0.50 && p_tradeable >= 0.60) return 'GO';
    if (combined >= 0.30 && p_tradeable >= 0.40) return 'WAIT';
    return 'NO-GO';
  }

  public getTradeabilityClass(): string {
    const label = this.getTradeabilityLabel();
    if (label === 'GO') return 'go';
    if (label === 'WAIT') return 'wait';
    return 'no';
  }

  public getForcePercent(value: number): number {
    return Math.max(0, Math.min(100, Math.abs(value) * 50));
  }

  public getConfluenceQuality(): string {
    const level = this.selectedLevel();
    if (!level) return 'moderate';
    if (level.confluence.level <= 3) return 'premium';
    if (level.confluence.level <= 5) return 'strong';
    return 'moderate';
  }
}
