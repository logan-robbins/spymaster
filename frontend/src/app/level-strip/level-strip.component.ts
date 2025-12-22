import { Component, computed, inject, Input } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { DataStreamService, LevelSignal } from '../data-stream.service';

@Component({
  selector: 'app-level-strip',
  standalone: true,
  imports: [CommonModule, DecimalPipe],
  styles: [`
    .level-strip {
      background: rgba(26, 32, 44, 0.95);
      border: 1px solid #4a5568;
      border-radius: 8px;
      padding: 0.75rem;
      backdrop-filter: blur(8px);
    }

    .strip-header {
      font-size: 0.75rem;
      color: #94a3b8;
      margin-bottom: 0.5rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .levels-container {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .level-item {
      display: grid;
      grid-template-columns: auto 1fr auto auto;
      gap: 0.5rem;
      align-items: center;
      padding: 0.5rem;
      border-radius: 4px;
      background: #2d3748;
      border-left: 3px solid transparent;
      transition: all 0.2s;
    }

    .level-item:hover {
      background: #374151;
      transform: translateX(2px);
    }

    .level-item.signal-BREAK {
      border-left-color: #ef4444;
      background: rgba(239, 68, 68, 0.1);
    }

    .level-item.signal-REJECT {
      border-left-color: #10b981;
      background: rgba(16, 185, 129, 0.1);
    }

    .level-item.signal-CONTESTED {
      border-left-color: #fbbf24;
      background: rgba(251, 191, 36, 0.1);
    }

    .level-badge {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 2.5rem;
      height: 2.5rem;
      border-radius: 50%;
      font-weight: 700;
      font-size: 0.875rem;
    }

    .badge-BREAK {
      background: #ef4444;
      color: white;
    }

    .badge-REJECT {
      background: #10b981;
      color: white;
    }

    .badge-CONTESTED {
      background: #fbbf24;
      color: #78350f;
    }

    .badge-NEUTRAL {
      background: #4a5568;
      color: #cbd5e0;
    }

    .level-info {
      display: flex;
      flex-direction: column;
      gap: 0.125rem;
      min-width: 0;
    }

    .level-name {
      font-weight: 600;
      font-size: 0.875rem;
      color: #e2e8f0;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .level-price {
      font-size: 0.75rem;
      color: #94a3b8;
    }

    .level-direction {
      font-size: 0.7rem;
      font-weight: 600;
    }

    .direction-SUPPORT { color: #10b981; }
    .direction-RESISTANCE { color: #ef4444; }

    .level-distance {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }

    .distance-value {
      font-weight: 600;
      font-size: 0.875rem;
      color: #cbd5e0;
    }

    .distance-label {
      font-size: 0.65rem;
      color: #64748b;
      text-transform: uppercase;
    }

    .level-score {
      text-align: right;
      font-variant-numeric: tabular-nums;
      min-width: 3rem;
    }

    .score-value {
      font-weight: 700;
      font-size: 1rem;
    }

    .score-BREAK { color: #ef4444; }
    .score-REJECT { color: #10b981; }
    .score-CONTESTED { color: #fbbf24; }
    .score-NEUTRAL { color: #94a3b8; }

    .score-label {
      font-size: 0.65rem;
      color: #64748b;
    }

    .no-levels {
      text-align: center;
      padding: 1rem;
      color: #64748b;
      font-size: 0.8rem;
      font-style: italic;
    }

    .spy-price-indicator {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem;
      background: rgba(59, 130, 246, 0.1);
      border-radius: 4px;
      border-left: 3px solid #3b82f6;
      margin-bottom: 0.5rem;
    }

    .spy-icon {
      width: 2.5rem;
      height: 2.5rem;
      border-radius: 50%;
      background: #3b82f6;
      color: white;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 0.875rem;
    }

    .spy-details {
      display: flex;
      flex-direction: column;
      gap: 0.125rem;
    }

    .spy-label {
      font-size: 0.7rem;
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .spy-value {
      font-size: 1rem;
      font-weight: 700;
      color: #60a5fa;
      font-variant-numeric: tabular-nums;
    }
  `],
  template: `
    <div class="level-strip">
      <div class="strip-header">Nearest Levels</div>

      @if (levelsData()) {
        <div class="spy-price-indicator">
          <div class="spy-icon">SPY</div>
          <div class="spy-details">
            <div class="spy-label">Current Price</div>
            <div class="spy-value">{{ levelsData()!.spy.spot | number:'1.2-2' }}</div>
          </div>
        </div>
      }

      @if (nearestLevels().length > 0) {
        <div class="levels-container">
          @for (level of nearestLevels(); track level.id) {
            <div 
              class="level-item"
              [ngClass]="'signal-' + level.signal"
            >
              <div 
                class="level-badge"
                [ngClass]="'badge-' + level.signal"
              >
                {{ level.break_score_smooth }}
              </div>

              <div class="level-info">
                <div class="level-name">{{ level.id }}</div>
                <div class="level-price">
                  ${{ level.price | number:'1.2-2' }}
                </div>
                <div 
                  class="level-direction"
                  [ngClass]="'direction-' + level.direction"
                >
                  {{ level.direction }}
                </div>
              </div>

              <div class="level-distance">
                <div class="distance-value">
                  {{ level.distance | number:'1.2-2' }}
                </div>
                <div class="distance-label">away</div>
              </div>

              <div class="level-score">
                <div class="score-value" [ngClass]="'score-' + level.signal">
                  {{ level.signal === 'BREAK' ? '🔴' : level.signal === 'REJECT' ? '🟢' : level.signal === 'CONTESTED' ? '🟡' : '⚪' }}
                </div>
                <div class="score-label">{{ level.signal }}</div>
              </div>
            </div>
          }
        </div>
      } @else {
        <div class="no-levels">
          No nearby levels
        </div>
      }
    </div>
  `
})
export class LevelStripComponent {
  @Input() maxLevels: number = 5;

  public dataStream = inject(DataStreamService);
  public levelsData = computed(() => this.dataStream.levelsData());

  // Get the N nearest levels, sorted by absolute distance
  public nearestLevels = computed(() => {
    const data = this.levelsData();
    if (!data || !data.levels) return [];
    
    return [...data.levels]
      .sort((a, b) => Math.abs(a.distance) - Math.abs(b.distance))
      .slice(0, this.maxLevels);
  });
}

