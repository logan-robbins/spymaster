import { Component, computed, inject } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { DataStreamService, LevelSignal } from '../data-stream.service';

@Component({
  selector: 'app-level-table',
  standalone: true,
  imports: [CommonModule, DecimalPipe],
  styles: [`
    .level-table-container {
      background: #1a202c;
      border-radius: 8px;
      padding: 1rem;
      height: 100%;
      overflow-y: auto;
    }

    .connection-status {
      padding: 0.5rem;
      margin-bottom: 1rem;
      border-radius: 4px;
      text-align: center;
      font-size: 0.875rem;
      font-weight: 500;
    }

    .status-connecting {
      background: #fbbf24;
      color: #78350f;
    }

    .status-connected {
      background: #10b981;
      color: #064e3b;
    }

    .status-disconnected {
      background: #ef4444;
      color: #7f1d1d;
    }

    .table-header {
      margin-bottom: 1rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .table-title {
      font-size: 1.25rem;
      font-weight: 600;
      color: #e2e8f0;
    }

    .spy-info {
      font-size: 0.875rem;
      color: #cbd5e0;
    }

    .spy-price {
      font-weight: 600;
      color: #60a5fa;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.875rem;
    }

    thead {
      position: sticky;
      top: 0;
      background: #2d3748;
      z-index: 10;
    }

    th {
      padding: 0.75rem 0.5rem;
      text-align: left;
      font-weight: 600;
      color: #cbd5e0;
      border-bottom: 2px solid #4a5568;
      white-space: nowrap;
    }

    th.numeric {
      text-align: right;
    }

    td {
      padding: 0.5rem 0.5rem;
      border-bottom: 1px solid #374151;
    }

    td.numeric {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }

    tr:hover {
      background: #2d3748;
    }

    .level-id {
      font-weight: 600;
      color: #e2e8f0;
    }

    .level-price {
      color: #cbd5e0;
      font-size: 0.8rem;
    }

    .level-kind {
      display: inline-block;
      padding: 0.125rem 0.375rem;
      border-radius: 3px;
      font-size: 0.7rem;
      font-weight: 600;
      margin-top: 0.25rem;
    }

    .kind-STRIKE { background: #3b82f6; color: white; }
    .kind-ROUND { background: #8b5cf6; color: white; }
    .kind-VWAP { background: #06b6d4; color: white; }
    .kind-CALL_WALL, .kind-PUT_WALL { background: #f59e0b; color: white; }
    .kind-PM_HIGH, .kind-PM_LOW { background: #10b981; color: white; }
    .kind-OR_HIGH, .kind-OR_LOW { background: #14b8a6; color: white; }
    .kind-SESSION_HIGH, .kind-SESSION_LOW { background: #22c55e; color: white; }
    .kind-SMA_200, .kind-SMA_400 { background: #64748b; color: white; }

    .direction-DOWN { color: #10b981; }
    .direction-UP { color: #ef4444; }

    .score-container {
      display: flex;
      flex-direction: column;
      gap: 0.125rem;
    }

    .score-raw {
      font-weight: 600;
      font-size: 1rem;
    }

    .score-smooth {
      font-size: 0.75rem;
      color: #94a3b8;
    }

    .score-BREAK { color: #ef4444; }
    .score-BOUNCE { color: #10b981; }
    .score-CHOP { color: #fbbf24; }

    .signal-badge {
      display: inline-block;
      padding: 0.25rem 0.5rem;
      border-radius: 4px;
      font-weight: 600;
      font-size: 0.75rem;
    }

    .signal-BREAK {
      background: #ef4444;
      color: white;
    }

    .signal-BOUNCE {
      background: #10b981;
      color: white;
    }

    .signal-CHOP {
      background: #fbbf24;
      color: #78350f;
    }

    .signal-NEUTRAL {
      background: #4a5568;
      color: #e2e8f0;
    }

    .confidence-HIGH { font-weight: 700; }
    .confidence-MEDIUM { font-weight: 500; }
    .confidence-LOW { font-weight: 400; opacity: 0.7; }

    .barrier-state {
      font-weight: 600;
      font-size: 0.75rem;
    }

    .barrier-VACUUM { color: #ef4444; }
    .barrier-WALL { color: #10b981; }
    .barrier-ABSORPTION { color: #06b6d4; }
    .barrier-CONSUMED { color: #f59e0b; }
    .barrier-WEAK { color: #fbbf24; }
    .barrier-NEUTRAL { color: #94a3b8; }

    .fuel-effect {
      font-weight: 600;
      font-size: 0.75rem;
    }

    .fuel-AMPLIFY { color: #ef4444; }
    .fuel-DAMPEN { color: #10b981; }
    .fuel-NEUTRAL { color: #94a3b8; }


    .no-data {
      text-align: center;
      padding: 2rem;
      color: #94a3b8;
      font-style: italic;
    }
  `],
  template: `
    <div class="level-table-container">
      <div 
        class="connection-status"
        [ngClass]="{
          'status-connecting': dataStream.connectionStatus() === 'connecting',
          'status-connected': dataStream.connectionStatus() === 'connected',
          'status-disconnected': dataStream.connectionStatus() === 'disconnected'
        }"
      >
        {{ statusText() }}
      </div>

      <div class="table-header">
        <div class="table-title">Level Signals</div>
        @if (levelsData()) {
          <div class="spy-info">
            SPY: <span class="spy-price">{{ levelsData()!.spy.spot | number:'1.2-2' }}</span>
            ({{ levelsData()!.spy.bid | number:'1.2-2' }} / {{ levelsData()!.spy.ask | number:'1.2-2' }})
          </div>
        }
      </div>

      @if (sortedLevels().length > 0) {
        <table>
          <thead>
            <tr>
              <th>Level</th>
              <th>Direction</th>
              <th class="numeric">Distance</th>
              <th class="numeric">Score</th>
              <th>Signal</th>
              <th>Barrier</th>
              <th>Fuel</th>
              <th class="numeric">Tape Vel</th>
            </tr>
          </thead>
          <tbody>
            @for (level of sortedLevels(); track level.id) {
              <tr>
                <!-- Level -->
                <td>
                  <div class="level-id">{{ level.id }}</div>
                  <div class="level-price">{{ level.level_price | number:'1.2-2' }}</div>
                  <div class="level-kind" [ngClass]="'kind-' + level.level_kind_name">
                    {{ level.level_kind_name }}
                  </div>
                </td>

                <!-- Direction -->
                <td [ngClass]="'direction-' + level.direction">
                  {{ level.direction }}
                </td>

                <!-- Distance -->
                <td class="numeric">
                  {{ level.distance | number:'1.2-2' }}
                </td>

                <!-- Score -->
                <td class="numeric">
                  <div class="score-container">
                    <div class="score-raw" [ngClass]="'score-' + level.signal">
                      {{ level.break_score_raw }}
                    </div>
                    <div class="score-smooth">
                      ({{ level.break_score_smooth }})
                    </div>
                  </div>
                </td>

                <!-- Signal -->
                <td>
                  <span 
                    class="signal-badge"
                    [ngClass]="['signal-' + level.signal, 'confidence-' + level.confidence]"
                  >
                    {{ level.signal }}
                  </span>
                </td>

                <!-- Barrier State -->
                <td>
                  <div class="barrier-state" [ngClass]="'barrier-' + level.barrier_state">
                    {{ level.barrier_state }}
                  </div>
                  <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 0.125rem;">
                    Δ{{ level.barrier_delta_liq | number:'1.0-0' }}
                  </div>
                </td>

                <!-- Fuel Effect -->
                <td>
                  <div class="fuel-effect" [ngClass]="'fuel-' + level.fuel_effect">
                    {{ level.fuel_effect }}
                  </div>
                  <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 0.125rem;">
                    γ{{ level.gamma_exposure | number:'1.0-0' }}
                  </div>
                </td>

                <!-- Tape Velocity -->
                <td class="numeric">
                  <div [style.color]="level.tape_velocity >= 0 ? '#10b981' : '#ef4444'">
                    {{ level.tape_velocity | number:'1.3-3' }}
                  </div>
                  @if (level.sweep_detected) {
                    <div style="font-size: 0.7rem; color: #f59e0b; font-weight: 600;">
                      SWEEP
                    </div>
                  }
                </td>
              </tr>
            }
          </tbody>
        </table>
      } @else {
        <div class="no-data">
          {{ noDataMessage() }}
        </div>
      }
    </div>
  `
})
export class LevelTableComponent {
  public dataStream = inject(DataStreamService);

  public levelsData = computed(() => this.dataStream.levelsData());

  // Sort levels by absolute distance (nearest first)
  public sortedLevels = computed(() => {
    const data = this.levelsData();
    if (!data || !data.levels) return [];
    
    return [...data.levels].sort((a, b) => 
      Math.abs(a.distance) - Math.abs(b.distance)
    );
  });

  public statusText = computed(() => {
    const status = this.dataStream.connectionStatus();
    switch (status) {
      case 'connecting': return '🔄 Connecting to Stream...';
      case 'connected': return '✅ Connected';
      case 'disconnected': return '⚠️ Disconnected - Retrying...';
      default: return 'Unknown';
    }
  });

  public noDataMessage = computed(() => {
    const status = this.dataStream.connectionStatus();
    if (status === 'disconnected') {
      return 'Waiting for backend connection...';
    } else if (status === 'connecting') {
      return 'Connecting...';
    } else {
      return 'No level signals available';
    }
  });
}
