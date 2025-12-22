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
    .kind-GAMMA_WALL { background: #f59e0b; color: white; }
    .kind-USER { background: #ec4899; color: white; }
    .kind-SESSION_HIGH, .kind-SESSION_LOW { background: #10b981; color: white; }

    .direction-SUPPORT { color: #10b981; }
    .direction-RESISTANCE { color: #ef4444; }

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
    .score-REJECT { color: #10b981; }
    .score-CONTESTED { color: #fbbf24; }
    .score-NEUTRAL { color: #94a3b8; }

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

    .signal-REJECT {
      background: #10b981;
      color: white;
    }

    .signal-CONTESTED {
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

    .runway-container {
      display: flex;
      flex-direction: column;
      gap: 0.125rem;
    }

    .runway-distance {
      font-weight: 600;
    }

    .runway-quality {
      font-size: 0.7rem;
      color: #94a3b8;
    }

    .runway-CLEAR { color: #10b981; }
    .runway-OBSTRUCTED { color: #fbbf24; }

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
              <th class="numeric">Runway</th>
            </tr>
          </thead>
          <tbody>
            @for (level of sortedLevels(); track level.id) {
              <tr>
                <!-- Level -->
                <td>
                  <div class="level-id">{{ level.id }}</div>
                  <div class="level-price">{{ level.price | number:'1.2-2' }}</div>
                  <div class="level-kind" [ngClass]="'kind-' + level.kind">
                    {{ level.kind }}
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
                  <div class="barrier-state" [ngClass]="'barrier-' + level.barrier.state">
                    {{ level.barrier.state }}
                  </div>
                  <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 0.125rem;">
                    Δ{{ level.barrier.delta_liq | number:'1.0-0' }}
                  </div>
                </td>

                <!-- Fuel Effect -->
                <td>
                  <div class="fuel-effect" [ngClass]="'fuel-' + level.fuel.effect">
                    {{ level.fuel.effect }}
                  </div>
                  <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 0.125rem;">
                    γ{{ level.fuel.net_dealer_gamma | number:'1.0-0' }}
                  </div>
                </td>

                <!-- Tape Velocity -->
                <td class="numeric">
                  <div [style.color]="level.tape.velocity >= 0 ? '#10b981' : '#ef4444'">
                    {{ level.tape.velocity | number:'1.3-3' }}
                  </div>
                  @if (level.tape.sweep.detected) {
                    <div style="font-size: 0.7rem; color: #f59e0b; font-weight: 600;">
                      SWEEP
                    </div>
                  }
                </td>

                <!-- Runway -->
                <td class="numeric">
                  <div class="runway-container">
                    <div class="runway-distance">
                      {{ level.runway.distance | number:'1.2-2' }}
                    </div>
                    <div class="runway-quality" [ngClass]="'runway-' + level.runway.quality">
                      {{ level.runway.quality }}
                    </div>
                  </div>
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

