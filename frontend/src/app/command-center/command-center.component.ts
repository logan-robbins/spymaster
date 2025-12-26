import { Component, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DataStreamService } from '../data-stream.service';
import { LevelDerivedService } from '../level-derived.service';
import { SmartChartComponent } from '../smart-chart/smart-chart.component';
import { ChartControlsComponent } from '../chart-controls/chart-controls.component';
import { LevelDetailPanelComponent } from '../level-detail-panel/level-detail-panel.component';
import { OptionsPanelComponent } from '../options-panel/options-panel.component';
import { ViewportSelectorComponent } from '../viewport-selector/viewport-selector.component';

@Component({
  selector: 'app-command-center',
  standalone: true,
  imports: [
    CommonModule,
    SmartChartComponent,
    ChartControlsComponent,
    LevelDetailPanelComponent,
    OptionsPanelComponent,
    ViewportSelectorComponent
  ],
  template: `
    <div class="command-center">
      <header class="command-header">
        <div class="brand">
          <div class="brand-title">SPYMASTER <span class="v2-badge">V2</span></div>
          <div class="brand-subtitle">0DTE BOUNCE/BREAK COMMAND</div>
        </div>
        <div class="status">
          <div class="status-pill" [ngClass]="'status-' + connectionStatus()">
            {{ statusText() }}
          </div>
          @if (spy()) {
            <div class="spy-price">SPY {{ spy()!.spot | number:'1.2-2' }}</div>
          }
        </div>
      </header>

      <!-- V2 Layout: 2 Columns (2fr Chart, 1fr Details) -->
      <div class="command-grid v2-layout">
        <!-- Main Chart Section -->
        <section class="panel chart-panel">
          <div class="chart-wrapper">
             <app-smart-chart></app-smart-chart>
          </div>
          <div class="controls-wrapper">
             <app-chart-controls></app-chart-controls>
          </div>
        </section>

        <!-- Right Detail Column -->
        <section class="panel right-panel">
          <!-- ML Viewport / Level Details -->
          <div class="detail-section flex-1">
             <app-level-detail-panel></app-level-detail-panel>
          </div>
          
          <!-- Options Activity (Secondary) -->
          <div class="options-section flex-1 mt-4">
             <app-options-panel></app-options-panel>
          </div>
        </section>
      </div>
    </div>
  `,
  styles: [`
    .command-center {
      height: 100vh;
      padding: 1.5rem;
      background: radial-gradient(circle at 15% 20%, rgba(56, 189, 248, 0.12), transparent 40%),
        radial-gradient(circle at 80% 10%, rgba(248, 113, 113, 0.08), transparent 35%),
        #0b1120;
      color: #e2e8f0;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
      overflow: hidden;
    }

    .command-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1.5rem;
      flex-wrap: wrap;
    }

    .brand-title {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 1.6rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .v2-badge {
       background: #ef4444;
       color: white;
       font-size: 0.7rem;
       padding: 0.1rem 0.4rem;
       border-radius: 4px;
       vertical-align: middle;
    }

    .brand-subtitle {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.3em;
      color: #94a3b8;
      margin-top: 0.35rem;
    }

    .status {
      display: flex;
      align-items: center;
      gap: 1rem;
      font-family: 'IBM Plex Mono', monospace;
    }

    .status-pill {
      padding: 0.35rem 0.7rem;
      border-radius: 999px;
      font-size: 0.7rem;
      letter-spacing: 0.2em;
      text-transform: uppercase;
    }

    .status-connecting {
      background: rgba(251, 191, 36, 0.15);
      color: #fbbf24;
      border: 1px solid rgba(251, 191, 36, 0.4);
    }

    .status-connected {
      background: rgba(34, 197, 94, 0.15);
      color: #22c55e;
      border: 1px solid rgba(34, 197, 94, 0.4);
    }

    .status-disconnected {
      background: rgba(248, 113, 113, 0.15);
      color: #f87171;
      border: 1px solid rgba(248, 113, 113, 0.4);
    }

    .spy-price {
      font-size: 1rem;
      font-weight: 700;
      color: #38bdf8;
    }

    /* V2 Grid Layout */
    .command-grid.v2-layout {
      display: grid;
      grid-template-columns: 2fr 1fr; /* 2:1 Ratio */
      gap: 1.5rem;
      align-items: stretch;
      flex: 1;
      min-height: 0;
    }

    .panel {
      display: flex;
      flex-direction: column;
      min-height: 0;
      overflow: hidden;
      border: 1px solid rgba(148, 163, 184, 0.1);
      border-radius: 12px;
      background: rgba(15, 23, 42, 0.4);
    }
    
    .chart-panel {
       display: flex;
       flex-direction: column;
    }
    
    .chart-wrapper {
       flex: 1;
       min-height: 0;
       position: relative;
    }
    
    .controls-wrapper {
       height: auto;
    }

    .right-panel {
       display: flex;
       flex-direction: column;
       gap: 1rem;
       background: transparent;
       border: none;
       padding: 0; 
    }
    
    .detail-section, .options-section {
       background: rgba(15, 23, 42, 0.8);
       border: 1px solid rgba(148, 163, 184, 0.2);
       border-radius: 16px;
       overflow: hidden;
       position: relative;
    }

    @media (max-width: 1024px) {
      .command-grid.v2-layout {
        grid-template-columns: 1fr;
        grid-template-rows: 1fr 1fr;
      }
    }
  `]
})
export class CommandCenterComponent {
  private dataStream = inject(DataStreamService);
  private derived = inject(LevelDerivedService);

  public spy = computed(() => this.derived.spy());
  public connectionStatus = this.dataStream.connectionStatus;
  public dataStatus = this.dataStream.dataStatus;

  public statusText = computed(() => {
    const status = this.connectionStatus();
    if (status === 'connected') {
      return this.dataStatus() === 'ok' ? 'Stream OK' : 'Data unavailable';
    }
    if (status === 'connecting') return 'Connecting';
    return 'Offline';
  });
}
