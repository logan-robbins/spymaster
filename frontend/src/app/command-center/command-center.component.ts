import { Component, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DataStreamService } from '../data-stream.service';
import { LevelDerivedService } from '../level-derived.service';
import { PriceLadderComponent } from '../price-ladder/price-ladder.component';
import { StrengthCockpitComponent } from '../strength-cockpit/strength-cockpit.component';
import { AttributionBarComponent } from '../attribution-bar/attribution-bar.component';
import { ConfluenceStackComponent } from '../confluence-stack/confluence-stack.component';
import { OptionsPanelComponent } from '../options-panel/options-panel.component';

@Component({
  selector: 'app-command-center',
  standalone: true,
  imports: [
    CommonModule,
    PriceLadderComponent,
    StrengthCockpitComponent,
    AttributionBarComponent,
    ConfluenceStackComponent,
    OptionsPanelComponent
  ],
  template: `
    <div class="command-center">
      <header class="command-header">
        <div class="brand">
          <div class="brand-title">SPYMASTER</div>
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

      <div class="command-grid">
        <section class="panel ladder-panel">
          <app-price-ladder [range]="6"></app-price-ladder>
        </section>

        <section class="panel cockpit-panel">
          <app-strength-cockpit></app-strength-cockpit>
          <app-attribution-bar></app-attribution-bar>
          <app-confluence-stack></app-confluence-stack>
        </section>

        <section class="panel options-panel">
          <app-options-panel></app-options-panel>
        </section>
      </div>
    </div>
  `,
  styles: [`
    .command-center {
      min-height: 100vh;
      padding: 1.5rem;
      background: radial-gradient(circle at 15% 20%, rgba(56, 189, 248, 0.12), transparent 40%),
        radial-gradient(circle at 80% 10%, rgba(248, 113, 113, 0.08), transparent 35%),
        #0b1120;
      color: #e2e8f0;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
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

    .command-grid {
      display: grid;
      grid-template-columns: minmax(280px, 1fr) minmax(320px, 1.1fr) minmax(300px, 1fr);
      gap: 1.5rem;
      align-items: stretch;
    }

    .panel {
      display: flex;
      flex-direction: column;
      gap: 1rem;
      min-height: 0;
    }

    .options-panel {
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 16px;
      padding: 1rem;
    }

    @media (max-width: 1200px) {
      .command-grid {
        grid-template-columns: 1fr;
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
