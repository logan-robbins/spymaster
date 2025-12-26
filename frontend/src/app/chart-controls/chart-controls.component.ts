import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
    selector: 'app-chart-controls',
    standalone: true,
    imports: [CommonModule],
    template: `
    <div class="controls-bar">
       <div class="control-group">
          <label>Timeframe</label>
          <div class="toggle-group">
             <button class="active">2m</button>
             <button>5m</button>
             <button>15m</button>
          </div>
       </div>
       
       <div class="control-group">
          <label>Overlays</label>
          <div class="checks">
             <label><input type="checkbox" checked> GEX Levels</label>
             <label><input type="checkbox" checked> VWAP</label>
          </div>
       </div>
    </div>
  `,
    styles: [`
    .controls-bar {
       display: flex;
       gap: 2rem;
       padding: 1rem;
       background: rgba(15, 23, 42, 0.8);
       border-top: 1px solid rgba(51, 65, 85, 0.5);
       font-family: 'IBM Plex Mono', monospace;
       font-size: 0.75rem;
    }
    .control-group {
       display: flex;
       flex-direction: column;
       gap: 0.5rem;
    }
    label {
        color: #64748b;
        text-transform: uppercase;
        font-size: 0.65rem;
        letter-spacing: 0.1em;
    }
    .toggle-group {
        display: flex;
        background: #1e293b;
        border-radius: 4px;
        padding: 2px;
    }
    button {
       background: transparent;
       border: none;
       color: #94a3b8;
       padding: 0.3rem 0.6rem;
       font-size: 0.75rem;
       cursor: pointer;
       border-radius: 2px;
    }
    button.active {
       background: #334155;
       color: #fff;
    }
    .checks {
       display: flex;
       gap: 1rem;
       color: #e2e8f0;
    }
  `]
})
export class ChartControlsComponent { }
