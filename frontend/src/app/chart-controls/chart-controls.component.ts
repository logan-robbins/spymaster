import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ChartSettingsService } from '../chart-settings.service';

@Component({
  selector: 'app-chart-controls',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="controls-bar">
      <!-- Timeframe Selection -->
      <div class="control-group">
        <label class="group-label">Timeframe</label>
        <div class="toggle-group">
          <button 
            [class.active]="timeframe() === '2m'" 
            (click)="timeframe.set('2m')">
            2m
          </button>
          <button 
            [class.active]="timeframe() === '5m'" 
            (click)="timeframe.set('5m')">
            5m
          </button>
          <button 
            [class.active]="timeframe() === '15m'" 
            (click)="timeframe.set('15m')">
            15m
          </button>
        </div>
      </div>
      
      <!-- Overlay Toggles -->
      <div class="control-group">
        <label class="group-label">Overlays</label>
        <div class="checks">
          <label class="check-label">
            <input 
              type="checkbox" 
              [checked]="showGEX()" 
              (change)="showGEX.set(!showGEX())"
            />
            <span>GEX Levels</span>
          </label>
          <label class="check-label">
            <input 
              type="checkbox" 
              [checked]="showVWAP()" 
              (change)="showVWAP.set(!showVWAP())"
            />
            <span>VWAP</span>
          </label>
          <label class="check-label">
            <input 
              type="checkbox" 
              [checked]="showSMAs()" 
              (change)="showSMAs.set(!showSMAs())"
            />
            <span>SMAs</span>
          </label>
        </div>
      </div>

      <!-- Level Filters -->
      <div class="control-group">
        <label class="group-label">Level Filters</label>
        <div class="checks">
          <label class="check-label">
            <input 
              type="checkbox" 
              [checked]="showWalls()" 
              (change)="showWalls.set(!showWalls())"
            />
            <span>Walls</span>
          </label>
          <label class="check-label">
            <input 
              type="checkbox" 
              [checked]="showStructural()" 
              (change)="showStructural.set(!showStructural())"
            />
            <span>Structural</span>
          </label>
        </div>
      </div>

      <!-- Confluence Filter -->
      <div class="control-group">
        <label class="group-label">Min Confluence</label>
        <div class="slider-group">
          <input 
            type="range" 
            min="0" 
            max="10" 
            [value]="minConfluence()" 
            (input)="minConfluence.set(+$any($event.target).value)"
            class="confluence-slider"
          />
          <span class="slider-value">{{ minConfluence() }}/10</span>
        </div>
      </div>

      <!-- Tradeability Filter -->
      <div class="control-group">
        <label class="group-label">Min Tradeability</label>
        <div class="slider-group">
          <input 
            type="range" 
            min="0" 
            max="100" 
            step="10"
            [value]="minTradeability()" 
            (input)="minTradeability.set(+$any($event.target).value)"
            class="tradeability-slider"
          />
          <span class="slider-value">{{ minTradeability() }}%</span>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .controls-bar {
      display: flex;
      gap: 1.5rem;
      padding: 0.75rem 1rem;
      background: rgba(15, 23, 42, 0.8);
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.75rem;
      flex-wrap: wrap;
    }

    .control-group {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }

    .group-label {
      color: #64748b;
      text-transform: uppercase;
      font-size: 0.6rem;
      letter-spacing: 0.1em;
      font-weight: 600;
    }

    .toggle-group {
      display: flex;
      background: #1e293b;
      border-radius: 6px;
      padding: 2px;
      border: 1px solid rgba(148, 163, 184, 0.2);
    }

    button {
      background: transparent;
      border: none;
      color: #94a3b8;
      padding: 0.3rem 0.7rem;
      font-size: 0.7rem;
      cursor: pointer;
      border-radius: 4px;
      transition: all 0.2s ease;
      font-family: 'IBM Plex Mono', monospace;
      font-weight: 600;
    }

    button:hover {
      color: #cbd5e1;
      background: rgba(51, 65, 85, 0.5);
    }

    button.active {
      background: #334155;
      color: #f8fafc;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    .checks {
      display: flex;
      gap: 1rem;
      flex-wrap: wrap;
    }

    .check-label {
      display: flex;
      align-items: center;
      gap: 0.35rem;
      color: #cbd5e1;
      cursor: pointer;
      font-size: 0.7rem;
      transition: color 0.2s ease;
    }

    .check-label:hover {
      color: #f8fafc;
    }

    .check-label input[type="checkbox"] {
      width: 14px;
      height: 14px;
      accent-color: #38bdf8;
      cursor: pointer;
    }

    .slider-group {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    input[type="range"] {
      width: 100px;
      height: 6px;
      accent-color: #38bdf8;
      cursor: pointer;
      border-radius: 3px;
    }

    .confluence-slider {
      accent-color: #22c55e;
    }

    .tradeability-slider {
      accent-color: #fbbf24;
    }

    .slider-value {
      font-size: 0.7rem;
      color: #e2e8f0;
      font-weight: 600;
      min-width: 45px;
      text-align: right;
    }
  `]
})
export class ChartControlsComponent {
  private settings = inject(ChartSettingsService);

  // Expose signals for the template
  public timeframe = this.settings.timeframe;
  public showGEX = this.settings.showGEX;
  public showVWAP = this.settings.showVWAP;
  public showSMAs = this.settings.showSMAs;
  public showWalls = this.settings.showWalls;
  public showStructural = this.settings.showStructural;
  public minConfluence = this.settings.minConfluence;
  public minTradeability = this.settings.minTradeability;
}
