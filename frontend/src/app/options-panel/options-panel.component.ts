import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { StrikeGridComponent } from '../strike-grid/strike-grid.component';
import { FlowChartComponent } from '../flow-chart/flow-chart.component';

type OptionsTab = 'strikes' | 'flow';

@Component({
  selector: 'app-options-panel',
  standalone: true,
  imports: [CommonModule, StrikeGridComponent, FlowChartComponent],
  template: `
    <div class="options-panel">
      <div class="options-header">
        <div class="title">Options Activity</div>
        <div class="tabs">
          <button class="tab" [class.active]="activeTab() === 'strikes'" (click)="activeTab.set('strikes')">
            Strikes
          </button>
          <button class="tab" [class.active]="activeTab() === 'flow'" (click)="activeTab.set('flow')">
            Flow
          </button>
        </div>
      </div>

      <div class="options-body">
        @if (activeTab() === 'strikes') {
          <app-strike-grid></app-strike-grid>
        } @else {
          <app-flow-chart></app-flow-chart>
        }
      </div>
    </div>
  `,
  styles: [`
    .options-panel {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      height: 100%;
    }

    .options-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
    }

    .title {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 0.9rem;
      letter-spacing: 0.2em;
      text-transform: uppercase;
      color: #e2e8f0;
    }

    .tabs {
      display: inline-flex;
      border: 1px solid rgba(148, 163, 184, 0.25);
      border-radius: 999px;
      overflow: hidden;
      background: rgba(15, 23, 42, 0.7);
    }

    .tab {
      background: transparent;
      border: none;
      color: #94a3b8;
      padding: 0.35rem 0.8rem;
      font-size: 0.7rem;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .tab.active {
      color: #0f172a;
      background: #38bdf8;
    }

    .options-body {
      flex: 1;
      min-height: 0;
      overflow: hidden;
    }
  `]
})
export class OptionsPanelComponent {
  public activeTab = signal<OptionsTab>('strikes');
}
