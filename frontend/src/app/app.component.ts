import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FlowDashboardComponent } from './flow-dashboard/flow-dashboard.component';

@Component({
    selector: 'app-root',
    standalone: true,
    imports: [CommonModule, FlowDashboardComponent],
    template: `
    <div class="min-h-screen bg-gray-950 text-white font-sans">
      <header class="p-4 border-b border-gray-800 bg-gray-900/50 backdrop-blur">
        <h1 class="text-xl font-black tracking-tight text-white">
          <span class="text-green-500">SPY</span>MASTER <span class="text-xs text-gray-500 font-mono">0DTE FLOW</span>
        </h1>
      </header>
      <main>
        <app-flow-dashboard></app-flow-dashboard>
      </main>
    </div>
  `,
    styles: []
})
export class AppComponent {
    title = 'frontend';
}
