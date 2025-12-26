import { Component, input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
    selector: 'app-level-detail-panel',
    standalone: true,
    imports: [CommonModule],
    template: `
    <div class="detail-panel">
       <header class="panel-header">
         <h3>ML VIEWPORT</h3>
         <span class="badge">AUTO</span>
       </header>
       
       <div class="content-placeholder">
          <p class="text-xs text-slate-400">Select a level to view details</p>
          
          <!-- Mocking the UI from Screenshot -->
          <div class="level-card mt-4">
             <div class="flex justify-between items-center mb-2">
                <span class="text-xl font-bold text-white">VWAP</span>
                <span class="status-wait">WAIT</span>
             </div>
             <div class="text-3xl text-white font-mono mb-2">$584.76</div>
             
             <div class="grid grid-cols-2 gap-2 text-xs text-slate-400">
                <div>TRADEABLE</div>
                <div class="text-right text-blue-400">50% UTILITY</div>
             </div>
          </div>
          
          <div class="strength-cockpit mt-6">
             <h4 class="text-xs uppercase text-slate-500 mb-2">Setup Quality</h4>
             <div class="p-3 bg-slate-800 rounded border border-slate-700">
                <div class="flex justify-between items-center">
                    <span class="text-yellow-400 font-bold">EXTENDED</span>
                    <span class="text-xs text-slate-400">Pressure 35%</span>
                </div>
             </div>
          </div>
       </div>
    </div>
  `,
    styles: [`
    .detail-panel {
      height: 100%;
      background: rgba(15, 23, 42, 0.6);
      border-left: 1px solid rgba(148, 163, 184, 0.1);
      padding: 1rem;
    }
    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.5rem;
    }
    h3 {
       font-size: 0.8rem;
       font-weight: 700;
       letter-spacing: 0.1em;
       color: #fca5a5; /* red-ish */
    }
    .badge {
        font-size: 0.65rem;
        background: #1e293b;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        color: #94a3b8;
    }
    .level-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(51, 65, 85, 0.5);
        border-radius: 12px;
        padding: 1rem;
    }
    .status-wait {
        background: rgba(234, 179, 8, 0.2);
        color: #fbbf24;
        font-size: 0.7rem;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 700;
    }
  `]
})
export class LevelDetailPanelComponent {
    // Logic to come later
}
