import { Component, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ViewportSelectionService } from '../viewport-selection.service';
import { MLIntegrationService } from '../ml-integration.service';
import { ViewportTarget } from '../data-stream.service';

/**
 * Viewport Selector Component
 * 
 * Displays ML-ranked levels and allows trader to select focus level.
 * 
 * Shows:
 * - Level kind and price
 * - ML tradeability signal (GO/WAIT/NO-GO)
 * - Utility score (ML's confidence in this setup)
 * - Direction bias (BREAK/BOUNCE)
 */
@Component({
  selector: 'app-viewport-selector',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="viewport-selector">
      <div class="selector-header">
        <div class="title">🎯 ML Viewport</div>
        <div class="mode-badge" [ngClass]="'mode-' + selectionMode()">
          {{ selectionMode() }}
        </div>
      </div>
      
      @if (targets().length === 0) {
        <div class="empty-state">
          <div class="empty-icon">📊</div>
          <div class="empty-text">No ML predictions available</div>
          <div class="empty-hint">Train models or enable VIEWPORT_SCORING</div>
        </div>
      } @else {
        <div class="targets-list">
          @for (target of targets(); track target.level_id) {
            <div 
              class="target-card" 
              [ngClass]="{
                'selected': isSelected(target.level_id),
                'pinned': isPinned() && isSelected(target.level_id),
                'tradeable-go': getTradeability(target) === 'GO',
                'tradeable-wait': getTradeability(target) === 'WAIT',
                'tradeable-no': getTradeability(target) === 'NO-GO'
              }"
              (click)="selectTarget(target.level_id)"
              [attr.title]="'Click to focus on this level'"
            >
              <div class="target-header">
                <div class="target-level">
                  <div class="level-kind">{{ target.level_kind_name }}</div>
                  <div class="level-price">\${{ target.level_price | number:'1.2-2' }}</div>
                </div>
                <div class="target-signal" [ngClass]="'signal-' + getTradeability(target).toLowerCase()">
                  {{ getTradeability(target) }}
                </div>
              </div>
              
              <div class="target-metrics">
                <div class="metric">
                  <span class="metric-label">Tradeable</span>
                  <span class="metric-value">{{ (target.p_tradeable_2 * 100) | number:'1.0-0' }}%</span>
                </div>
                <div class="metric">
                  <span class="metric-label">Utility</span>
                  <span class="metric-value">{{ (target.utility_score * 100) | number:'1.0-0' }}%</span>
                </div>
              </div>
              
              <div class="target-direction">
                <div class="direction-badge" [ngClass]="target.direction">
                  {{ target.direction }}
                </div>
                <div class="direction-prob">
                  @if (target.p_break > target.p_bounce) {
                    <span class="prob-label">Break</span>
                    <span class="prob-value">{{ (target.p_break * 100) | number:'1.0-0' }}%</span>
                  } @else {
                    <span class="prob-label">Bounce</span>
                    <span class="prob-value">{{ (target.p_bounce * 100) | number:'1.0-0' }}%</span>
                  }
                </div>
              </div>
              
              @if (isSelected(target.level_id)) {
                <div class="selected-indicator">
                  @if (isPinned()) {
                    <span class="pin-icon">📌</span>
                    <span>Pinned</span>
                  } @else {
                    <span class="focus-icon">●</span>
                    <span>Focused</span>
                  }
                </div>
              }
            </div>
          }
        </div>
        
        @if (selectedTarget()) {
          <div class="actions">
            @if (isPinned()) {
              <button class="action-btn unpin" (click)="togglePin()">
                📌 Unpin
              </button>
            } @else if (selectedLevelId()) {
              <button class="action-btn pin" (click)="togglePin()">
                📍 Pin Focus
              </button>
            }
            @if (selectedLevelId()) {
              <button class="action-btn clear" (click)="clearSelection()">
                ↺ Auto Select
              </button>
            }
          </div>
        }
      }
    </div>
  `,
  styles: [`
    .viewport-selector {
      background: #0f172a;
      border: 1px solid #233047;
      border-radius: 16px;
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      max-height: 400px;
      overflow: hidden;
    }

    .selector-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .title {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 0.9rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #e2e8f0;
    }

    .mode-badge {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.65rem;
      padding: 0.25rem 0.5rem;
      border-radius: 6px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      font-weight: 600;
    }

    .mode-Auto { 
      background: rgba(56, 189, 248, 0.15);
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.3);
    }

    .mode-Manual {
      background: rgba(251, 191, 36, 0.15);
      color: #fbbf24;
      border: 1px solid rgba(251, 191, 36, 0.3);
    }

    .mode-Pinned {
      background: rgba(168, 85, 247, 0.15);
      color: #a855f7;
      border: 1px solid rgba(168, 85, 247, 0.3);
    }

    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.5rem;
      padding: 2rem 1rem;
      color: #64748b;
      text-align: center;
    }

    .empty-icon {
      font-size: 2rem;
      opacity: 0.5;
    }

    .empty-text {
      font-size: 0.85rem;
      color: #94a3b8;
    }

    .empty-hint {
      font-size: 0.7rem;
      font-style: italic;
    }

    .targets-list {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      overflow-y: auto;
      max-height: 300px;
      padding-right: 0.25rem;
    }

    .targets-list::-webkit-scrollbar {
      width: 4px;
    }

    .targets-list::-webkit-scrollbar-track {
      background: rgba(148, 163, 184, 0.1);
      border-radius: 2px;
    }

    .targets-list::-webkit-scrollbar-thumb {
      background: rgba(148, 163, 184, 0.3);
      border-radius: 2px;
    }

    .target-card {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 10px;
      padding: 0.75rem;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .target-card:hover {
      border-color: rgba(56, 189, 248, 0.5);
      background: rgba(15, 23, 42, 0.9);
      transform: translateX(2px);
    }

    .target-card.selected {
      border-color: rgba(56, 189, 248, 0.7);
      background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.7));
      box-shadow: 0 0 20px rgba(56, 189, 248, 0.2);
    }

    .target-card.pinned {
      border-color: rgba(168, 85, 247, 0.7);
      box-shadow: 0 0 20px rgba(168, 85, 247, 0.3);
    }

    .target-card.tradeable-go {
      border-left: 3px solid #22c55e;
    }

    .target-card.tradeable-wait {
      border-left: 3px solid #fbbf24;
    }

    .target-card.tradeable-no {
      border-left: 3px solid #f87171;
    }

    .target-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .target-level {
      display: flex;
      flex-direction: column;
      gap: 0.15rem;
    }

    .level-kind {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.7rem;
      color: #94a3b8;
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }

    .level-price {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 1.1rem;
      font-weight: 700;
      color: #f8fafc;
    }

    .target-signal {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.75rem;
      font-weight: 800;
      padding: 0.3rem 0.5rem;
      border-radius: 6px;
      letter-spacing: 0.15em;
    }

    .signal-go {
      background: rgba(34, 197, 94, 0.2);
      color: #86efac;
      border: 1px solid rgba(34, 197, 94, 0.4);
    }

    .signal-wait {
      background: rgba(251, 191, 36, 0.2);
      color: #fbbf24;
      border: 1px solid rgba(251, 191, 36, 0.4);
    }

    .signal-no-go {
      background: rgba(248, 113, 113, 0.2);
      color: #fca5a5;
      border: 1px solid rgba(248, 113, 113, 0.4);
    }

    .target-metrics {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.5rem;
    }

    .metric {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.7rem;
    }

    .metric-label {
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }

    .metric-value {
      color: #38bdf8;
      font-weight: 600;
    }

    .target-direction {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding-top: 0.25rem;
      border-top: 1px solid rgba(148, 163, 184, 0.1);
    }

    .direction-badge {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.65rem;
      padding: 0.2rem 0.4rem;
      border-radius: 4px;
      letter-spacing: 0.1em;
      font-weight: 600;
    }

    .direction-badge.UP {
      background: rgba(248, 113, 113, 0.15);
      color: #fca5a5;
      border: 1px solid rgba(248, 113, 113, 0.3);
    }

    .direction-badge.DOWN {
      background: rgba(34, 197, 94, 0.15);
      color: #86efac;
      border: 1px solid rgba(34, 197, 94, 0.3);
    }

    .direction-prob {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.7rem;
    }

    .prob-label {
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }

    .prob-value {
      color: #f8fafc;
      font-weight: 600;
    }

    .selected-indicator {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.7rem;
      color: #38bdf8;
      padding-top: 0.5rem;
      border-top: 1px solid rgba(56, 189, 248, 0.3);
      font-weight: 600;
      letter-spacing: 0.1em;
    }

    .pin-icon, .focus-icon {
      font-size: 0.85rem;
    }

    .actions {
      display: flex;
      gap: 0.5rem;
      padding-top: 0.5rem;
      border-top: 1px solid rgba(148, 163, 184, 0.2);
    }

    .action-btn {
      flex: 1;
      padding: 0.5rem 0.75rem;
      border-radius: 8px;
      border: 1px solid rgba(148, 163, 184, 0.3);
      background: rgba(15, 23, 42, 0.6);
      color: #cbd5f5;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.7rem;
      font-weight: 600;
      letter-spacing: 0.1em;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .action-btn:hover {
      border-color: rgba(56, 189, 248, 0.5);
      background: rgba(30, 41, 59, 0.8);
      color: #f8fafc;
    }

    .action-btn.pin {
      border-color: rgba(168, 85, 247, 0.4);
    }

    .action-btn.pin:hover {
      border-color: rgba(168, 85, 247, 0.7);
      background: rgba(168, 85, 247, 0.1);
      color: #e9d5ff;
    }

    .action-btn.unpin {
      border-color: rgba(248, 113, 113, 0.4);
    }

    .action-btn.unpin:hover {
      border-color: rgba(248, 113, 113, 0.7);
      background: rgba(248, 113, 113, 0.1);
      color: #fca5a5;
    }

    .action-btn.clear {
      border-color: rgba(148, 163, 184, 0.3);
    }
  `]
})
export class ViewportSelectorComponent {
  private viewportService = inject(ViewportSelectionService);
  private mlService = inject(MLIntegrationService);
  
  public targets = this.viewportService.viewportTargets;
  public selectedTarget = this.viewportService.selectedTarget;
  public isPinned = this.viewportService.isPinned;
  public selectionMode = this.viewportService.getSelectionMode();
  public selectedLevelId = computed(() => this.selectedTarget()?.level_id ?? null);
  
  public selectTarget(levelId: string): void {
    const currentlySelected = this.selectedTarget();
    
    // If clicking the same level, toggle pin
    if (currentlySelected?.level_id === levelId) {
      this.viewportService.togglePin();
    } else {
      // Select new level (not pinned by default)
      this.viewportService.selectLevel(levelId, false);
    }
  }
  
  public togglePin(): void {
    this.viewportService.togglePin();
  }
  
  public clearSelection(): void {
    this.viewportService.clearSelection();
  }
  
  public isSelected(levelId: string): boolean {
    return this.viewportService.isSelected(levelId);
  }
  
  public getTradeability(target: ViewportTarget): 'GO' | 'WAIT' | 'NO-GO' {
    const p_tradeable = target.p_tradeable_2;
    const direction_confidence = Math.max(target.p_break, target.p_bounce);
    const combined = p_tradeable * direction_confidence;
    
    if (combined >= 0.50 && p_tradeable >= 0.60) {
      return 'GO';
    } else if (combined >= 0.30 && p_tradeable >= 0.40) {
      return 'WAIT';
    } else {
      return 'NO-GO';
    }
  }
}

