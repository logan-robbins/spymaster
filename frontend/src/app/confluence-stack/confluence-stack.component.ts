import { Component, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { LevelDerivedService } from '../level-derived.service';

@Component({
  selector: 'app-confluence-stack',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="confluence">
      <div class="header">
        <div class="title">Confluence Stack</div>
        <div class="meta">Band {{ band() | number:'1.2-2' }}</div>
      </div>

      @if (groups().length) {
        <div class="group-list">
          @for (group of groups(); track group.id) {
            <div class="group-card" [ngClass]="'bias-' + group.bias">
              <div class="group-header">
                <div class="group-price">{{ group.centerPrice | number:'1.2-2' }}</div>
                <div class="group-strength">{{ group.strength }}%</div>
              </div>
              <div class="group-meta">
                <span class="group-bias">{{ group.bias }}</span>
                <span class="group-score">Score {{ group.score | number:'1.1-1' }}</span>
              </div>
              <div class="group-levels">
                @for (level of group.levels; track level.id) {
                  <span class="level-chip">{{ level.kind }}</span>
                }
              </div>
            </div>
          }
        </div>
      } @else {
        <div class="empty">Waiting for confluence data...</div>
      }
    </div>
  `,
  styles: [`
    .confluence {
      background: #0f172a;
      border: 1px solid #233047;
      border-radius: 16px;
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.8rem;
    }

    .header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 1rem;
    }

    .title {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 0.85rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: #e2e8f0;
    }

    .meta {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.7rem;
      color: #94a3b8;
    }

    .group-list {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      max-height: 320px;
      overflow-y: auto;
    }

    .group-card {
      padding: 0.75rem;
      border-radius: 12px;
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(148, 163, 184, 0.2);
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .group-card.bias-BREAK {
      border-color: rgba(248, 113, 113, 0.4);
    }

    .group-card.bias-BOUNCE {
      border-color: rgba(34, 197, 94, 0.4);
    }

    .group-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: 'IBM Plex Mono', monospace;
    }

    .group-price {
      font-size: 1rem;
      font-weight: 700;
      color: #f8fafc;
    }

    .group-strength {
      font-size: 0.9rem;
      color: #38bdf8;
    }

    .group-meta {
      display: flex;
      justify-content: space-between;
      font-size: 0.65rem;
      text-transform: uppercase;
      letter-spacing: 0.15em;
      color: #94a3b8;
    }

    .group-levels {
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
    }

    .level-chip {
      font-size: 0.6rem;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      padding: 0.2rem 0.4rem;
      border-radius: 999px;
      background: rgba(56, 189, 248, 0.15);
      color: #e2e8f0;
    }

    .empty {
      font-size: 0.75rem;
      color: #64748b;
    }
  `]
})
export class ConfluenceStackComponent {
  private derived = inject(LevelDerivedService);

  public groups = computed(() => this.derived.confluenceGroups().slice(0, 5));
  public band = this.derived.getConfluenceBand();
}
