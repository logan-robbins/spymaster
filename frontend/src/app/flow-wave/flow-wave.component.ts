import { Component, Input, OnChanges } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
    selector: 'app-flow-wave',
    standalone: true,
    imports: [CommonModule],
    template: `
    <div class="rounded-lg border-2 border-gray-700 bg-gray-900/60 p-4">
      <div class="mb-2 flex items-center justify-between gap-2">
        <div class="text-sm font-bold text-gray-100">{{ label }}</div>
        <div class="text-sm font-mono text-gray-200 font-semibold">
          {{ latest | number:'1.0-2' }}
        </div>
      </div>

      <svg viewBox="0 0 100 40" class="h-16 w-full bg-gray-950/50 rounded">
        <!-- Midline -->
        <line x1="0" y1="20" x2="100" y2="20" stroke="rgba(148,163,184,0.4)" stroke-width="1" />
        <!-- Now marker (center) -->
        <line x1="50" y1="0" x2="50" y2="40" stroke="rgba(148,163,184,0.3)" stroke-width="1" />
        <path
          [attr.d]="pathD"
          [attr.stroke]="stroke"
          stroke-width="2.5"
          fill="none"
          stroke-linecap="round"
          stroke-linejoin="round"
        />
      </svg>
    </div>
  `
})
export class FlowWaveComponent implements OnChanges {
    @Input({ required: true }) series: readonly number[] = [];
    @Input() label = '';
    @Input() stroke = 'rgb(34,197,94)';

    // Render style: keep "now" at center, draw history into the left half (wave "flows into" the now-line).
    @Input() nowMode: 'center' | 'right' = 'center';

    public pathD = 'M 0 20';
    public latest = 0;

    ngOnChanges(): void {
        const values = this.series ?? [];
        this.latest = values.length ? values[values.length - 1] : 0;
        this.pathD = this.buildPath(values, this.nowMode);
    }

    private buildPath(values: readonly number[], nowMode: 'center' | 'right'): string {
        if (!values.length) return 'M 0 20';

        const maxAbs = Math.max(1e-9, ...values.map((v) => Math.abs(v)));
        const n = values.length;

        // x range: [0..50] if now=center, else [0..100]
        const xEnd = nowMode === 'center' ? 50 : 100;
        const xStart = nowMode === 'center' ? 0 : 0;

        const dx = n <= 1 ? 0 : (xEnd - xStart) / (n - 1);
        const yMid = 20;
        const yAmp = 18; // leaves headroom

        let d = '';
        for (let i = 0; i < n; i++) {
            const x = xStart + i * dx;
            const y = yMid - (values[i] / maxAbs) * yAmp;
            d += i === 0 ? `M ${x.toFixed(2)} ${y.toFixed(2)}` : ` L ${x.toFixed(2)} ${y.toFixed(2)}`;
        }

        // If now is centered, shift the wave so the most recent point lands on x=50.
        if (nowMode === 'center') {
            const shift = 50 - xEnd; // xEnd is 50 already, so shift is 0 (kept for clarity if we tune later)
            if (shift !== 0) return `M 0 20 ${d}`; // fallback
        }

        return d;
    }
}


