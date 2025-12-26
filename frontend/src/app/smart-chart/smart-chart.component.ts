import { Component, ElementRef, OnInit, ViewChild, effect, inject, input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { createChart, IChartApi, ISeriesApi, CandlestickData, ColorType, LineStyle } from 'lightweight-charts';
import { HistoricalDataService, Candle } from '../historical-data.service';
import { DataStreamService } from '../data-stream.service';
import { LevelStreamService } from '../level-stream.service';

@Component({
    selector: 'app-smart-chart',
    standalone: true,
    imports: [CommonModule],
    template: `
    <div class="chart-container" #chartContainer></div>
    <div class="overlays">
       <!-- Option to render HTML overlays if needed, but we use Canvas lines -->
    </div>
  `,
    styles: [`
    :host {
      display: block;
      width: 100%;
      height: 100%;
      position: relative;
    }
    .chart-container {
      width: 100%;
      height: 100%;
    }
  `]
})
export class SmartChartComponent implements OnInit {
    private historicalService = inject(HistoricalDataService);
    private dataStream = inject(DataStreamService);
    private levelStream = inject(LevelStreamService);

    @ViewChild('chartContainer', { static: true }) container!: ElementRef<HTMLDivElement>;

    private chart!: IChartApi;
    private candlestickSeries!: ISeriesApi<"Candlestick">;

    // Signal updates
    levels = this.levelStream.levels;
    spy = this.dataStream.spy; // Use direct spy signal for real-time updates

    constructor() {
        // Effect to handle real-time price updates
        effect(() => {
            const quote = this.spy();
            if (quote && quote.spot && this.candlestickSeries) {
                // We need to update the current candle
                // For simplicity in this V2 prototype, we just update the 'close' of the last candle 
                // if the time is within the interval, or add a new one.
                // Doing full real-time candle formation requires accumulating OHLC.
                this.updateRealtimeCandle(quote.spot);
            }
        });

        // Effect to update level lines
        effect(() => {
            const levels = this.levels();
            this.updateLevelMarkers(levels);
        });
    }

    ngOnInit() {
        this.initChart();
        this.loadHistory();
    }

    private initChart() {
        this.chart = createChart(this.container.nativeElement, {
            layout: {
                background: { type: ColorType.Solid, color: '#0f172a' }, // Matches dark theme
                textColor: '#94a3b8',
            },
            grid: {
                vertLines: { color: 'rgba(51, 65, 85, 0.4)' },
                horzLines: { color: 'rgba(51, 65, 85, 0.4)' },
            },
            rightPriceScale: {
                visible: true,
                borderColor: 'rgba(51, 65, 85, 1)',
            },
            timeScale: {
                borderColor: 'rgba(51, 65, 85, 1)',
                timeVisible: true,
            },
        });

        this.candlestickSeries = this.chart.addCandlestickSeries({
            upColor: '#22c55e',
            downColor: '#ef4444',
            borderVisible: false,
            wickUpColor: '#22c55e',
            wickDownColor: '#ef4444',
        });

        // Fit content on resize
        new ResizeObserver(entries => {
            if (entries.length === 0 || !entries[0].target) return;
            const newRect = entries[0].contentRect;
            this.chart.applyOptions({ width: newRect.width, height: newRect.height });
        }).observe(this.container.nativeElement);
    }

    private loadHistory() {
        this.historicalService.getCandles().subscribe(candles => {
            if (candles.length > 0) {
                // Sort just in case
                candles.sort((a, b) => a.time - b.time);

                // Cast to CandlestickData
                const data: CandlestickData[] = candles.map(c => ({
                    time: c.time as any, // TV expects UTCTimestamp usually, but number is accepted
                    open: c.open,
                    high: c.high,
                    low: c.low,
                    close: c.close
                }));

                this.candlestickSeries.setData(data);
                this.chart.timeScale().fitContent();
            }
        });
    }

    private lastCandle: CandlestickData | null = null;

    private updateRealtimeCandle(price: number) {
        // Simplified logic: Assume we update the latest candle or create a new one every 2 min
        // Real implementation would track start time of bar.
        // For prototype: we will just update the close of the VERY LAST candle if it exists.

        // TODO: robust interval logic.
    }

    private updateLevelMarkers(levels: any[]) {
        // Use PriceLines for levels
        // Clear existing (Lightweight charts doesn't have "clearPriceLines", need to manage instances)
        // Implementation pending detail requirements. For now, we focus on the candles.
    }
}
