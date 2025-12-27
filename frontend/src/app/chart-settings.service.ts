import { Injectable, signal } from '@angular/core';

export type ChartTimeframe = '2m' | '5m' | '15m';

@Injectable({ providedIn: 'root' })
export class ChartSettingsService {
  // Timeframe
  public timeframe = signal<ChartTimeframe>('2m');

  // Overlays
  public showGEX = signal(true);
  public showVWAP = signal(true);
  public showSMAs = signal(true);

  // Level filters
  public showWalls = signal(true);
  public showStructural = signal(true);

  // Quality filters
  public minConfluence = signal(0);
  public minTradeability = signal(0);
}


