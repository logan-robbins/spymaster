import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, catchError, map, of } from 'rxjs';

export interface Candle {
    time: number; // unix timestamp in seconds
    open: number;
    high: number;
    low: number;
    close: number; // close is reserved keyword in some contexts but fine here
    volume: number;
}

@Injectable({
    providedIn: 'root'
})
export class HistoricalDataService {
    private http = inject(HttpClient);

    // Hardcoded for now. In real app, use environment.ts
    private apiUrl = 'http://localhost:8000/api/history/candles';

    getCandles(symbol: string = 'SPY', interval: number = 2, days: number = 1): Observable<Candle[]> {
        return this.http.get<Candle[]>(this.apiUrl, {
            params: {
                symbol,
                interval: interval.toString(),
                days: days.toString()
            }
        }).pipe(
            catchError(err => {
                console.error('Failed to fetch candles', err);
                return of([]);
            })
        );
    }
}
