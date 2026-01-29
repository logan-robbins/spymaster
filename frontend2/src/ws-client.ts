import { tableFromIPC } from 'apache-arrow';

export interface SnapRow {
  window_end_ts_ns: bigint;
  mid_price: number;
  spot_ref_price_int: bigint;
  book_valid: boolean;
}

export interface VelocityRow {
  window_end_ts_ns: bigint;
  spot_ref_price_int: bigint;
  rel_ticks: number;
  side: string;
  liquidity_velocity: number;
  rho: number;
  nu: number;
  kappa: number;
  pressure_grad: number;
  u_wave_energy: number;
  Omega: number;
}

export interface OptionsRow {
  window_end_ts_ns: bigint;
  spot_ref_price_int: bigint;
  rel_ticks: number;  // Always multiples of 20 ($5 increments)
  liquidity_velocity: number;  // Aggregated across C/P and A/B
}

export interface ForecastRow {
  window_end_ts_ns: bigint;
  horizon_s: number;
  predicted_spot_tick: bigint; // Relative in contract? No, name implies absolute. Or is it delta?
  // Schema says predicted_spot_tick (null, long).
  // Also predicted_tick_delta.
  predicted_tick_delta: bigint;
  confidence: number;
  RunScore_up: number;
  RunScore_down: number;
  D_up: number | null;
  D_down: number | null;
}

export interface StreamCallbacks {
  onTick: (ts: bigint, surfaces: string[]) => void;
  onSnap: (row: SnapRow) => void;
  onVelocity: (rows: VelocityRow[]) => void;
  onOptions: (rows: OptionsRow[]) => void;
  onForecast: (rows: ForecastRow[]) => void;
}

export function connectStream(
  symbol: string,
  dt: string,
  callbacks: StreamCallbacks
): WebSocket {
  const wsUrl = `ws://localhost:8001/v1/velocity/stream?symbol=${symbol}&dt=${dt}`;
  console.log(`Connecting to: ${wsUrl}`);

  const ws = new WebSocket(wsUrl);
  let pendingSurface: string | null = null;

  const messageQueue: MessageEvent[] = [];
  let isProcessing = false;

  const processQueue = async () => {
    if (isProcessing) return;
    isProcessing = true;

    while (messageQueue.length > 0) {
      const event = messageQueue.shift();
      if (!event) continue;

      try {
        if (typeof event.data === 'string') {
          const msg = JSON.parse(event.data);

          if (msg.type === 'batch_start') {
            const ts = BigInt(msg.window_end_ts_ns);
            const surfaceList = msg.surfaces || [];
            callbacks.onTick(ts, surfaceList);
          } else if (msg.type === 'surface_header') {
            pendingSurface = msg.surface;
          }
        } else if (event.data instanceof Blob) {
          const surface = pendingSurface;
          if (!surface) continue;

          const buffer = await event.data.arrayBuffer();
          const table = tableFromIPC(buffer);

          if (surface === 'snap' && table.numRows > 0) {
            const row = table.get(0);
            if (row) {
              const json = row.toJSON() as Record<string, unknown>;
              callbacks.onSnap({
                window_end_ts_ns: BigInt(json.window_end_ts_ns as string),
                mid_price: json.mid_price as number,
                spot_ref_price_int: BigInt(json.spot_ref_price_int as string),
                book_valid: json.book_valid as boolean,
              });
            }
          } else if (surface === 'velocity') {
            const rows: VelocityRow[] = [];
            for (let i = 0; i < table.numRows; i++) {
              const row = table.get(i);
              if (row) {
                const json = row.toJSON() as Record<string, unknown>;
                rows.push({
                  window_end_ts_ns: BigInt(json.window_end_ts_ns as string),
                  spot_ref_price_int: BigInt(json.spot_ref_price_int as string),
                  rel_ticks: Number(json.rel_ticks),
                  side: json.side as string,
                  liquidity_velocity: json.liquidity_velocity as number,
                  rho: (json.rho ?? 0) as number,
                  nu: (json.nu ?? 0) as number,
                  kappa: (json.kappa ?? 0) as number,
                  pressure_grad: (json.pressure_grad ?? 0) as number,
                  u_wave_energy: (json.u_wave_energy ?? 0) as number,
                  Omega: (json.Omega ?? 0) as number,
                });
              }
            }
            callbacks.onVelocity(rows);
          } else if (surface === 'options') {
            const rows: OptionsRow[] = [];
            for (let i = 0; i < table.numRows; i++) {
              const row = table.get(i);
              if (row) {
                const json = row.toJSON() as Record<string, unknown>;
                rows.push({
                  window_end_ts_ns: BigInt(json.window_end_ts_ns as string),
                  spot_ref_price_int: BigInt(json.spot_ref_price_int as string),
                  rel_ticks: Number(json.rel_ticks),
                  liquidity_velocity: json.liquidity_velocity as number,
                });
              }
            }
            callbacks.onOptions(rows);
          } else if (surface === 'forecast') {
            const rows: ForecastRow[] = [];
            for (let i = 0; i < table.numRows; i++) {
              const row = table.get(i);
              if (row) {
                const json = row.toJSON() as Record<string, unknown>;
                rows.push({
                  window_end_ts_ns: BigInt(json.window_end_ts_ns as string),
                  horizon_s: json.horizon_s as number,
                  predicted_spot_tick: BigInt(json.predicted_spot_tick as string),
                  predicted_tick_delta: BigInt(json.predicted_tick_delta as string),
                  confidence: json.confidence as number,
                  RunScore_up: json.RunScore_up as number,
                  RunScore_down: json.RunScore_down as number,
                  D_up: json.D_up as number | null,
                  D_down: json.D_down as number | null,
                });
              }
            }
            callbacks.onForecast(rows);
          }

          pendingSurface = null;
        }
      } catch (e) {
        console.error('Error processing message:', e);
      }
    }

    isProcessing = false;
  };

  ws.onopen = () => console.log('WebSocket connected');
  ws.onmessage = (event) => {
    messageQueue.push(event);
    processQueue();
  };
  ws.onerror = (err) => console.error('WebSocket error:', err);
  ws.onclose = () => console.log('WebSocket closed');

  return ws;
}
