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
  pressure_grad: number;
  u_wave_energy: number;
  nu: number;
  Omega: number;
}

export interface ForecastRow {
  window_end_ts_ns: bigint;
  horizon_s: number;
  predicted_spot_tick: bigint;
  predicted_tick_delta: bigint;
  confidence: number;
  // Diagnostic fields (h=0 row)
  run_score_up?: number;
  run_score_down?: number;
  d_up?: number;
  d_down?: number;
}

export interface ProductMeta {
  tick_size: number;
  tick_int: number;
  strike_ticks: number;
  grid_max_ticks: number;
}

export interface StreamCallbacks {
  onTick: (ts: bigint, surfaces: string[]) => void;
  onSnap: (row: SnapRow) => void;
  onVelocity: (rows: VelocityRow[]) => void;
  onOptions: (rows: OptionsRow[]) => void;
  onForecast: (rows: ForecastRow[]) => void;
  onProductMeta?: (meta: ProductMeta) => void;
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
            if (callbacks.onProductMeta && msg.tick_size !== undefined) {
              callbacks.onProductMeta({
                tick_size: msg.tick_size,
                tick_int: msg.tick_int,
                strike_ticks: msg.strike_ticks,
                grid_max_ticks: msg.grid_max_ticks,
              });
            }
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
                  pressure_grad: (json.pressure_grad ?? 0) as number,
                  u_wave_energy: (json.u_wave_energy ?? 0) as number,
                  nu: (json.nu ?? 0) as number,
                  Omega: (json.Omega ?? 0) as number,
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
                  horizon_s: (json.horizon_s ?? 0) as number,
                  predicted_spot_tick: BigInt((json.predicted_spot_tick ?? '0') as string),
                  predicted_tick_delta: BigInt((json.predicted_tick_delta ?? '0') as string),
                  confidence: (json.confidence ?? 0) as number,
                  run_score_up: (json.run_score_up ?? json.RunScore_up ?? 0) as number,
                  run_score_down: (json.run_score_down ?? json.RunScore_down ?? 0) as number,
                  d_up: (json.d_up ?? json.D_up ?? 0) as number,
                  d_down: (json.d_down ?? json.D_down ?? 0) as number,
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
