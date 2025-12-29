# Bronze Layer Schema

## Futures Trades
**Path**: `bronze/futures/trades/symbol=ES/date=YYYY-MM-DD/hour=HH/*.parquet`

| Field | Type | Description |
|-------|------|-------------|
| ts_event_ns | int64 | Event timestamp (nanoseconds) |
| ts_recv_ns | int64 | Receive timestamp (nanoseconds) |
| source | string | Data source |
| symbol | string | Futures symbol |
| price | double | Trade price |
| size | int64 | Trade size (contracts) |
| aggressor | int64 | Aggressor side |
| exchange | string | Exchange identifier |
| conditions | null | Trade conditions |
| seq | int64 | Sequence number |

## Futures MBP-10
**Path**: `bronze/futures/mbp10/symbol=ES/date=YYYY-MM-DD/hour=HH/*.parquet`

| Field | Type | Description |
|-------|------|-------------|
| ts_event_ns | int64 | Event timestamp (nanoseconds) |
| ts_recv_ns | int64 | Receive timestamp (nanoseconds) |
| source | string | Data source |
| symbol | string | Futures symbol |
| is_snapshot | bool | Snapshot indicator |
| seq | int64 | Sequence number |
| bid_px_1 | double | Bid price level 1 |
| bid_sz_1 | int64 | Bid size level 1 |
| ask_px_1 | double | Ask price level 1 |
| ask_sz_1 | int64 | Ask size level 1 |
| bid_px_2 | double | Bid price level 2 |
| bid_sz_2 | int64 | Bid size level 2 |
| ask_px_2 | double | Ask price level 2 |
| ask_sz_2 | int64 | Ask size level 2 |
| bid_px_3 | double | Bid price level 3 |
| bid_sz_3 | int64 | Bid size level 3 |
| ask_px_3 | double | Ask price level 3 |
| ask_sz_3 | int64 | Ask size level 3 |
| bid_px_4 | double | Bid price level 4 |
| bid_sz_4 | int64 | Bid size level 4 |
| ask_px_4 | double | Ask price level 4 |
| ask_sz_4 | int64 | Ask size level 4 |
| bid_px_5 | double | Bid price level 5 |
| bid_sz_5 | int64 | Bid size level 5 |
| ask_px_5 | double | Ask price level 5 |
| ask_sz_5 | int64 | Ask size level 5 |
| bid_px_6 | double | Bid price level 6 |
| bid_sz_6 | int64 | Bid size level 6 |
| ask_px_6 | double | Ask price level 6 |
| ask_sz_6 | int64 | Ask size level 6 |
| bid_px_7 | double | Bid price level 7 |
| bid_sz_7 | int64 | Bid size level 7 |
| ask_px_7 | double | Ask price level 7 |
| ask_sz_7 | int64 | Ask size level 7 |
| bid_px_8 | double | Bid price level 8 |
| bid_sz_8 | int64 | Bid size level 8 |
| ask_px_8 | double | Ask price level 8 |
| ask_sz_8 | int64 | Ask size level 8 |
| bid_px_9 | double | Bid price level 9 |
| bid_sz_9 | int64 | Bid size level 9 |
| ask_px_9 | double | Ask price level 9 |
| ask_sz_9 | int64 | Ask size level 9 |
| bid_px_10 | double | Bid price level 10 |
| bid_sz_10 | int64 | Bid size level 10 |
| ask_px_10 | double | Ask price level 10 |
| ask_sz_10 | int64 | Ask size level 10 |

## Options Trades
**Path**: `bronze/options/trades/underlying=ES/date=YYYY-MM-DD/hour=HH/*.parquet`

| Field | Type | Description |
|-------|------|-------------|
| ts_event_ns | int64 | Event timestamp (nanoseconds) |
| ts_recv_ns | int64 | Receive timestamp (nanoseconds) |
| source | string | Data source |
| underlying | string | Underlying symbol |
| option_symbol | string | Option symbol |
| exp_date | string | Expiration date |
| strike | double | Strike price |
| right | string | Call/Put |
| price | double | Trade price |
| size | int64 | Trade size (contracts) |
| opt_bid | null | Option bid (unused) |
| opt_ask | null | Option ask (unused) |
| seq | int64 | Sequence number |
| aggressor | int64 | Aggressor side |

## Options NBBO
**Path**: `bronze/options/nbbo/underlying=ES/date=YYYY-MM-DD/hour=HH/*.parquet`

| Field | Type | Description |
|-------|------|-------------|
| ts_event_ns | int64 | Event timestamp (nanoseconds) |
| ts_recv_ns | int64 | Receive timestamp (nanoseconds) |
| source | string | Data source |
| underlying | string | Underlying symbol |
| option_symbol | string | Option symbol |
| exp_date | string | Expiration date |
| strike | double | Strike price |
| right | string | Call/Put |
| bid_px | double | National best bid |
| ask_px | double | National best ask |
| bid_sz | int64 | Bid size (contracts) |
| ask_sz | int64 | Ask size (contracts) |
| seq | int64 | Sequence number |

