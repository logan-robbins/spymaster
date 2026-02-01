# Raw Data Layer Inventory

**Generated:** 2026-01-30  
**Updated:** 2026-01-31 (Complete Jan 5-29 coverage for all products)  
**Total Size:** 285 GB (uncompressed .dbn files)  
**Location:** `/backend/lake/raw/source=databento/`

**Context:** Raw layer is the source-of-truth input layer. Downstream layers (Bronze, Silver, Gold) are derived from this data.

**Coverage:** All 18 trading days from 2026-01-05 to 2026-01-29 (excl. weekends & MLK Day Jan 19)

---

## Summary by Product Type

| Product Type | Symbols | Size | Dates | Tables | Format |
|--------------|---------|------|-------|--------|--------|
| future_mbo | ES | 16 GB | 22 (incl Sunday sessions) | market_by_order_dbn | .dbn |
| future_option_mbo | ES | 185 GB ✅ | 18 (all trading days) | market_by_order_dbn | .dbn |
| equity_mbo | QQQ | 40 GB | 19 | market_by_order_dbn | .dbn |
| equity_option_cmbp_1 | QQQ | 43 GB | 14 (0DTE days only) | cmbp_1 | .dbn |
| equity_option_statistics | QQQ | 160 MB | 14 | statistics | .dbn |
| dataset=definition | GLBX/OPRA | ~1 GB | Various | definition | .dbn |

**Total:** ~285 GB

✅ **ES Options:** All 18 trading days with **ONLY 0DTE contracts**
✅ **QQQ Options:** Mon/Wed/Fri 0DTE dates (QQQ doesn't have daily 0DTE like ES)

---

## Detailed Inventory

### 1. ES Futures MBO (16 GB)

**Product Type:** `future_mbo`  
**Schema:** Market By Order (MBO)  
**Dataset:** GLBX.MDP3

#### Symbol: ES (E-mini S&P 500)
- **Date Range:** 2026-01-05 to 2026-01-29 (22 files, includes Sunday sessions)
- **Trading Days Covered:** All 18 required (Jan 5-9, 12-16, 20-23, 26-29)
- **Size:** 16 GB
- **Format:** .dbn uncompressed

---

### 2. ES Futures Options MBO (185 GB) ✅ 0DTE ONLY

**Product Type:** `future_option_mbo`  
**Schema:** Market By Order (MBO)  
**Dataset:** GLBX.MDP3

#### Symbol: ES Options
- **Date Range:** 2026-01-05 to 2026-01-29 (18 files)
- **Trading Days Covered:** All 18 required
- **Size:** 185 GB
- **Format:** .dbn uncompressed
- **Filter:** 0DTE contracts only (downloaded via raw_symbol from definition files)

**0DTE Parent Symbols by Day Type:**
- Mon-Thu: E1-E5.OPT + E1A-E5E.OPT variants (daily expirations)
- Fridays (non-quarterly): EW.OPT, EW1-EW4.OPT (weekly expirations)
- 3rd Friday (quarterly months only): ES.OPT (standard monthly)

---

### 3. QQQ Equity MBO (40 GB)

**Product Type:** `equity_mbo`  
**Schema:** Market By Order (MBO)  
**Dataset:** XNAS.ITCH

#### Symbol: QQQ (Invesco QQQ Trust)
- **Date Range:** 2026-01-02 to 2026-01-29 (19 files)
- **Trading Days Covered:** All 18 required (Jan 5-29)
- **Size:** 40 GB
- **Format:** .dbn uncompressed

---

### 4. QQQ Equity Options CMBP-1 (43 GB)

**Product Type:** `equity_option_cmbp_1`  
**Schema:** Consolidated Market By Price (Level 1)  
**Dataset:** OPRA.PILLAR

#### Symbol: QQQ Options
- **Dates:** 14 days (0DTE dates only: Mon/Wed/Fri)
- **Size:** 43 GB
- **Format:** .dbn uncompressed
- **Note:** QQQ doesn't have daily 0DTE like ES - only Mon/Wed/Fri expirations

---

### 5. QQQ Equity Options Statistics (160 MB)

**Product Type:** `equity_option_statistics`  
**Schema:** Statistics  
**Dataset:** OPRA.PILLAR

#### Symbol: QQQ Options
- **Dates:** 14 days (matching cmbp-1)
- **Size:** 160 MB
- **Format:** .dbn uncompressed

---

### 6. Definition Datasets (~1 GB)

**Locations:**
- `dataset=definition/` - GLBX.MDP3 (ES futures options)
- `dataset=definition/venue=opra/` - OPRA.PILLAR (QQQ equity options)

**Purpose:** Contract definitions for 0DTE filtering:
- Instrument metadata and raw_symbol mappings
- Expiration dates for filtering 0DTE contracts
- Strike prices and underlying mappings

---

## Data Structure

Each raw data partition follows this structure:

```
lake/raw/source=databento/
├── dataset=definition/
│   ├── glbx-mdp3-YYYYMMDD.definition.dbn
│   └── GLBX-YYYYMMDD-{JOB_ID}/
│       ├── metadata.json
│       ├── manifest.json
│       └── glbx-mdp3-YYYYMMDD.definition.dbn
├── product_type={PRODUCT_TYPE}/
│   └── symbol={SYMBOL}/
│       ├── metadata.json (symbol-level)
│       ├── manifest.json (batch download manifest)
│       ├── symbology.json (symbol mappings)
│       └── table={TABLE}/
│           ├── condition.json (query conditions)
│           ├── metadata.json (table-level)
│           ├── manifest.json (table-level manifest)
│           └── {DATASET}-YYYYMMDD-{JOB_ID}/
│               ├── condition.json
│               ├── metadata.json
│               ├── manifest.json
│               └── {dataset}-{YYYYMMDD}.{schema}.dbn.zst
```

---

## Metadata Files

Each data partition includes:

1. **metadata.json**: Query parameters including:
   - Dataset and schema
   - Symbol list
   - Date range (start/end timestamps)
   - Encoding and compression settings

2. **manifest.json**: File listing with:
   - Filenames
   - File sizes
   - SHA256 hashes
   - Download URLs (HTTPS and FTP)

3. **condition.json**: Databento condition mappings

4. **symbology.json**: Symbol resolution mappings (for futures)

---

## Data Characteristics

### ES Futures
- **Coverage:** Continuous front-month contract data
- **Resolution:** Tick-by-tick order book (MBO)
- **Hours:** 23/5 trading (Sunday evening through Friday)
- **Depth:** Full order book with order IDs
- **History:** 22 days (Jan 5-29, 2026 incl. Sunday sessions)

### ES Futures Options (0DTE ONLY)
- **Coverage:** ONLY 0DTE contracts (contracts expiring same day)
- **Resolution:** Tick-by-tick order book (MBO)
- **MBO:** Order-level detail for 0DTE contracts
- **History:** 18 trading days (Jan 5-29, 2026)
- **Size:** ~10 GB/day average (185 GB total)

### QQQ Equity
- **Coverage:** Main QQQ ETF
- **Resolution:** Tick-by-tick order book (MBO)
- **Hours:** Regular trading hours + extended hours
- **Depth:** Full order book with order IDs
- **History:** 19 days (Jan 2-29, 2026)

### QQQ Equity Options (0DTE ONLY)
- **Coverage:** ONLY 0DTE contracts
- **CMBP-1:** Level 1 consolidated best bid/offer
- **Strike Range:** Centered around ATM
- **History:** 14 days (Mon/Wed/Fri only - QQQ doesn't have daily 0DTE like ES)

---

## Data Sources

All data sourced from **Databento**:

- **Futures/Futures Options:** GLBX.MDP3 (CME Globex MDP 3.0)
- **Equities:** XNAS.ITCH (Nasdaq TotalView-ITCH)
- **Equity Options:** OPRA.PILLAR (Options Price Reporting Authority)

---

## Notes

1. **Complete Coverage:** All 18 trading days from Jan 5-29, 2026 covered for ES and QQQ
2. **0DTE Only:** Options data filtered to include ONLY same-day expiring contracts
3. **QQQ 0DTE Schedule:** QQQ has 0DTE Mon/Wed/Fri only (not daily like ES)
4. **Script Fix:** `batch_download_futures.py` fixed to handle non-quarterly 3rd Friday dates correctly

---

## Storage Format

- **Format:** Databento Binary Encoding (DBN) - uncompressed
- **Average File Sizes (Uncompressed):**
  - ES futures: ~700 MB/day
  - ES options MBO (0DTE): ~10 GB/day
  - QQQ equity: ~2 GB/day
  - QQQ options CMBP-1 (0DTE): ~3 GB/day

---

## Next Steps

To process this raw data:

1. **Contract Selection:** Run `mbo_contract_day_selector` to map session dates to front-month contracts
2. **Bronze Layer:** Normalize raw DBN files to parquet with standardized schemas
3. **Silver Layer:** Reconstruct order books and generate snapshots
4. **Gold Layer:** Compute features for ML/analysis

See README.md for detailed pipeline commands.
