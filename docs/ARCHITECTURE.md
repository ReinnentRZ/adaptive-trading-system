# Technical Architecture & Maintenance SOP

> **Adaptive Trading System: Corong 3-Pilar & Layer-2 Meta-Labeler**  
> *Single Source of Truth (SSOT), Multi-Pair Scanner, and Model Governance Guide*

---

## 📑 Daftar Isi
1. [Arsitektur Aliran Sinyal & Eksekusi](#1-arsitektur-aliran-sinyal--eksekusi)
2. [Tata Kelola State Portofolio (`live_bot_state.json`)](#2-tata-kelola-state-portofolio-live_bot_statejson)
3. [Arsitektur Model Kuantitatif & Machine Learning](#3-arsitektur-model-kuantitatif--machine-learning)
4. [SOP Pemeliharaan & Retraining Berkala (Siklus 6 Bulan)](#4-sop-pemeliharaan--retraining-berkala-siklus-6-bulan)
5. [Mitigasi Risiko & Protokol Penanganan Kegagalan (Safeguards)](#5-mitigasi-risiko--protokol-penanganan-kegagalan-safeguards)

---

## 1. Arsitektur Aliran Sinyal & Eksekusi

Sistem beroperasi menggunakan model *event-driven loop* berbasis penutupan lilin 1 Jam (*1H candle close*). Seluruh proses pengambilan keputusan mengikuti prinsip **Single Source of Truth (SSOT)**:

```
[ CCXT Binance REST API ]
           │
           ▼
[ Fetch 1H OHLCV per Symbol ]  (BTC/USDT, ETH/USDT, SOL/USDT sekuensial)
           │
           ├─► Rate-limit Guard: sleep(0.5s) antar koin
           │
           ▼
[ Feature Engineering Pipeline ]
   ├── Log Return, Normalized ATR, Signed Volume
   ├── Causal 1D Kalman Filter (Trend Denoising)
   ├── GARCH(1,1) Volatility Forecasting (Dynamic Brackets)
   └── Cross-Asset BTC Macro Features (btc_return_1h, relative_strength_vs_btc)
           │
           ▼
[ Pilar 1: Macro Regime Funnel ]
   ├── Gaussian HMM Forward-Filtering (Zero-Lookahead)
   ├── Syarat: HMM State == 0 (Bullish Momentum)
   ├── Macro Trend: Close > EMA(200)
   └── State Freshness: state_age <= 4 jam & Single-Shot Lock
           │ (Lolos Pilar 1)
           ▼
[ Pilar 2: Micro Pullback Trigger ]
   ├── Syarat: Low <= EMA(9) ATAU RSI(14) <= 52.0
           │ (Kandidat Sinyal Terbentuk)
           ▼
[ Layer-2 Gate: LightGBM Meta-Labeler ]
   ├── Evaluasi Vektor Fitur Skala-Invarian (9 Fitur)
   ├── Model: CalibratedClassifierCV (Sigmoid Probabilities)
   ├── Gerbang: P(Win) >= 0.50 (Default Threshold)
           │ (Disetujui AI & Slot Tersedia)
           ▼
[ Slot Quota & Ledger Inspection ]
   ├── Cek active_positions di data/live_bot_state.json
   └── Validasi: Total posisi aktif < max_concurrent_positions (3)
           │ (Kuota Lolos)
           ▼
[ Pilar 3: Asymmetric Trade Execution ]
   ├── Pasang Maker Limit Order di EMA(9) (Fee 0.02%, Zero-Slippage)
   ├── Hitung Dynamic Risk Brackets (ATR-based):
   │     • Initial Stop Loss = Entry - (0.70 * ATR)
   │     • Take Profit 1     = Entry + (0.80 * ATR) -> Scale out 50%, Geser SL ke BE
   │     • Take Profit 2     = Entry + (1.20 * ATR) -> Tutup sisa 50%
   │     • Emergency Exit    = HMM State 3 (Bear Dump)
           │
           ▼
[ State Ledger Mutation & Persistence ]
   └── Tulis pembaruan tiket ke data/live_bot_state.json secara atomik
```

---

## 2. Tata Kelola State Portofolio (`live_bot_state.json`)

Status seluruh posisi, kuota portofolio, dan riwayat transaksi disimpan dalam file persistensi lokal host `data/live_bot_state.json`.

### Skema Ledger Multi-Pair
```json
{
  "active_positions": {
    "BTC/USDT": null,
    "ETH/USDT": {
      "symbol": "ETH/USDT",
      "status": "OPEN",
      "side": "BUY",
      "entry_price": 2650.50,
      "current_price": 2685.20,
      "initial_size": 0.0377,
      "remaining_size": 0.01885,
      "initial_sl": 2615.00,
      "current_sl": 2657.12,
      "tp1_price": 2682.00,
      "tp2_price": 2705.00,
      "tp1_hit": true,
      "be_locked": true,
      "entry_time": 1727931600000,
      "holding_bars": 3,
      "trade_id": "POS-ETH-1727931600"
    },
    "SOL/USDT": null
  },
  "max_concurrent_positions": 3,
  "allocation_per_trade_usd": 100.0,
  "trade_history": [
    {
      "trade_id": "POS-BTC-1727900000",
      "symbol": "BTC/USDT",
      "side": "BUY",
      "entry_price": 64200.00,
      "exit_price": 64950.00,
      "realized_pnl_usd": 1.42,
      "pnl_pct": 1.42,
      "exit_reason": "TP2_FULL_EXIT",
      "exit_time": 1727920000000
    }
  ]
}
```

### Auto-Bootstrapping Logic
Jika file `data/live_bot_state.json` belum ada atau kosong (misalnya saat instalasi baru di kontainer Docker):
1. Bot secara otomatis membuat folder `data/` jika belum tersedia.
2. Menginisialisasi ledger dengan slot `active_positions` kosong untuk setiap simbol dalam `SYMBOLS`.
3. Mencatat log konfirmasi: `[INFO] Initialized new persistent state ledger at data/live_bot_state.json`.

---

## 3. Arsitektur Model Kuantitatif & Machine Learning

### A. Gaussian Hidden Markov Model (HMM)
- **Implementasi**: [`hmmlearn.hmm.GaussianHMM`](file:///home/rei/reinn/projects/adaptive-trading-system/src/strategies/regime_funnel.py)
- **Jumlah Status**: 4 States (`0: Bullish Momentum`, `1: Ranging Consolidation`, `2: High Volatility Exhaustion`, `3: Bearish Dump`).
- **Inferensi**: *Online Forward Filtering* $P(S_t = j \mid X_0 \dots X_t)$ menghitung probabilitas status terkini tanpa melihat ke depan (*Zero Lookahead Bias*).
- **Fitur Masukan (Scale-Invariant)**:
  1. $\text{Log Return} = \ln(\text{Close}_t / \text{Close}_{t-1})$
  2. $\text{Normalized ATR} = \text{ATR}_{14} / \text{Close}_t$
  3. $\text{Signed Volume Intensity} = \text{Volume}_t \times \text{sign}(\text{Close}_t - \text{Open}_t) / \text{SMA}_{20}(\text{Volume})$

### B. Layer-2 LightGBM Meta-Labeler
- **Implementasi**: [`scripts/train_funnel_meta_labeler.py`](file:///home/rei/reinn/projects/adaptive-trading-system/scripts/train_funnel_meta_labeler.py) & [`src/strategies/meta_labeler.py`](file:///home/rei/reinn/projects/adaptive-trading-system/src/strategies/meta_labeler.py)
- **Pelatihan**: *Pooled multi-asset dataset* (BTC, ETH, SOL digabungkan dalam satu event matrix).
- **Validasi Cross-Validation**: `PurgedGroupTimeSeriesSplit` dengan 5 lilin *embargo period* untuk mengeliminasi *label overlap*.
- **Kalibrasi Probabilitas**: `CalibratedClassifierCV(method='sigmoid', cv='prefit')` memastikan output probabilitas mencerminkan rasio kemenangan empiris.
- **Vektor Fitur (9 Fitur)**:
  - `rsi`, `cci`, `adx`, `wt_diff`, `volatility_ratio`, `normalized_atr`, `lorentzian_signal`
  - Makro Bitcoin: `btc_return_1h` (return lilin 1H BTC)
  - Makro Relatif: `relative_strength_vs_btc` ($R_{\text{asset}} - R_{\text{btc}}$)

### C. Filter Kausal Kalman & GARCH
- **1D Kalman Filter**: State-space estimator satu dimensi yang membersihkan noise harga secara kausal murni tanpa lag.
- **GARCH(1,1)**: Model ekonometrika untuk meramalkan estimasi volatilitas bersyarat $\sigma_{t+1}$ pada lilin berikutnya.

---

## 4. SOP Pemeliharaan & Retraining Berkala (Siklus 6 Bulan)

Pasar kripto mengalami pergeseran distribusi (*regime drift*). Disarankan untuk melakukan evaluasi dan pelatihan ulang model setiap **3 hingga 6 bulan** dengan prosedur standar berikut:

### Langkah 1: Pengunduhan & Resampling Dataset Baru
Unduh data OHLCV historis terbaru untuk BTC, ETH, dan SOL dari Binance:
```bash
# Pastikan dataset kontinu 1H tersimpan di folder dataset/
python3 scripts/process_datasets.py --symbols BTC/USDT,ETH/USDT,SOL/USDT --timeframe 1h
```

### Langkah 2: Retraining Gaussian HMM per Koin
Latih model HMM untuk setiap aset secara mandiri:
```bash
# Retrain HMM BTC
python3 scripts/train_regime_model.py --symbol BTC/USDT --timeframe 1h

# Retrain HMM ETH
python3 scripts/train_regime_model.py --symbol ETH/USDT --timeframe 1h

# Retrain HMM SOL
python3 scripts/train_regime_model.py --symbol SOL/USDT --timeframe 1h
```
*Hasil artifact akan tersimpan di:*
- `models/{symbol}_1h_regime_hmm.joblib`
- `models/{symbol}_1h_regime_hmm_metadata.json`

### Langkah 3: Retraining Pooled LightGBM Meta-Labeler
Jalankan pipeline pelatihan terpadu multi-aset:
```bash
python3 scripts/train_funnel_meta_labeler.py
```
*Hasil artifact akan tersimpan di:*
- `models/multi_asset_1h_funnel_metalabeler.joblib`
- `models/multi_asset_1h_funnel_metalabeler_metadata.json`

### Langkah 4: Kriteria Penerimaan Model (*Deployment Gate Criteria*)
Sebelum model baru dipromosikan ke lingkungan produksi, periksa file evaluasi JSON dan pastikan memenuhi ambang batas minimum berikut:

| Metrik Evaluasi | Batas Minimum Penerimaan | Catatan |
| :--- | :---: | :--- |
| **ROC-AUC Score** | **$\ge 0.72$** | Menjamin daya diskriminasi model di atas random walk (0.50) |
| **Brier Score Loss** | **$\le 0.23$** | Mengukur akurasi kalibrasi probabilitas |
| **Precision @ Threshold 0.50** | **$\ge 58.0\%$** | Memastikan win rate kandidat yang lolos gerbang |
| **Approval Rate @ 0.50** | **$30\% - 55\%$** | Memastikan model tidak terlalu permisif atau over-conservative |

Jika seluruh metrik lolos kriteria, restart kontainer untuk memuat model baru:
```bash
docker compose restart adaptive-trading-bot
```

---

## 5. Mitigasi Risiko & Protokol Penanganan Kegagalan (Safeguards)

### A. Binance API Weight & Rate Limiting
- **Jeda Sekuensial**: Setiap iterasi pemindaian multi-pair menyisipkan jeda `time.sleep(0.5)` antar simbol.
- **REST Limit**: Membatasi panggilan candle maksimal 200–500 bar per request, jauh di bawah batas *1200 request weight/menit* Binance.

### B. Isolasi Kesalahan per Pasangan Koin (*Fault Isolation*)
Pemindaian lilin 1H dibungkus dalam blok `try...except` terisolasi per simbol:
```python
for symbol in self.symbols:
    try:
        self.evaluate_symbol(symbol)
    except Exception as e:
        logger.error(f"Error evaluating {symbol}: {e}", exc_info=True)
        # Scan tetap berlanjut ke simbol berikutnya tanpa mematikan bot
        continue
    time.sleep(0.5)
```
*Jika koneksi atau order untuk `SOL/USDT` mengalami kendala, evaluasi untuk `BTC/USDT` dan `ETH/USDT` tetap berjalan normal.*

### C. Penanganan Masalah Izin OS & Logging
- **Resilient FileHandler**: Jika folder log host tidak memiliki hak tulis atau terbentur izin permission non-root, logger secara otomatis beralih ke `StreamHandler (stdout)` tanpa menimbulkan `PermissionError`.
- **Pre-created Directory Permissions**: Script instalasi CI dan Dockerfile memastikan direktori `data/` dan `logs/` dibuat dengan izin akses penuh (`chmod -R 777`).

### D. Manajemen Risiko Likuiditas & Fee (Maker Tier)
- **Maker Order Only**: Seluruh posisi masuk dipasang pada limit order pasif di level EMA(9), menghindari biaya *taker* tinggi dan slippage harga mendadak.
- **Hard-Floored Break-Even**: Saat TP1 tercapai, Stop Loss dinaikkan ke level $\text{Entry} \times 1.0025$, menjamin biaya komisi bursa (0.04% roundtrip) telah terkompensasi penuh.
