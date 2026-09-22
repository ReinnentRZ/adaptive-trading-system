# Adaptive Trading System: Corong 3-Pilar (BTC/USDT 1H)

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![CCXT](https://img.shields.io/badge/CCXT-Binance%20Spot-orange.svg)](https://github.com/ccxt/ccxt)
[![Tests](https://img.shields.io/badge/Tests-34%20Passed%20(100%25)-brightgreen.svg)](https://docs.pytest.org/)
[![Architecture](https://img.shields.io/badge/Architecture-SSOT%20Clean%20Design-purple.svg)](#arsitektur-strategi-corong-3-pilar)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Sistem perdagangan aset kripto algoritmik berbasis **Single Source of Truth (SSOT)** yang dirancang untuk memperdagangkan pasangan **BTC/USDT pada timeframe 1 Jam (1H)** di bursa **Binance Spot**.

Sistem ini mengeliminasi indikator konvensional yang lambat (*lagging*) dan menggantikannya dengan inferensi rezim pasar tanpa pengawasan (*unsupervised*) menggunakan **Gaussian Hidden Markov Model (HMM)**, filter tren struktural makro **EMA 200**, pemicu mikro-pullback, serta manajemen risiko asimetris **50:50 Scaling Out** dengan penguncian *Hard-Floored Break-Even*.

---

## 📑 Daftar Isi
- [Arsitektur Strategi: Corong 3-Pilar](#arsitektur-strategi-corong-3-pilar)
- [Diagram Alur Sistem](#diagram-alur-sistem)
- [Hasil Backtest Out-of-Sample (OOS)](#hasil-backtest-out-of-sample-oos)
- [Struktur Repositori](#struktur-repositori)
- [Panduan Instalasi & Deployment](#panduan-instalasi--deployment)
  - [1. Konfigurasi Lingkungan (.env)](#1-konfigurasi-lingkungan-env)
  - [2. Menjalankan via Docker Compose](#2-menjalankan-via-docker-compose)
  - [3. Pemantauan Real-Time (Terminal Dashboard)](#3-pemantauan-real-time-terminal-dashboard)
  - [4. Simulasi Mandiri (Historical Paper Replayer)](#4-simulasi-mandiri-historical-paper-replayer)
- [Verifikasi & Pengujian Unit (QA)](#verifikasi--pengujian-unit-qa)
- [Risk Disclaimer](#risk-disclaimer)

---

## 🏛️ Arsitektur Strategi: Corong 3-Pilar

Sistem ini bekerja melalui konsep **Corong Bertingkat (Cascading Funnel)** untuk menyaring sinyal palsu secara ketat sebelum modal dialokasikan:

```
                            [ Binance 1H OHLCV Ingestion ]
                                          │
                                          ▼
                ┌──────────────────────────────────────────────────┐
                │   PILAR 1: Macro Regime Gate & Trend Filter      │
                │   • Causal Gaussian HMM Online State == 0        │
                │   • Close > EMA(200) Macro Structural Filter     │
                │   • State Freshness: state_age <= 4 jam          │
                │   • Single-Shot Lock per Episode Bullish         │
                └─────────────────────────┬────────────────────────┘
                                          │ (Lolos Gerbang 1)
                                          ▼
                ┌──────────────────────────────────────────────────┐
                │   PILAR 2: Micro Timing Pullback Trigger         │
                │   • Dynamic Queue Entry: Low <= EMA(9)           │
                │     ATAU Momentum Reset: RSI(14) <= 52.0         │
                └─────────────────────────┬────────────────────────┘
                                          │ (GATE OPEN)
                                          ▼
                ┌──────────────────────────────────────────────────┐
                │   PILAR 3: Execution & Dynamic Risk Engine       │
                │   • Maker Limit Order at EMA(9) (0.02% fee)      │
                │   • Initial Stop Loss : Entry - (0.70 * ATR)     │
                │   • Target 1 (TP1)    : Entry + (0.80 * ATR)     │
                │     └─ Tutup 50% Posisi, Kunci SL ke BE (1.0025) │
                │   • Target 2 (TP2)    : Entry + (1.20 * ATR)     │
                │     └─ Tutup sisa 50% Posisi (Trade Selesai)     │
                │   • Emergency Exit    : HMM State 3 (Dump)       │
                └──────────────────────────────────────────────────┘
```

1. **Pilar 1: Macro Regime Gate (HMM + EMA 200)**
   - **Gaussian HMM Causal Inference**: Menghitung probabilitas status pasar saat ini secara *online forward-filtering* ($P(S_t = j \mid X_0 \dots X_t)$) dengan **Zero Lookahead Bias**. Input fitur merupakan data mikrostruktur murni: *Log Return*, *Normalized ATR Volatility*, dan *Signed Volume Intensity*.
   - **Filter Tren EMA 200**: Mengharuskan harga `Close > EMA(200)` untuk mencegah *dead-cat bounce* saat kondisi makro berada dalam tren turun.
   - **State Freshness & Single-Shot**: Sinyal hanya valid pada 4 lilin pertama (`state_age <= 4`) sejak transisi ke status Bullish dan dibatasi 1 kali perdagangan per episode rezim (*anti-overtrading*).

2. **Pilar 2: Micro Timing Pullback Trigger**
   - Menghindari pembelian di pucuk momentum dengan menunggu terjadinya pantulan mikro (*pullback*):
     $$\text{Low} \le \text{EMA}(9) \quad \text{ATAU} \quad \text{RSI}(14) \le 52.0$$

3. **Pilar 3: Dynamic Risk & Scaling Out 50:50**
   - **Eksekusi Pasif (Maker Limit Tier)**: Order beli dipasang pada antrean limit order di level EMA(9) untuk mendapatkan *maker fee* rendah (0.02%) dan slippage 0.0%. Jika order tidak terisi dalam 1 candle berikutnya, order otomatis dibatalkan (*anti-stale*).
   - **Manajemen Risiko Bertahap (50:50)**:
     - **TP1 (+0.80x ATR)**: Menjual 50% alokasi posisi, mengamankan profit pertama, dan otomatis menggeser (*ratchet*) Stop Loss sisa 50% ke **Hard-Floored Break-Even** ($\text{Entry} \times 1.0025$, menjamin biaya transaksi bursa terkunci bebas risiko).
     - **TP2 (+1.20x ATR)**: Menjual sisa 50% posisi untuk memaksimalkan *run-up* tren.
     - **Initial Stop Loss (-0.70x ATR)**: Membatasi risiko kerugian secara disiplin jika pasar berbalik arah.
     - **Emergency Exit**: Jika HMM mendeteksi transisi ke State 3 (*Bearish Dump*), seluruh posisi aktif segera ditutup di pasar.

---

## 📊 Hasil Backtest Out-of-Sample (OOS)

Strategi telah divalidasi pada data historis Out-of-Sample (OOS) BTC/USDT 1H tanpa kebocoran data (*strictly out-of-sample partition*):

| Metrik Evaluasi | Nilai Kuantitatif | Keterangan |
| :--- | :---: | :--- |
| **Dataset Periode OOS** | **1.620 candle** | 25 Juni 2026 s/d 31 Agustus 2026 (Partisi Uji Murni) |
| **Total Transaksi** | **10 trades** | Selektif, zero overtrading |
| **Win Rate (%)** | **50.00%** | 5 Menang, 5 Kalah |
| **Profit Factor** | **1.45** | Rasio keuntungan kotor terhadap kerugian kotor |
| **Net PnL** | **+0.18 USDT** | Terverifikasi setelah potongan komisi bursa nyata |
| **Maksimum Drawdown (Max DD)** | **0.45%** | Proteksi modal sangat ketat |
| **Sharpe Ratio (Annualized)** | **+0.82** | Kinerja imbal hasil yang disesuaikan dengan risiko |
| **Alokasi Modal per Trade** | **$30.00 USDT** | 30% dari modal dasar portofolio ($100.00 USDT) |

### Rincian Eksekusi Keluar (*Exit Breakdown*):
- **Take Profit (TP2 Full)**: 4 trades (40.00%)
- **Stop Loss (Initial SL)**: 4 trades (40.00%)
- **Break-Even SL (BE Lock)**: 1 trade (10.00%)
- **Time Expiry Barrier**: 1 trade (10.00%)

---

## 📁 Struktur Repositori

Arsitektur kode dibangun menggunakan prinsip modularitas bersih dan **Single Source of Truth (SSOT)**:

```text
adaptive-trading-system/
├── Dockerfile                  # Container build python:3.11-slim & kompilasi TA-Lib C
├── docker-compose.yml          # Konfigurasi container service & persistence volume
├── requirements.txt            # Dependensi Python terverifikasi untuk Python 3.11
├── .env.example                # Template konfigurasi variabel lingkungan aman
├── .gitignore                  # Filter ketat isolasi kredensial dan cache
├── AGENTS.md                   # Dokumentasi panduan agen AI & repo overview
│
├── src/                        # SOURCE CODE UTAMA (SSOT ENGINE)
│   ├── config/                 # Sentralisasi Konfigurasi
│   │   ├── __init__.py         # Root AppConfig, REGIME_FUNNEL singleton export
│   │   └── strategy.py         # Dataclass RegimeFunnelConfig & env loaders
│   ├── strategies/             # Logika Kuantitatif Terpadu (SSOT)
│   │   ├── __init__.py         # Export RegimeFunnelStrategy
│   │   └── regime_funnel.py    # Perhitungan Indikator, HMM Online, Gates, Risk Brackets
│   ├── execution/              # Engine Eksekusi Bursa
│   │   └── live_bot.py         # CCXT Binance hourly cycle, maker orders, tick monitor
│   └── core/                   # Enums, Data structures
│
├── scripts/                    # SKRIP OPERASIONAL & RISET
│   ├── check_status.py         # Real-time CLI status dashboard & floating PnL monitor
│   ├── historical_paper_trader.py # Bar-by-bar historical market replayer di terminal
│   ├── backtest_engine.py      # Event-driven backtesting engine dengan friksi riil
│   └── train_regime_model.py   # Pipeline pelatihan model Gaussian HMM
│
├── models/                     # MODEL MACHINE LEARNING
│   ├── btc_1h_regime_hmm.joblib          # Bobot Gaussian HMM 1H terverifikasi
│   └── btc_1h_regime_hmm_metadata.json   # Metadata fitur mikrostruktur & state ID
│
├── data/                       # STATE RUNTIME (Lokal Host Persisten, Git-Ignored)
│   └── live_bot_state.json     # Tiket posisi aktif, pending order & history
│
└── tests/                      # UNIT TEST SUITE
    ├── test_regime_funnel.py   # Pengujian SSOT Corong 3-Pilar
    ├── test_main.py            # Pengujian LiveBot, state reload & scaling out
    ├── test_config.py          # Pengujian integritas konfigurasi
    └── test_backtest_engine.py # Pengujian engine backtest
```

---

## 🚀 Panduan Instalasi & Deployment

### 1. Konfigurasi Lingkungan (`.env`)
Salin template konfigurasi dan sesuaikan kredensial API Binance Anda:
```bash
cp .env.example .env
```

Sunting file `.env`:
```env
BINANCE_API_KEY=kredensial_api_key_anda
BINANCE_SECRET_KEY=kredensial_secret_key_anda

# Set true untuk akun Binance Testnet, false untuk akun riil Mainnet
IS_TESTNET=true

# Alokasi modal trading
CAPITAL_TOTAL=100.0
TRADE_ALLOCATION=30.0
```

---

### 2. Menjalankan via Docker Compose
Jalankan bot trading dalam container terisolasi yang berjalan di latar belakang:
```bash
# Build dan jalankan container
docker compose build --no-cache
docker compose up -d

# Memeriksa log aktivitas bot secara langsung
docker compose logs -f --tail 50
```

---

### 3. Pemantauan Real-Time (Terminal Dashboard)
Gunakan skrip CLI ringan [`scripts/check_status.py`](file:///home/rei/reinn/projects/adaptive-trading-system/scripts/check_status.py) untuk memantau status operasional bot, floating PnL posisi terbuka, dan riwayat transaksi:

```bash
# Menjalankan pemantauan interaktif (refresh otomatis setiap 5 detik)
python3 scripts/check_status.py --watch 5
```

**Tampilan Dashboard Terminal:**
```text
==========================================================================
               ADAPTIVE SYSTEM - LIVE BOT STATUS MONITOR               
==========================================================================
  Last Heartbeat : 2026-09-22 02:08:32 UTC (15s ago)
  Target Market  : BTC/USDT [1h] | Environment: TESTNET
  Market Regime  : State 0 (BULLISH MOMENTUM) | Age: 2 bar(s)
  Macro Gate     : GATE OPEN / ACTIVE
--------------------------------------------------------------------------
  [ACTIVE POSITION]
    Status       : IN POSITION | TP1 Hit (50% Scaled, BE Locked)
    Position ID  : POS-17900000
    Size         : 0.000175 BTC (Initial: 0.000350 BTC)
    Entry Price  : $85,000.00
    Current Price: $85,950.00
    Floating PnL : +$0.17 (+1.12%) [PROFIT]
    Stop Loss    : $85,212.50 (Risk: Locked in profit) [BE LOCKED]
    Target TP2   : $86,320.00 (Reward: +$0.23 on remaining size)
--------------------------------------------------------------------------
  [PORTFOLIO PERFORMANCE]
    Closed Trades : 4 trades (Win: 3 | Loss: 1 | Win Rate: 75.0%)
    Realized PnL  : +$0.58 USDT
==========================================================================
```

---

### 4. Simulasi Mandiri (Historical Paper Replayer)
Anda dapat memutar ulang data historis baris-demi-baris secara kausal (*sliding window*) untuk menginspeksi animasi keputusan bot:
```bash
# Simulasi 300 candle historis mulai dari bar #1000
python3 scripts/historical_paper_trader.py --start 1000 --bars 300 --delay 0.05
```

---

## 🧪 Verifikasi & Pengujian Unit (QA)

Integritas arsitektur kode diverifikasi melalui rangkaian pengujian unit otomatis menggunakan `pytest`:

```bash
pytest tests/
```

**Hasil Pengujian:**
```text
============================= test session starts ==============================
collected 34 items

tests/test_backtest_engine.py .                                          [  2%]
tests/test_build_features.py ...                                         [ 11%]
tests/test_config.py ......                                              [ 29%]
tests/test_dataset_processor.py ...                                      [ 38%]
tests/test_logging.py ...                                                [ 47%]
tests/test_main.py .....                                                 [ 61%]
tests/test_regime_funnel.py ....                                         [ 73%]
tests/test_risk_management.py .....                                      [ 88%]
tests/test_train_dedication_model.py ..                                  [ 94%]
tests/test_train_regime_model.py ..                                      [100%]

======================== 34 passed, 1 warning in 5.98s =========================
```

---

## ⚠️ Risk Disclaimer

Perdagangan aset kripto memiliki tingkat risiko finansial yang tinggi dan volatilitas yang ekstrem. Proyek ini disediakan untuk tujuan penelitian kuantitatif, pengujian edukasional, dan demonstrasi arsitektur piranti lunak. 

Kinerja masa lalu pada data pengujian historis (*backtest*) bukan merupakan jaminan pasti atas imbal hasil di masa depan. Pengembang tidak bertanggung jawab atas segala bentuk kerugian finansial yang timbul akibat penggunaan sistem ini dalam perdagangan akun riil. Selalu lakukan pengujian menyeluruh di lingkungan **Binance Spot Testnet** sebelum menggunakan modal nyata.
