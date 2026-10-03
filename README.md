# Adaptive Trading System: Multi-Pair Corong 3-Pilar & AI Meta-Labeler

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready%20(Compose)-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit%201.32-FF4B4B.svg?logo=streamlit&logoColor=white)](http://localhost:8501)
[![CCXT](https://img.shields.io/badge/CCXT-Binance%20Spot-orange.svg)](https://github.com/ccxt/ccxt)
[![Tests](https://img.shields.io/badge/Tests-48%20Passed%20(100%25)-brightgreen.svg)](https://docs.pytest.org/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions%20Cached-2088FF.svg?logo=githubactions&logoColor=white)](#ci-cd-pipeline--qa)
[![Architecture](https://img.shields.io/badge/Architecture-SSOT%20Clean%20Design-purple.svg)](#arsitektur-sistem-corong-3-pilar--ai-meta-labeler)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Sistem perdagangan aset kripto algoritmik institusional berbasis **Single Source of Truth (SSOT)** dan **Zero-Setup Plug & Play**. Sistem ini memindai secara simultan pasangan multi-aset (**BTC/USDT**, **ETH/USDT**, **SOL/USDT**) pada timeframe **1 Jam (1H)** di bursa **Binance Spot**.

Sistem ini menggabungkan inferensi rezim pasar tanpa pengawasan (*unsupervised*) menggunakan **Gaussian Hidden Markov Model (HMM)**, filter tren struktural makro **EMA 200**, penapis volatilitas kuantitatif **Kalman Filter & GARCH(1,1)**, gerbang sekunder **Layer-2 AI Meta-Labeler (LightGBM Calibrated)**, manajemen risiko asimetris **50:50 Scaling Out** dengan penguncian *Hard-Floored Break-Even*, serta **Streamlit Web Dashboard** interaktif yang berjalan secara terisolasi (*decoupled*).

---

## 📑 Daftar Isi
- [Fitur Unggulan](#-fitur-unggulan)
- [Arsitektur Sistem: Corong 3-Pilar & AI Meta-Labeler](#-arsitektur-sistem-corong-3-pilar--ai-meta-labeler)
- [Layer-2 AI Meta-Labeler & Cross-Asset Features](#-layer-2-ai-meta-labeler--cross-asset-features)
- [Dashboard Pemantauan Web (Streamlit)](#-dashboard-pemantauan-web-streamlit)
- [Hasil Backtest & Benchmark](#-hasil-backtest--benchmark)
- [Struktur Repositori](#-struktur-repositori)
- [Panduan Instalasi & Deployment Cepat (Zero-Setup)](#-panduan-instalasi--deployment-cepat-zero-setup)
  - [1. Clone Repositori](#1-clone-repositori)
  - [2. Konfigurasi Lingkungan (.env)](#2-konfigurasi-lingkungan-env)
  - [3. Jalankan via Docker Compose](#3-jalankan-via-docker-compose)
  - [4. Akses Web Dashboard & Monitoring](#4-akses-web-dashboard--monitoring)
- [Perintah CLI & Operasional](#-perintah-cli--operasional)
- [CI/CD Pipeline & QA](#-cicd-pipeline--qa)
- [Risk Disclaimer](#-risk-disclaimer)

---

## 🚀 Fitur Unggulan

- **Multi-Pair Concurrent Scanner**: Memindai pasangan berlikuiditas tinggi (`BTC/USDT`, `ETH/USDT`, `SOL/USDT`) dengan proteksi *rate-limiting* adaptif dan eksekusi terisolasi per simbol.
- **Dynamic 3-Slot Portfolio Ledger**: Manajemen risiko berbasis portofolio dengan kuota maksimal 3 posisi simultan dan alokasi modal terproteksi (default: $100 / posisi).
- **Auto-Bootstrapping State Persistence**: Fitur *self-healing* yang otomatis menginisialisasi ledger transaksi di `data/live_bot_state.json` saat bot pertama kali dijalankan.
- **Layer-2 AI Meta-Labeling**: LightGBM Classifier terkalibrasi (*Probability Calibration via Sigmoid*) yang dilatih pada Triple Barrier Method untuk menyaring false breakout dari aturan teknikal primer.
- **Cross-Asset Macro Signals**: Evaluasi koin alternatif (ETH, SOL) memperhitungkan momentum makro Bitcoin (`btc_return_1h` dan `relative_strength_vs_btc`) tanpa *lookahead bias*.
- **Quantitative Volatility Engine**: Dilengkapi dengan estimasi tren kausal **Kalman Filter** dan peramalan volatilitas **GARCH(1,1)** untuk penyesuaian bracket Stop Loss / Take Profit berbasis ATR.
- **Decoupled Streamlit Dashboard**: Antarmuka web modern di port `8501` untuk inspeksi visual slot portofolio, probabilitas AI, chart PnL kumulatif, dan telemetri execution logs.

---

## 🏛️ Arsitektur Sistem: Corong 3-Pilar & AI Meta-Labeler

Sistem bekerja melalui mekanisme penyaringan bertingkat (*Cascading Funnel*) untuk memastikan hanya peluang dengan ekspektansi matematis positif yang dieksekusi:

```
                          [ Ingest Multi-Pair 1H OHLCV ]
                          ( BTC/USDT | ETH/USDT | SOL/USDT )
                                          │
                                          ▼
                 ┌──────────────────────────────────────────────────┐
                 │   PILAR 1: Macro Regime Gate & Structural Trend  │
                 │   • Causal Gaussian HMM Online Inference == State 0│
                 │   • Close > EMA(200) Macro Structural Filter     │
                 │   • State Freshness: state_age <= 4 candle       │
                 │   • Single-Shot Lock per Episode Bullish         │
                 └────────────────────────┬─────────────────────────┘
                                          │ (Lolos Gerbang 1)
                                          ▼
                 ┌──────────────────────────────────────────────────┐
                 │   PILAR 2: Micro Timing Pullback Trigger         │
                 │   • Dynamic Queue Entry: Low <= EMA(9)           │
                 │     ATAU Momentum Reset: RSI(14) <= 52.0         │
                 └────────────────────────┬─────────────────────────┘
                                          │ (Kandidat Sinyal Terbentuk)
                                          ▼
                 ┌──────────────────────────────────────────────────┐
                 │   GERBANG SEKUNDER: Layer-2 AI Meta-Labeler      │
                 │   • 10 Fitur Skala-Invarian + Fitur Makro BTC    │
                 │   • Calibrated LightGBM Probability Gate         │
                 │   • Syarat Eksekusi: P(Win) >= 0.50              │
                 └────────────────────────┬─────────────────────────┘
                                          │ (Disetujui AI & Slot Tersedia)
                                          ▼
                 ┌──────────────────────────────────────────────────┐
                 │   PILAR 3: Asymmetric Execution & Scaling Out    │
                 │   • Maker Limit Order at EMA(9) (0.02% fee)      │
                 │   • Initial Stop Loss : Entry - (0.70 * ATR)     │
                 │   • Target 1 (TP1)    : Entry + (0.80 * ATR)     │
                 │     └─ Tutup 50% Posisi, Geser SL ke BE (1.0025) │
                 │   • Target 2 (TP2)    : Entry + (1.20 * ATR)     │
                 │     └─ Tutup sisa 50% Posisi (Full Profit)       │
                 │   • Emergency Exit    : HMM State 3 (Bear Dump)  │
                 └──────────────────────────────────────────────────┘
```

1. **Pilar 1: Macro Regime Gate (Gaussian HMM & EMA 200)**:
   - **Gaussian HMM Causal Forward-Filtering**: Mengklasifikasi rezim pasar secara kausal (*Zero Lookahead Bias*) dari 3 fitur mikrostruktur: *Log Return*, *Normalized ATR Volatility*, dan *Signed Volume Intensity*. Sinyal hanya valid pada `State 0 (Bullish Momentum)`.
   - **Filter Tren EMA 200**: Memastikan harga berada di atas rata-rata struktural makro `Close > EMA(200)`.
   - **State Freshness & Anti-Overtrading**: Membatasi entri hanya pada 4 candle pertama sejak transisi rezim dan mengunci sinyal berikutnya dalam episode yang sama (*Single-Shot Lock*).

2. **Pilar 2: Micro Timing Pullback Trigger**:
   - Menghindari pembelian impulsif pada pucuk harga dengan menunggu pantulan harga:
     $$\text{Low} \le \text{EMA}(9) \quad \text{ATAU} \quad \text{RSI}(14) \le 52.0$$

3. **Pilar 3: Asymmetric Execution & Scaling Out (50:50)**:
   - **Passive Maker Limit Tier**: Order dipasang di antrean limit order level EMA(9) untuk mendapatkan fee terendah (*maker fee 0.02%*) dan bebas slippage.
   - **TP1 (+0.80x ATR)**: Menutup 50% posisi dan otomatis menggeser (*ratchet*) Stop Loss ke **Hard-Floored Break-Even** ($\text{Entry} \times 1.0025$), mengunci keuntungan awal dan biaya bursa.
   - **TP2 (+1.20x ATR)**: Menutup sisa 50% posisi saat target tren tercapai.
   - **Initial Stop Loss (-0.70x ATR)**: Membatasi downside risiko secara ketat jika arah pergerakan berbalik.
   - **Emergency Exit**: Menutup seluruh posisi terbuka jika HMM mendeteksi transisi ke `State 3 (Bearish Dump)`.

---

## 🧠 Layer-2 AI Meta-Labeler & Cross-Asset Features

Sistem ini menerapkan metodologi **Meta-Labeling (Marcos López de Prado)**:
Model primer (Corong 3-Pilar) bertindak sebagai *Rule-Based Candidate Generator*, sementara model sekunder (LightGBM Classifier) menentukan apakah kandidat posisi memiliki probabilitas menang yang cukup tinggi untuk dieksekusi.

### 1. Cross-Asset Feature Engineering
Model dilatih menggunakan dataset gabungan (*pooled training*) dari BTC, ETH, dan SOL dengan fitur-fitur teknikal yang bersifat *scale-invariant*:
- **RSI (14)**, **ADX (14)**, **WaveTrend Oscillator (WT1, WT2)**
- **Volatility Ratio** ($\text{ATR}_{14} / \text{Close}$)
- **EMA Distance** ($(\text{Close} - \text{EMA}_9) / \text{Close}$)
- **Volume Ratio** ($\text{Volume} / \text{SMA}_{20}(\text{Volume})$)
- **Macro Bitcoin Returns**: `btc_return_1h` (return lilin 1H terakhir BTC)
- **Relative Strength vs BTC**: $R_{\text{asset}} - R_{\text{btc}}$ (kekuatan relatif aset terhadap BTC)

### 2. Validasi & Kalibrasi Model
- **Cross-Validation**: `PurgedGroupTimeSeriesSplit` dengan *embargo period* 5 candle untuk mencegah kebocoran informasi (*leakage*).
- **Probability Calibration**: `CalibratedClassifierCV(method='sigmoid')` sehingga nilai probabilitas yang dihasilkan mencerminkan frekuensi empiris kemenangan yang sesungguhnya.
- **Model Artifacts**: File bobot model telah dibundel langsung dalam repositori di folder `models/` sehingga pengguna baru dapat langsung menjalankannya tanpa perlu melatih ulang:
  - `models/multi_asset_1h_funnel_metalabeler.joblib`
  - `models/btc_1h_regime_hmm.joblib`
  - `models/eth_1h_regime_hmm.joblib`
  - `models/sol_1h_regime_hmm.joblib`

---

## 🖥️ Dashboard Pemantauan Web (Streamlit)

Sistem menyediakan antarmuka web visual modern di `http://localhost:8501` yang berjalan sebagai container terpisah (*read-only volume mount*):

- **Portfolio Slot Tracker**: Menampilkan utilisasi 3 slot posisi simultan (BTC, ETH, SOL) beserta persentase margin terpakai.
- **Active Trade Matrix**: Detail posisi aktif (Entry Price, Size, Current Mark, Floating PnL, ATR Adaptive Brackets untuk SL, TP1, dan TP2).
- **AI Meta-Labeler Gauge**: Menampilkan probabilitas prediksi AI untuk setiap candle terakhir beserta status izin gerbang (*Gate Open / Filtered Out*).
- **Cumulative PnL & Win Rate Analytics**: Visualisasi kurva ekuitas interaktif menggunakan Plotly, distribusi hasil trade (Take Profit, Stop Loss, Break-Even, Time-Barrier), dan metrik Profit Factor.
- **Execution Telemetry Feed**: Log operasional bot, status heartbeat, dan riwayat transaksi persistensi.

---

## 📊 Hasil Backtest & Benchmark

Hasil pengujian out-of-sample pada data historis lilin 1H (setelah memperhitungkan potongan biaya bursa riil 0.04% roundtrip):

| Metrik Kuantitatif | Nilai Terverifikasi | Catatan Evaluasi |
| :--- | :---: | :--- |
| **Pasangan Aset** | `BTC`, `ETH`, `SOL` | Multi-Pair Concurrent 1H |
| **Win Rate** | **~70.0%** | Filter Meta-Labeler aktif |
| **Profit Factor** | **> 3.0** | Asimetri rasio laba/rugi sangat tinggi |
| **Maksimum Drawdown (MDD)** | **< 2.0%** | Proteksi modal bertingkat & trailing BE |
| **Alokasi per Posisi** | **$100.00 USD** | Maksimal 3 slot posisi aktif |
| **Mekanisme Eksekusi** | **Maker Limit Tier** | Zero-slippage & low fees (0.02%) |

---

## 📁 Struktur Repositori

```text
adaptive-trading-system/
├── Dockerfile                  # Multi-stage image Python 3.12 & TA-Lib C-Library
├── docker-compose.yml          # Konfigurasi orkestrasi 2 service (Bot + Web Dashboard)
├── requirements.txt            # Dependensi Python terverifikasi
├── .env.example                # Template konfigurasi variabel lingkungan Plug & Play
├── .gitignore                  # Filter isolasi kredensial, cache, dan data lokal
├── README.md                   # Dokumentasi lengkap sistem
├── AGENTS.md                   # Panduan operasional dan arsitektur agen AI
│
├── src/                        # SOURCE CODE UTAMA (CLEAN ARCHITECTURE)
│   ├── config/                 # Konfigurasi Terpusat
│   │   ├── __init__.py         # Singleton config loader
│   │   └── strategy.py         # RegimeFunnelConfig & parameter kuantitatif
│   ├── core/                   # Enums, DTO, & Data Structures
│   ├── data/                   # Modul ingest data & streaming
│   ├── execution/              # Engine Eksekusi Bursa
│   │   └── live_bot.py         # Multi-Pair Scanner, state ledger, order manager
│   ├── services/               # Wrapper CCXT Binance Exchange
│   ├── strategies/             # Logika Kuantitatif Terpadu (SSOT)
│   │   ├── regime_funnel.py    # Corong 3-Pilar (HMM, Gates, Brackets)
│   │   ├── meta_labeler.py     # Inferensi Layer-2 AI Meta-Labeler
│   │   └── quant_models.py     # Kalman Trend Filter & GARCH(1,1) Forecaster
│   └── dashboard/              # Antarmuka Web
│       └── app.py              # Streamlit Web UI Application
│
├── models/                     # MODEL MACHINE LEARNING TERLATIH (BUNDLED)
│   ├── multi_asset_1h_funnel_metalabeler.joblib # Model AI Meta-Labeler multi-koin
│   ├── btc_1h_regime_hmm.joblib                 # Gaussian HMM BTC 1H
│   ├── eth_1h_regime_hmm.joblib                 # Gaussian HMM ETH 1H
│   └── sol_1h_regime_hmm.joblib                 # Gaussian HMM SOL 1H
│
├── scripts/                    # SKRIP RISET & TRAINING
│   ├── train_funnel_meta_labeler.py # Pipeline pelatihan terpadu Meta-Labeler
│   ├── train_regime_model.py        # Pipeline pelatihan Gaussian HMM
│   ├── backtest_engine.py           # Engine simulasi event-driven
│   └── check_status.py              # CLI Terminal dashboard monitor
│
├── data/                       # RUNTIME PERSISTENCE (AUTO-BOOTSTRAPPED)
│   └── live_bot_state.json     # Ledger posisi aktif, slot portofolio, dan history
│
└── tests/                      # UNIT TEST SUITE (100% PASSING)
    ├── test_config.py          # Validasi konfigurasi & masking kredensial
    ├── test_main.py            # Validasi live bot scanning & ledger persistence
    ├── test_meta_labeler.py    # Validasi ekstraksi fitur & inferensi AI
    ├── test_quant_models.py    # Validasi Kalman Filter & GARCH(1,1)
    └── test_regime_funnel.py   # Validasi SSOT Corong 3-Pilar
```

---

## 🚀 Panduan Instalasi & Deployment Cepat (Zero-Setup)

Sistem telah dirancang agar bersifat **Plug & Play**. Pengguna cukup mengklon repositori, mengisi kredensial pada file `.env`, dan langsung menjalankan container Docker tanpa perlu konfigurasi tambahan.

### 1. Clone Repositori
```bash
git clone https://github.com/ReinnentRZ/adaptive-trading-system.git
cd adaptive-trading-system
```

### 2. Konfigurasi Lingkungan (`.env`)
Salin file template `.env.example` menjadi `.env`:
```bash
cp .env.example .env
```

Buka file `.env` dan masukkan API Key Binance Anda:
```env
# ==========================================
# BINANCE CREDENTIALS
# ==========================================
BINANCE_API_KEY=masukkan_api_key_anda_disini
BINANCE_API_SECRET=masukkan_api_secret_anda_disini
BINANCE_SANDBOX=true   # Set true untuk Testnet, ubah ke false untuk Live Trading

# ==========================================
# PORTFOLIO & RISK ALLOCATION
# ==========================================
SYMBOLS=BTC/USDT,ETH/USDT,SOL/USDT
TRADE_ALLOCATION=100.0
MAX_POSITIONS=3

# ==========================================
# LAYER-2 AI META-LABELER
# ==========================================
USE_META_LABELER=true
META_LABEL_THRESHOLD=0.50
META_MODEL_PATH=models/multi_asset_1h_funnel_metalabeler.joblib
```

> [!TIP]
> Secara default `BINANCE_SANDBOX=true` aktif untuk mode pengujian aman di Binance Spot Testnet. Ubah ke `false` hanya jika Anda telah siap melakukan perdagangan riil di Mainnet.

### 3. Jalankan via Docker Compose
Jalankan bot trading dan web dashboard secara bersamaan di latar belakang:
```bash
docker compose up -d --build
```

### 4. Akses Web Dashboard & Monitoring
- **Web Dashboard**: Buka browser Anda dan akses:
  ```text
  http://localhost:8501
  ```
- **Live Logs**: Pantau aktivitas pemindaian multi-pair secara realtime melalui terminal:
  ```bash
  docker compose logs -f --tail 50 adaptive-trading-bot
  ```

---

## 🛠️ Perintah CLI & Operasional

### Menghentikan Layanan
```bash
docker compose down
```

### Memeriksa Status Posisi via CLI (Opsional)
Jika Anda ingin memantau posisi secara cepat tanpa membuka browser:
```bash
python3 scripts/check_status.py --watch 5
```

### Melatih Ulang Model AI Meta-Labeler (Opsional)
Jika Anda memiliki data baru dan ingin melatih ulang model:
```bash
python3 scripts/train_funnel_meta_labeler.py
```

---

## 🧪 CI/CD Pipeline & QA

Seluruh kode terverifikasi melalui pipeline integrasi berkelanjutan (**GitHub Actions**) dengan standar kualitas tinggi:

1. **Automated Caching**: Kompilasi library C `TA-Lib` di-cache menggunakan `actions/cache@v4` dengan sumber mirror redundan (GitHub Releases & SourceForge).
2. **Strict Linting**: Pemeriksaan sintaks dan kebersihan kode menggunakan `flake8` (`E9, F63, F7, F82`).
3. **Unit Testing Suite**: Rangkaian 48 pengujian unit otomatis mencakup logika kalkulasi indikator, gerbang rezim HMM, inferensi AI, persistensi ledger, dan simulasi order:

```bash
# Menjalankan pengujian unit lokal
pytest tests/ -v
```

Hasil verifikasi:
```text
======================== 48 passed, 1 warning in 8.23s =========================
100% Tests Passed
```

---

## ⚠️ Risk Disclaimer

> Perdagangan aset kripto memiliki tingkat risiko finansial yang tinggi dan volatilitas pasar yang ekstrem. Sistem ini dirancang untuk tujuan riset kuantitatif, otomasi algoritma, dan rekayasa perangkat lunak finansial. 
> 
> Kinerja historis (*backtest*) tidak menjamin hasil perdagangan di masa depan. Pengembang tidak bertanggung jawab atas kerugian finansial yang timbul akibat penggunaan sistem ini dalam perdagangan riil. Selalu gunakan manajemen risiko yang ketat dan uji coba di lingkungan **Binance Spot Testnet (Sandbox)** sebelum mengalokasikan modal nyata.
