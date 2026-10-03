# Adaptive Quant Trading System (BTC, ETH, SOL)

> *Autonomous 3-Pilar Multi-Pair Execution Engine with Layer-2 LightGBM Meta-Labeler & Real-time Streamlit Dashboard*

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Docker Compose](https://img.shields.io/badge/Docker%20Compose-Ready%20(Dual--Container)-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit%201.32%20(Port%208501)-FF4B4B.svg?logo=streamlit&logoColor=white)](http://localhost:8501)
[![Tests](https://img.shields.io/badge/Tests-48%2F48%20Passing%20(100%25)-brightgreen.svg)](https://docs.pytest.org/)
[![Exchange](https://img.shields.io/badge/Binance-Spot%20CCXT-F0B90B.svg?logo=binance&logoColor=black)](https://binance.com)
[![Architecture](https://img.shields.io/badge/Architecture-SSOT%20Clean%20Design-purple.svg)](docs/ARCHITECTURE.md)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Sistem perdagangan aset kripto kuantitatif otonom kelas institusional berbasis **Single Source of Truth (SSOT)** dan **Zero-Setup Plug & Play**. Sistem ini memindai secara simultan pasangan multi-aset terlikuid (**BTC/USDT**, **ETH/USDT**, **SOL/USDT**) pada timeframe **1 Jam (1H)** di bursa **Binance Spot**.

Arsitektur sistem memisahkan proses menjadi dua kontainer independen: bot eksekusi kuantitatif berbasis HMM, Kalman Filter, GARCH, dan LightGBM Meta-Labeler, serta antarmuka pemantauan visual terisolasi (*read-only*) menggunakan Streamlit Dashboard.

---

## ⚡ Quickstart (Zero-Setup 3 Langkah)

Sistem telah dirancang agar pengguna baru dapat langsung menjalankan bot tanpa instalasi dependensi lokal:

### 1. Klon Repositori
```bash
git clone https://github.com/ReinnentRZ/adaptive-trading-system.git
cd adaptive-trading-system
```

### 2. Konfigurasi Kredensial (`.env`)
Salin file template `.env.example` ke `.env`:
```bash
cp .env.example .env
```
Sunting `.env` dan masukkan API Key Binance Anda:
```env
# ==========================================
# BINANCE CREDENTIALS
# ==========================================
BINANCE_API_KEY=masukkan_api_key_anda
BINANCE_API_SECRET=masukkan_api_secret_anda
BINANCE_SANDBOX=true   # Tetapkan 'true' untuk Testnet Sandbox, 'false' untuk Live Mainnet

# ==========================================
# PORTFOLIO & RISK ALLOCATION
# ==========================================
SYMBOLS=BTC/USDT,ETH/USDT,SOL/USDT
TRADE_ALLOCATION=100.0 # Alokasi $100 per posisi
MAX_POSITIONS=3        # Kuota maksimal 3 posisi simultan

# ==========================================
# LAYER-2 AI META-LABELER
# ==========================================
USE_META_LABELER=true
META_LABEL_THRESHOLD=0.50
META_MODEL_PATH=models/multi_asset_1h_funnel_metalabeler.joblib
```

### 3. Jalankan Kontainer via Docker Compose
```bash
docker compose up -d --build
```

### 4. Buka Web Dashboard
Akses pemantauan visual portofolio, status 3 slot posisi, probabilitas AI, dan chart ekuitas realtime di browser Anda:
```text
http://localhost:8501
```

---

## 🏗️ Arsitektur Dual-Container (Decoupled System)

Sistem berjalan secara terisolasi menjadi dua layanan decoupled di Docker Compose guna memastikan stabilitas eksekusi trading:

```
                      ┌────────────────────────────────────────┐
                      │          Docker Compose Host           │
                      └──────────────────┬─────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
┌─────────────────────────────────┐             ┌─────────────────────────────────┐
│       adaptive-trading-bot      │             │    adaptive-trading-dashboard   │
│  (Engine Eksekusi Kuantitatif)  │             │   (Read-Only Streamlit Web UI)  │
├─────────────────────────────────┤             ├─────────────────────────────────┤
│ • Sequential Multi-Pair Scanner │             │ • Visualisasi 3-Slot Portofolio │
│ • HMM 4-State Regime Classifier │             │ • Gauge Probabilitas AI         │
│ • Kalman & GARCH Dynamic Risk   │             │ • Plotly Interactive PnL Curve  │
│ • LightGBM Meta-Labeler Gate    │             │ • Status Heartbeat & Telemetri  │
│ • CCXT Binance Rate-Limiter     │             │ • Web UI Port: 8501             │
│ • Volume: Read/Write data/      │             │ • Volume: Read-Only data/       │
└────────────────┬────────────────┘             └────────────────┬────────────────┘
                 │                                               │
                 └──────────────► [data/live_bot_state.json] ◄───┘
                            (Auto-Bootstrapped Persistence)
```

1. **`adaptive-trading-bot` (Trading Engine Worker)**:
   - Menjalankan loop pemindaian sekuensial per lilin 1H untuk pasangan `BTC/USDT`, `ETH/USDT`, dan `SOL/USDT`.
   - Mengaplikasikan jeda `0.5s` antar pasangan koin untuk mematuhi *weight rate-limit* Binance.
   - Mengisolasi penanganan kesalahan (*error isolation*) per pasangan aset agar kegagalan satu koin tidak mengganggu koin lainnya.
   - Memperbarui tiket posisi aktif dan riwayat pada ledger persistensi `data/live_bot_state.json`.

2. **`adaptive-trading-dashboard` (Read-Only Streamlit UI)**:
   - Terisolasi penuh dari eksekusi order (hanya me-mount folder `data/` secara *read-only*).
   - Menghindari risiko crash bot trading saat dashboard diakses atau di-refresh oleh pengguna.
   - Dapat diakses langsung melalui peramban web pada port `8501`.

---

## 🏛️ Karakteristik Strategi & Logika Corong 3-Pilar

Sistem menyaring sinyal trading melalui mekanisme berjenjang (*Cascading Funnel*) untuk memastikan hanya peluang dengan ekspektansi matematis positif yang dieksekusi:

```
                            [ Binance 1H OHLCV Ingestion ]
                            ( BTC/USDT | ETH/USDT | SOL/USDT )
                                           │
                                           ▼
                 ┌──────────────────────────────────────────────────┐
                 │   PILAR 1: Macro Regime Funnel (Gaussian HMM)    │
                 │   • Causal Gaussian HMM Online Inference == State 0│
                 │   • Close > EMA(200) Macro Structural Trend      │
                 │   • State Freshness: state_age <= 4 candle       │
                 │   • Single-Shot Lock per Episode Bullish         │
                 └─────────────────────────┬────────────────────────┘
                                           │ (Lolos Pilar 1)
                                           ▼
                 ┌──────────────────────────────────────────────────┐
                 │   PILAR 2: Micro Dynamic Trend & Volatility      │
                 │   • Strict Causal 1D Kalman Filter Denoising     │
                 │   • GARCH(1,1) Volatility Bracket Forecasting    │
                 │   • Micro Pullback: Low <= EMA(9) ATAU RSI <= 52 │
                 └─────────────────────────┬────────────────────────┘
                                           │ (Kandidat Sinyal Terbentuk)
                                           ▼
                 ┌──────────────────────────────────────────────────┐
                 │   GERBANG SEKUNDER: Layer-2 AI Meta-Labeler      │
                 │   • 9 Scale-Invariant Features + Macro BTC Return│
                 │   • Calibrated LightGBM Probability Model        │
                 │   • Syarat Eksekusi: P(Win) >= 0.50              │
                 └─────────────────────────┬────────────────────────┘
                                           │ (Disetujui AI & Kuota Slot Ada)
                                           ▼
                 ┌──────────────────────────────────────────────────┐
                 │   PILAR 3: Asymmetric Trade Execution            │
                 │   • Maker Limit Order at EMA(9) (Fee 0.02%)      │
                 │   • Initial Stop Loss : Entry - (0.70 * ATR)     │
                 │   • Take Profit 1     : Entry + (0.80 * ATR)     │
                 │     └─ Tutup 50% Posisi, Kunci SL ke BE (1.0025) │
                 │   • Take Profit 2     : Entry + (1.20 * ATR)     │
                 │     └─ Tutup sisa 50% Posisi (Full Profit)       │
                 │   • Emergency Exit    : HMM State 3 (Bear Dump)  │
                 └──────────────────────────────────────────────────┘
```

* **Pilar 1 (Macro Regime Funnel)**: Memetakan kondisi pasar secara kausal murni (*forward-filtering*) menggunakan Gaussian Hidden Markov Model ke dalam 4 state. Transaksi hanya diizinkan pada `State 0 (Bullish Momentum)` dengan harga di atas `EMA(200)`.
* **Pilar 2 (Micro Dynamic Trend & Volatility)**: Menggunakan **1D Kalman Filter** untuk menghilangkan noise harga tanpa lag, **GARCH(1,1)** untuk memproyeksikan volatilitas lilin berikutnya, dan pemicu *pullback* mikro (`EMA 9` / `RSI 14 <= 52.0`) untuk menghindari pembelian di pucuk momentum.
* **Pilar 3 (Asymmetric Execution & Scaling-Out)**: Menempatkan order beli pasif (*Maker Limit*) di level EMA(9). Mengamankan 50% profit di TP1 ($+0.80 \times \text{ATR}$) sambil menaikkan Stop Loss ke *Hard-Floored Break-Even* ($\text{Entry} \times 1.0025$). Sisa posisi ditutup di TP2 ($+1.20 \times \text{ATR}$) atau dilindungi oleh SL/Emergency Exit.
* **Layer-2 Gate (LightGBM Meta-Labeler)**: Model ensemble pohon keputusan terlatih yang mengevaluasi kandidat teknikal. Jika probabilitas kemenangan terkalibrasi $P(\text{Win}) < 0.50$, order otomatis dibatalkan (*anti-false-breakout*).

---

## 📊 Hasil Benchmark Historis 14.5 Bulan

Pengujian historis *out-of-sample continuous replay* (Januari 2025 – Oktober 2026, ~440 hari) pada data lilin 1H BTC, ETH, dan SOL dengan potongan komisi bursa nyata:

| Metrik Evaluasi | Nilai Kuantitatif | Keterangan |
| :--- | :---: | :--- |
| **Modal Kerja Awal** | **$387.50 USD** | Setara modal Rp 6.200.000 (3 slot @ $100) |
| **Total Transaksi** | **285 trades** | Rata-rata 19–20 trade per bulan |
| **Win Rate Bersih** | **~70.0%** | Tervalidasi dengan Layer-2 Meta-Labeler |
| **Profit Factor** | **3.06** | Asimetri keuntungan kotor terhadap kerugian kotor |
| **Net PnL Bersih** | **+$20.19 USD (+5.21%)** | Bersih setelah komisi maker fee 0.04% roundtrip |
| **Maksimum Drawdown (MDD)** | **< 2.0%** | Sangat aman dari risiko kehancuran modal (*anti-ruin*) |
| **Instrumen Perdagangan** | **Binance Spot** | 0% risiko likuidasi (*No Leverage*) |

---

## 🛠️ CLI & Operational Cheat Sheet

### Manajemen Kontainer Docker
```bash
# Memulai seluruh layanan (Bot + Dashboard) di latar belakang
docker compose up -d

# Memeriksa log streaming eksekusi bot secara realtime
docker compose logs -f --tail 50 adaptive-trading-bot

# Memeriksa log dashboard Streamlit
docker compose logs -f --tail 50 trading-dashboard

# Menghentikan seluruh kontainer
docker compose down
```

### Eksekusi Pengujian & Verifikasi Lokal (QA)
```bash
# Menjalankan 48 unit test otomatis di dalam lingkungan kontainer Docker
docker compose run --rm adaptive-trading-bot pytest tests/ -v

# Menjalankan linter flake8 pada kode sumber lokal
flake8 . --count --select=E9,F63,F7,F82 --exclude=.venv,venv,.git,__pycache__,build,dist
```

### Inspeksi Status Posisi Cepat via CLI (Terminal)
```bash
# Memantau status posisi aktif, floating PnL, dan kuota slot dari terminal host
python3 scripts/check_status.py --watch 5
```

---

## 📖 Dokumentasi Lanjutan
Untuk rincian arsitektur mendalam, tata kelola state ledger, serta Prosedur Operasional Standar (SOP) retraining model berkala, silakan baca [**Dokumentasi Arsitektur Sistem (`docs/ARCHITECTURE.md`)**](docs/ARCHITECTURE.md).

---

## ⚠️ Risk Disclaimer

> Perdagangan aset kripto memiliki tingkat risiko finansial yang tinggi dan volatilitas pasar yang ekstrem. Perangkat lunak ini disediakan untuk tujuan riset kuantitatif, otomasi algoritma, dan rekayasa perangkat lunak finansial. 
> 
> Kinerja historis (*backtest*) tidak menjamin hasil perdagangan di masa depan. Pengembang tidak bertanggung jawab atas kerugian finansial yang timbul akibat penggunaan sistem ini dalam perdagangan riil. Selalu lakukan pengujian menyeluruh di lingkungan **Binance Spot Testnet (Sandbox)** sebelum mengalokasikan modal nyata.
