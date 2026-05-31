# Bab 3: Metodologi

## 3.1 Dataset

### 3.1.1 Sumber Data

Penelitian ini menggunakan dataset **DSL-StrongPasswordData** yang dikembangkan oleh Killourhy dan Maxion (2009) di Carnegie Mellon University. Dataset ini merupakan benchmark standar dalam riset keystroke dynamics.

**Sumber**: [Killourhy & Maxion Keystroke Dynamics Dataset](https://www.cs.cmu.edu/~keystroke/)

### 3.1.2 Deskripsi Dataset

| Atribut | Nilai |
|---------|-------|
| **Jumlah subjek** | 51 pengguna |
| **Kata sandi** | `.tie5Roanl` (10 karakter + Return) |
| **Repetisi per subjek** | 400 kali (8 sesi × 50 repetisi) |
| **Total observasi** | 20.400 baris |
| **Jumlah kolom** | 34 kolom |
| **Sesi per subjek** | 8 sesi |
| **Repetisi per sesi** | 50 repetisi |

### 3.1.3 Struktur Kolom

Kolom-kolom dataset mengikuti konvensi penamaan berikut:

| Prefiks | Jenis Fitur | Contoh |
|---------|-------------|--------|
| `h.*` | Hold/Dwell time tombol | `h.period`, `h.t`, `h.i`, ... |
| `ud.*.*` | Flight time Up-Down antar dua tombol | `ud.period.t`, `ud.t.i`, ... |
| `dd.*.*` | Flight time Down-Down antar dua tombol | `dd.period.t`, `dd.t.i`, ... |
| `subject` | Identitas pengguna | `s002`, `s003`, ..., `s057` |
| `sessionindex` | Nomor sesi (1-8) | 1, 2, ..., 8 |
| `rep` | Nomor repetisi dalam sesi (1-50) | 1, 2, ..., 50 |

### 3.1.4 Validasi Konsistensi Internal

Dilakukan validasi terhadap relasi matematis $DD = H + UD$ pada 8 pasangan triplet kolom. Hasil menunjukkan:

| Pasangan DD | Jumlah Baris | Mean Absolute Error | Max Absolute Error | Rasio Dalam Toleransi |
|------------|:------------:|:-------------------:|:------------------:|:---------------------:|
| `dd.period.t` | 20.400 | $9.27 \times 10^{-18}$ | $4.44 \times 10^{-16}$ | **100%** |
| `dd.t.i` | 20.400 | $6.20 \times 10^{-18}$ | $4.44 \times 10^{-16}$ | **100%** |
| `dd.i.e` | 20.400 | $5.96 \times 10^{-18}$ | $4.44 \times 10^{-16}$ | **100%** |
| `dd.e.five` | 20.400 | $1.34 \times 10^{-17}$ | $8.88 \times 10^{-16}$ | **100%** |
| `dd.o.a` | 20.400 | $5.76 \times 10^{-18}$ | $4.44 \times 10^{-16}$ | **100%** |
| `dd.a.n` | 20.400 | $5.67 \times 10^{-18}$ | $2.22 \times 10^{-16}$ | **100%** |
| `dd.n.l` | 20.400 | $7.58 \times 10^{-18}$ | $2.22 \times 10^{-16}$ | **100%** |
| `dd.l.return` | 20.400 | $1.13 \times 10^{-17}$ | $8.88 \times 10^{-16}$ | **100%** |

Semua deviasi berada dalam batas presisi floating-point, mengonfirmasi **integritas dataset 100%**.

## 3.2 Arsitektur Sistem

### 3.2.1 Struktur Modular Proyek

Proyek diorganisasi dalam struktur modular sebagai berikut:

```
Projek Keamanan Informasi/
├── data/raw/                       # Dataset mentah
├── data/processed/                 # Bundle sequence terproses
├── src/
│   ├── data/preprocessing.py       # Pipeline preprocessing
│   ├── models/baseline.py          # Arsitektur LSTM
│   ├── evaluation/
│   │   ├── metrics.py              # Metrik evaluasi (EER, FAR, FRR)
│   │   ├── attack_mia.py           # Membership Inference Attack
│   │   ├── leakage_attack.py       # Gradient Leakage Attack
│   │   └── explainability.py       # Integrated Gradients
│   └── visualization/              # Utilitas plotting
├── scripts/
│   ├── run_fl_manual.py            # Orchestrator FL baseline
│   ├── run_fl_dp_manual.py         # Orchestrator FL + DP
│   ├── run_fl_non_iid.py           # Orchestrator Non-IID FL
│   ├── run_fl_transfer.py          # Orchestrator Transfer FL
│   ├── run_leakage_attack.py       # Script serangan leakage
│   ├── run_explainability.py       # Script explainability
│   └── run_ablation.py             # Script ablation study
├── notebooks/modular/              # 10 notebook eksperimen
├── outputs/
│   ├── models/                     # Checkpoint model (.pt)
│   ├── reports/                    # Metrik JSON & CSV
│   └── figures/                    # Visualisasi PNG
└── REPRODUCIBILITY.md              # Panduan reproduksi
```

### 3.2.2 Arsitektur Model LSTM

Model `KeystrokeLSTM` didesain dengan arsitektur berikut:

```
Input Layer
  ↓  (batch_size, 11, 3) - 11 timestep × 3 fitur
LSTM Layer 1
  ↓  hidden_dim = 64
LSTM Layer 2
  ↓  hidden_dim = 64, dropout = 0.3
Take Final Hidden State h_n[-1]
  ↓  (batch_size, 64)
Fully Connected: 64 → 32
  ↓  ReLU + Dropout(0.3)
Fully Connected: 32 → 51
  ↓  (batch_size, 51) - logits untuk 51 kelas
Output (Softmax via CrossEntropyLoss)
```

| Parameter | Nilai |
|-----------|-------|
| `input_dim` | 3 (Dwell, Flight UD, Flight DD) |
| `hidden_dim` | 64 |
| `num_layers` | 2 |
| `num_classes` | 51 |
| `dropout` | 0.3 |
| **Total parameters** | ≈55.000 |

## 3.3 Pipeline Preprocessing

### 3.3.1 Alur Preprocessing

Pipeline preprocessing mengikuti urutan deterministik:

1. **Missing Value Handling**: Imputasi median untuk kolom numerik, mode untuk kolom kategorikal.
2. **Outlier Filtering**: Quantile-based filtering (1%-99%) pada fitur timing, dengan fallback ke data asli jika filtering menghasilkan dataset kosong.
3. **Session Segmentation**: Pengurutan data berdasarkan kolom `sessionindex`.
4. **Temporal Sequence Construction**: Ekstraksi 3 fitur per transisi tombol menghasilkan tensor shape `(N, 11, 3)`:
   - Timestep 0 (`.`): `[H_period, 0, 0]` - tidak ada flight time untuk tombol pertama
   - Timestep $k$ ($k > 0$): `[H_k, UD_{k-1,k}, DD_{k-1,k}]`
5. **Normalization**: StandardScaler opsional (diaktifkan untuk Transfer FL).

### 3.3.2 Pembagian Data

| Split | Proporsi | Jumlah Sampel | Stratifikasi |
|-------|:--------:|:-------------:|:------------:|
| Training | 70% | ~9.969 | Ya (per subjek) |
| Validation | 15% | ~2.136 | Ya (per subjek) |
| Test | 15% | ~2.137 | Ya (per subjek) |

Stratifikasi memastikan setiap subjek terwakili proporsional di setiap split.

### 3.3.3 Output Preprocessing

Preprocessing menghasilkan `sequence_bundle.npz` dengan komponen:
- `features`: Array shape `(14.242, 11, 3)` - setelah outlier filtering dari 20.400 observasi
- `labels`: Array string 14.242 label subjek
- `sequence_ids`: Identifier unik `{subject}_{session}_{rep}`

## 3.4 Desain Eksperimen

### 3.4.1 Konfigurasi Eksperimen

Penelitian mengevaluasi **7 konfigurasi** model:

| # | Konfigurasi | Perlindungan Privasi | Framework |
|---|-------------|---------------------|-----------|
| 1 | **Centralized Baseline LSTM** | Tidak ada | PyTorch |
| 2 | **Centralized DP LSTM** | DP-SGD lokal | PyTorch + Opacus |
| 3 | **Federated Baseline FL** | Data tetap lokal | Flower (gRPC) |
| 4 | **Joint FL + DP** | DP-SGD lokal + FedAvg | Flower + Opacus |
| 5 | **Non-IID FL** | Data lokal + heterogen | Flower (gRPC) |
| 6 | **Advanced Transfer FL** | Pre-trained init + scaling | Flower (gRPC) |
| 7 | **Ablation DP** | Variasi epsilon | Opacus |

### 3.4.2 Hyperparameter Pelatihan

| Parameter | Baseline | DP | FL | FL+DP |
|-----------|:--------:|:--:|:--:|:-----:|
| Learning rate | 0.001 | 0.001 | 0.001 | 0.001 |
| Batch size | 64 | 64 | 64 | 64 |
| Epochs | 30 | 30 | - | - |
| Optimizer | Adam | DP-Adam | Adam | DP-Adam |
| Loss function | CrossEntropy | CrossEntropy | CrossEntropy | CrossEntropy |
| FL Rounds | - | - | 5 | 5 |
| FL Clients | - | - | 5 | 5 |
| Local epochs | - | - | 1 | 1 |
| Noise multiplier | - | 1.1 | - | 1.1 |
| Clipping norm | - | 1.0 | - | 1.0 |
| Target delta | - | $10^{-5}$ | - | $10^{-5}$ |

### 3.4.3 Konfigurasi Federated Learning

**Partisi Data IID**: Data dibagi rata ke 5 klien (~2.278-2.279 sampel/klien) secara acak, menjaga proporsi label seragam.

**Partisi Data Non-IID**: Implementasi heterogenitas ganda:
- **Subject skew**: Setiap klien mendapat subset subjek yang berbeda (non-overlapping).
- **Speed scaling**: Dua klien menerapkan skala kecepatan ($\times 0.75$ dan $\times 1.25$) pada fitur timing, mensimulasikan perbedaan kecepatan mengetik.

| Client | Jumlah Sampel | Speed Scale | Keterangan |
|:------:|:-------------:|:-----------:|------------|
| 0 | 2.437 | 1.25× | Pengetik cepat |
| 1 | 2.379 | 0.75× | Pengetik lambat |
| 2 | 2.153 | 1.00× | Normal |
| 3 | 2.272 | 1.00× | Normal |
| 4 | 2.152 | 1.00× | Normal |

### 3.4.4 Konfigurasi Advanced Transfer FL

Untuk mengatasi rendahnya akurasi FL standar, diterapkan dua teknik:
1. **Pre-trained weight initialization**: Model global diinisialisasi dengan bobot dari `baseline_lstm.pt`, bukan bobot acak.
2. **Feature standardization**: Setiap klien menerapkan `StandardScaler` pada fitur timing sebelum pelatihan lokal, menormalisasi skala input.
3. **Learning rate reduction**: Menggunakan $lr = 0.0001$ (10× lebih kecil) untuk fine-tuning.

## 3.5 Metrik Evaluasi

### 3.5.1 Metrik Utilitas

| Metrik | Definisi | Formula |
|--------|----------|---------|
| **Accuracy** | Proporsi prediksi benar | $\frac{TP + TN}{Total}$ |
| **Precision** | Proporsi prediksi positif yang benar | $\frac{TP}{TP + FP}$ |
| **Recall** | Proporsi positif aktual yang terdeteksi | $\frac{TP}{TP + FN}$ |
| **F1-Score** | Harmonik mean precision dan recall | $\frac{2 \cdot P \cdot R}{P + R}$ |
| **Top-3 Accuracy** | Label benar di 3 prediksi teratas | - |
| **Top-5 Accuracy** | Label benar di 5 prediksi teratas | - |

### 3.5.2 Metrik Biometrik

| Metrik | Definisi |
|--------|----------|
| **FAR (False Acceptance Rate)** | Proporsi impostor yang diterima sebagai pengguna sah |
| **FRR (False Rejection Rate)** | Proporsi pengguna sah yang ditolak |
| **EER (Equal Error Rate)** | Titik di mana FAR = FRR. Semakin rendah, semakin baik |

### 3.5.3 Metrik Privasi

| Metrik | Definisi | Interpretasi |
|--------|----------|--------------|
| **Epsilon ($\epsilon$)** | Privacy budget kumulatif | $\epsilon < 1$: privasi kuat; $\epsilon = \infty$: tanpa privasi |
| **MIA AUC** | Area Under ROC Curve serangan MIA | ~0.50: aman; >0.60: rentan |
| **Cosine Similarity** | Kemiripan data rekonstruksi leakage | Mendekati 0: aman; mendekati 1: bocor |

### 3.5.4 Metrik Federated

| Metrik | Definisi |
|--------|----------|
| **Global Loss** | Loss model global pada data test setelah agregasi |
| **Global Accuracy** | Akurasi model global pada data test |
| **Per-round convergence** | Loss dan akurasi per communication round |

## 3.6 Alur Eksperimen

```
Dataset Mentah (DSL-StrongPasswordData.csv)
        ↓
    Preprocessing & Sequence Bundle
        ↓
┌───────────────────────────────────────────────┐
│           Level 1: Baseline LSTM              │
│     (Training terpusat tanpa proteksi)        │
└───────────────────┬───────────────────────────┘
                    ↓
┌───────────────────────────────────────────────┐
│          Level 2: Centralized DP              │
│        (DP-SGD via Opacus, ε=0.77)            │
└───────────────────┬───────────────────────────┘
                    ↓
┌───────────────────────────────────────────────┐
│        Level 3-4: FL & FL+DP                  │
│   (5 klien × 5 round, FedAvg via Flower)      │
└───────────────────┬───────────────────────────┘
                    ↓
┌───────────────────────────────────────────────┐
│       Level 5-6: Evaluasi Serangan            │
│      (MIA shadow classifier + Gradient        │
│       Leakage LBFGS reconstruction)           │
└───────────────────┬───────────────────────────┘
                    ↓
┌───────────────────────────────────────────────┐
│      Level 7-8: Non-IID & Ablation            │
│   (Subject skew + speed scaling; grid ε)      │
└───────────────────┬───────────────────────────┘
                    ↓
┌───────────────────────────────────────────────┐
│   Level 9-10: Leakage, Explainability,        │
│   Transfer FL, & Threat Modeling              │
└───────────────────┬───────────────────────────┘
                    ↓
            Dashboard & Laporan
```

## 3.7 Lingkungan Eksperimen

| Komponen | Spesifikasi |
|----------|-------------|
| **Sistem Operasi** | Windows |
| **Runtime** | Python 3.x |
| **Deep Learning** | PyTorch |
| **Differential Privacy** | Opacus |
| **Federated Learning** | Flower (gRPC) |
| **Data Processing** | NumPy, Pandas, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Notebook** | Jupyter Notebook |
| **Komputasi** | CPU-only (tanpa GPU) |
