# Lampiran

## A. Daftar Artefak Proyek

### A.1 Model Checkpoint (`outputs/models/`)

| File | Ukuran | Konfigurasi |
|------|:------:|-------------|
| `baseline_lstm.pt` | 224 KB | Centralized Baseline LSTM |
| `dp_lstm.pt` | 225 KB | Centralized DP LSTM (Opacus, ε=0.77) |
| `fl_lstm.pt` | 224 KB | Federated Baseline (5 round FedAvg) |
| `fl_dp_lstm.pt` | 225 KB | Joint FL + DP (FedAvg + DP-SGD) |
| `fl_non_iid_lstm.pt` | 224 KB | Non-IID Federated (subject skew + speed) |
| `fl_transfer_lstm.pt` | 224 KB | Advanced Transfer FL (pre-trained + scaled) |

### A.2 Laporan Metrik (`outputs/reports/`)

| File | Format | Konten |
|------|:------:|--------|
| `preprocessing_summary.json` | JSON | Shape data dan jumlah sampel pasca-preprocessing |
| `audit_summary.json` | JSON | Audit dataset mentah: kolom, distribusi, validasi DD=H+UD |
| `baseline_metrics.json` | JSON | Metrik evaluasi baseline (loss, accuracy, confusion matrix) |
| `baseline_metrics_final.json` | JSON | Metrik evaluasi akhir semua konfigurasi baseline |
| `dp_metrics.json` | JSON | Metrik DP (epsilon, delta, accuracy, confusion matrix) |
| `fl_metrics.json` | JSON | Metrik FL per-round (loss, accuracy, konvergensi) |
| `fl_dp_metrics.json` | JSON | Metrik FL+DP per-round |
| `fl_non_iid_metrics.json` | JSON | Metrik Non-IID FL (client sizes, speed scalers) |
| `fl_transfer_metrics.json` | JSON | Metrik Transfer FL per-round |
| `attack_metrics.json` | JSON | Metrik MIA (AUC, precision, recall) per konfigurasi |
| `leakage_metrics.json` | JSON | Metrik Gradient Leakage (MSE, cosine similarity) |
| `ablation_explainability.json` | JSON | Grid search epsilon dan best trade-off |
| `explainability_summary.json` | JSON | Integrated Gradients attributions per karakter |
| `final_summary_table.csv` | CSV | Dashboard ringkasan seluruh konfigurasi |
| `final_summary_bundle.json` | JSON | Bundle lengkap seluruh metrik proyek |

### A.3 Visualisasi (`outputs/figures/`)

| File | Konten |
|------|--------|
| `lstm_loss_curves_improved.png` | Kurva loss pelatihan baseline vs DP |
| `confusion_matrices_comparison.png` | Perbandingan confusion matrix konfigurasi awal |
| `confusion_matrices_final_comparison.png` | Perbandingan confusion matrix semua konfigurasi |
| `privacy_utility_tradeoff.png` | Grafik trade-off Accuracy vs Epsilon |
| `mia_vs_epsilon.png` | Grafik MIA AUC vs Epsilon |
| `gradient_reconstruction_leakage.png` | Visualisasi rekonstruksi gradient (standard vs DP) |
| `keystroke_feature_importance.png` | Heatmap atribusi Integrated Gradients |
| `fl_transfer_comparison.png` | Perbandingan konvergensi FL standar vs Transfer FL |

### A.4 Notebook Modular (`notebooks/modular/`)

| Notebook | Level | Konten |
|----------|:-----:|--------|
| `01_audit_dataset.ipynb` | Prep | Audit dataset mentah |
| `02_preprocessing.ipynb` | Prep | Pipeline preprocessing → `sequence_bundle.npz` |
| `03_baseline_lstm.ipynb` | 1 | Pelatihan dan evaluasi baseline LSTM |
| `04_differential_privacy.ipynb` | 2 | Pelatihan DP-SGD dan analisis epsilon |
| `05_federated_learning.ipynb` | 3-4 | Analisis hasil FL dan FL+DP |
| `06_privacy_attack_eval.ipynb` | 5-6 | Evaluasi MIA dan analisis kerentanan |
| `07_ablation_explainability.ipynb` | 8 | Studi ablasi dan atribusi fitur |
| `08_gradient_leakage_eval.ipynb` | 9 | Evaluasi Gradient Leakage Attack |
| `10_advanced_transfer_fl.ipynb` | 10 | Analisis Advanced Transfer FL |
| `99_report_artifacts.ipynb` | - | Dashboard kompilasi dan threat model |

---

## B. Konfigurasi Teknis

### B.1 Dependencies Utama

```
torch
opacus
flwr (flower)
numpy
pandas
scikit-learn
matplotlib
seaborn
nbclient
jupyter
```

### B.2 Seed dan Reproducibility

| Parameter | Nilai |
|-----------|:-----:|
| `random_state` (train-test split) | 42 |
| `torch.manual_seed` | 42 (di setiap script) |
| `numpy.random.seed` | 42 |

### B.3 Arsitektur Model (Detail)

```python
KeystrokeLSTM(
  (lstm): LSTM(3, 64, num_layers=2, batch_first=True, dropout=0.3)
  (fc): Sequential(
    (0): Linear(in_features=64, out_features=32, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.3, inplace=False)
    (3): Linear(in_features=32, out_features=51, bias=True)
  )
)
```

---

## C. Glosarium

| Istilah | Definisi |
|---------|----------|
| **AUC** | Area Under the ROC Curve - metrik evaluasi classifier |
| **Clipping Norm** | Batas maksimum norma L2 gradient per sampel dalam DP-SGD |
| **Client Drift** | Divergensi model lokal dalam FL akibat data heterogen |
| **Cosine Similarity** | Ukuran kemiripan arah dua vektor (−1 hingga +1) |
| **Delta (δ)** | Parameter probabilitas kegagalan dalam Differential Privacy |
| **DP-SGD** | Differentially Private Stochastic Gradient Descent |
| **Dwell Time** | Durasi penekanan satu tombol |
| **EER** | Equal Error Rate - titik di mana FAR = FRR |
| **Epsilon (ε)** | Privacy budget dalam Differential Privacy |
| **FAR** | False Acceptance Rate - proporsi impostor yang diterima |
| **FedAvg** | Federated Averaging - algoritma agregasi FL |
| **Flight Time** | Jeda antar penekanan dua tombol berurutan |
| **FRR** | False Rejection Rate - proporsi pengguna sah yang ditolak |
| **gRPC** | Google Remote Procedure Call - protokol komunikasi |
| **Integrated Gradients** | Metode atribusi fitur berbasis integral gradien |
| **LBFGS** | Limited-memory BFGS - algoritma optimisasi quasi-Newton |
| **LSTM** | Long Short-Term Memory - arsitektur RNN |
| **MIA** | Membership Inference Attack |
| **Noise Multiplier (σ)** | Rasio noise Gaussian terhadap clipping norm dalam DP |
| **Non-IID** | Non Independent and Identically Distributed |
| **Opacus** | Library DP untuk PyTorch oleh Meta |
| **Pareto Frontier** | Kumpulan solusi optimal yang tidak didominasi |
| **RDP** | Rényi Differential Privacy |
| **StandardScaler** | Z-score normalization (mean=0, std=1) |
| **Transfer Learning** | Inisialisasi model dengan bobot pre-trained |

---

## D. Referensi

1. Killourhy, K. S., & Maxion, R. A. (2009). *Comparing anomaly-detection algorithms for keystroke dynamics.* IEEE/IFIP International Conference on Dependable Systems & Networks.

2. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735-1780.

3. Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). *Calibrating noise to sensitivity in private data analysis.* Theory of Cryptography Conference.

4. Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). *Deep learning with differential privacy.* ACM Conference on Computer and Communications Security.

5. McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). *Communication-efficient learning of deep networks from decentralized data.* AISTATS.

6. Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). *Membership inference attacks against machine learning models.* IEEE Symposium on Security and Privacy.

7. Zhu, L., Liu, Z., & Han, S. (2019). *Deep leakage from gradients.* NeurIPS.

8. Sundararajan, M., Taly, A., & Yan, Q. (2017). *Axiomatic attribution for deep networks.* ICML.

9. Beutel, D. J., Tober, T., McNamara, D., et al. (2020). *Flower: A friendly federated learning research framework.* arXiv preprint arXiv:2007.14390.

10. Yousefpour, A., Shilov, I., Sablayrolles, A., et al. (2021). *Opacus: User-friendly differential privacy library in PyTorch.* arXiv preprint arXiv:2109.12298.

11. Gaines, R. S., Lisowski, W., Press, S. J., & Shapiro, N. (1980). *Authentication by keystroke timing: Some preliminary results.* Rand Report R-2526-NSF.

---

## E. Panduan Reproduksi Singkat

Untuk panduan reproduksi lengkap, lihat `REPRODUCIBILITY.md` di root proyek.

### E.1 Quickstart

```powershell
# 1. Clone repository
git clone <repo-url>
cd "Projek Keamanan Informasi"

# 2. Setup virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Jalankan notebook secara berurutan
# Buka di VS Code / Jupyter, Run All satu per satu:
#   01 → 02 → 03 → 04 → (jalankan script FL via terminal) → 05 → 06 → 07 → 08 → 10 → 99
```

### E.2 Menjalankan Simulasi FL

```powershell
# Kill proses Python yang menggantung (jika ada)
Stop-Process -Name python -Force -ErrorAction SilentlyContinue

# FL Baseline
python scripts/run_fl_manual.py

# FL + DP
python scripts/run_fl_dp_manual.py

# Non-IID FL
python scripts/run_fl_non_iid.py

# Transfer FL
python scripts/run_fl_transfer.py
```

### E.3 Tips Troubleshooting

| Masalah | Solusi |
|---------|--------|
| Notebook cell hang / tidak selesai | `Stop-Process -Name python -Force` kemudian restart kernel |
| `ModuleNotFoundError` | Pastikan `.venv` aktif dan `pip install -r requirements.txt` |
| Port gRPC conflict | Kill semua proses Python sebelum menjalankan script FL |
| `sequence_bundle.npz` tidak ditemukan | Jalankan `02_preprocessing.ipynb` terlebih dahulu |
