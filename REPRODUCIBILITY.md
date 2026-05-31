# PANDUAN REPRODUKSI PIPELINE KEYSTROKE DYNAMICS
Dokumen ini disusun sebagai panduan langkah-demi-langkah bagi kontributor/anggota kelompok untuk mereproduksi seluruh metrik, model, dan grafik visualisasi pada proyek Keamanan Informasi ini secara konsisten dan bebas dari kendala teknis (seperti *Jupyter freeze* atau *port lock*).

---

## 📌 1. Persiapan Lingkungan & Dependensi

Proyek ini dikembangkan menggunakan **Python 3.14.0** (atau Python 3.10+ ke atas) pada sistem operasi Windows.

### Langkah A: Setup Virtual Environment (`.venv`)
Buka PowerShell/Terminal di direktori root proyek ini, kemudian jalankan:
```powershell
# 1. Membuat virtual environment baru
python -m venv .venv

# 2. Mengaktifkan virtual environment
.venv\Scripts\Activate.ps1

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Menginstal seluruh dependensi pustaka
pip install -r requirements.txt
```
*(Atau Anda bisa langsung menjalankan skrip otomatisasi yang telah disediakan: `powershell -ExecutionPolicy Bypass -File bootstrap_venv.ps1`)*

### Langkah B: Penyiapan Data Mentah (Raw Dataset)
1. Unduh dataset resmi **Keystroke Dynamics Benchmark** dari Kaggle: [DSL-StrongPasswordData.csv](https://www.kaggle.com/datasets/carnegiecylab/keystroke-dynamics-benchmark-data-set).
2. Buat folder bernama `data/raw/` di direktori proyek Anda (jika belum ada).
3. Letakkan berkas hasil unduhan dengan nama presisi: `data/raw/DSL-StrongPasswordData.csv`.

---

## 🛠️ 2. Langkah Reproduksi Pipeline (Step-by-Step)

Ada **dua cara** untuk mereproduksi keluaran proyek ini. Cara terbaik dan paling stabil untuk menghindari kelambatan/hang pada editor Jupyter adalah menggunakan **Terminal VS Code** untuk bagian training berat/simulasi jaringan, kemudian menggunakan **Jupyter Notebook** untuk memvisualisasikan hasilnya.

### 🚀 OPSI A: Eksekusi Cepat via Terminal (Sangat Direkomendasikan & Stabil)
Jalankan perintah-perintah berikut di Terminal VS Code Anda secara berurutan. Metode ini menjamin kestabilan proses 100% dan bebas dari penyumbatan buffer konsol editor:

```powershell
# Langkah 1: Audit Dataset awal
.venv\Scripts\python.exe -c "import nbformat; from nbclient import NotebookClient; nb = nbformat.read('notebooks/modular/01_audit_dataset.ipynb', as_version=4); NotebookClient(nb).execute(cwd='.'); nbformat.write(nb, 'notebooks/modular/01_audit_dataset.ipynb'); print('01 AUDIT SUKSES')"

# Langkah 2: Preprocessing Data (Menghasilkan sequence_bundle.npz)
.venv\Scripts\python.exe -c "import nbformat; from nbclient import NotebookClient; nb = nbformat.read('notebooks/modular/02_preprocessing.ipynb', as_version=4); NotebookClient(nb).execute(cwd='.'); nbformat.write(nb, 'notebooks/modular/02_preprocessing.ipynb'); print('02 PREPROCESSING SUKSES')"

# Langkah 3: Melatih Baseline LSTM terpusat (Akurasi ~54%)
.venv\Scripts\python.exe -c "import nbformat; from nbclient import NotebookClient; nb = nbformat.read('notebooks/modular/03_baseline_lstm.ipynb', as_version=4); NotebookClient(nb).execute(cwd='.'); nbformat.write(nb, 'notebooks/modular/03_baseline_lstm.ipynb'); print('03 BASELINE SUKSES')"

# Langkah 4: Melatih Centralized DP LSTM (Opacus - epsilon = 0.77)
.venv\Scripts\python.exe -c "import nbformat; from nbclient import NotebookClient; nb = nbformat.read('notebooks/modular/04_differential_privacy.ipynb', as_version=4); NotebookClient(nb).execute(cwd='.'); nbformat.write(nb, 'notebooks/modular/04_differential_privacy.ipynb'); print('04 DP SUKSES')"

# Langkah 5: Menjalankan Simulasi Federated Learning (Flower gRPC - Port 8089)
.venv\Scripts\python.exe scripts/run_fl_manual.py

# Langkah 6: Menjalankan Simulasi FL + DP (Flower & Opacus - Port 8090)
.venv\Scripts\python.exe scripts/run_fl_dp_manual.py

# Langkah 7: Menjalankan Simulasi FL Non-IID (Port 8091)
.venv\Scripts\python.exe scripts/run_fl_non_iid.py

# Langkah 8: Menjalankan Evaluasi Serangan MIA (Shadow Classifier)
.venv\Scripts\python.exe -c "import nbformat; from nbclient import NotebookClient; nb = nbformat.read('notebooks/modular/06_privacy_attack_eval.ipynb', as_version=4); NotebookClient(nb).execute(cwd='.'); nbformat.write(nb, 'notebooks/modular/06_privacy_attack_eval.ipynb'); print('06 MIA ATTACK EVAL SUKSES')"

# Langkah 9: Menjalankan Ablation Study (Utility vs Privacy Curves)
.venv\Scripts\python.exe scripts/run_ablation.py

# Langkah 10: Menjalankan Serangan Rekonstruksi Gradien (LBFGS - Leakage)
.venv\Scripts\python.exe scripts/run_leakage_attack.py

# Langkah 11: Menjalankan Integrated Gradients (Explainability Attributions)
.venv\Scripts\python.exe scripts/run_explainability.py

# Langkah 12: Menjalankan Advanced Federated Transfer Learning (Akurasi FL melompat >65% - Port 8092)
.venv\Scripts\python.exe scripts/run_fl_transfer.py

# Langkah 13: Compile Seluruh Notebook Hasil & Dashboard Laporan Akhir
.venv\Scripts\python.exe -c "import nbformat; from nbclient import NotebookClient; nb = nbformat.read('notebooks/modular/07_ablation_explainability.ipynb', as_version=4); NotebookClient(nb).execute(cwd='.'); nbformat.write(nb, 'notebooks/modular/07_ablation_explainability.ipynb')"
.venv\Scripts\python.exe -c "import nbformat; from nbclient import NotebookClient; nb = nbformat.read('notebooks/modular/10_advanced_transfer_fl.ipynb', as_version=4); NotebookClient(nb).execute(cwd='.'); nbformat.write(nb, 'notebooks/modular/10_advanced_transfer_fl.ipynb')"
.venv\Scripts\python.exe -c "import nbformat; from nbclient import NotebookClient; nb = nbformat.read('notebooks/modular/99_report_artifacts.ipynb', as_version=4); NotebookClient(nb).execute(cwd='.'); nbformat.write(nb, 'notebooks/modular/99_report_artifacts.ipynb'); print('ALL NOTEBOOKS COMPILED')"
```

---

### 📓 OPSI B: Eksekusi Berurutan via Editor Jupyter Notebook (UI)
Jika Anda lebih menyukai eksekusi visual satu per satu menggunakan sel Jupyter di VS Code, silakan ikuti alur wajib di bawah ini:
1. Buka dan jalankan **`01_audit_dataset.ipynb`**, **`02_preprocessing.ipynb`**, **`03_baseline_lstm.ipynb`**, dan **`04_differential_privacy.ipynb`** secara berurutan.
2. **PENTING:** Saat Anda tiba pada **`05_federated_learning.ipynb`**, jangan jalankan sel training di dalam notebook jika komputer terasa lambat. Sebaiknya buka terminal dan jalankan `.venv\Scripts\python.exe scripts/run_fl_manual.py` dan `run_fl_dp_manual.py` terlebih dahulu. Setelah selesai, buka kembali notebook untuk memuat metrik instan yang sudah dihasilkan.
3. Jalankan **`06_privacy_attack_eval.ipynb`** untuk mengevaluasi kerentanan MIA.
4. Jalankan **`07_ablation_explainability.ipynb`** dan **`10_advanced_transfer_fl.ipynb`** untuk melihat visualisasi dan perbandingan konvergensi.
5. Jalankan **`99_report_artifacts.ipynb`** untuk menampilkan dashboard laporan komparatif final.

---

## 🔍 3. Panduan Pemecahan Masalah (Troubleshooting)

> [!WARNING]  
> **Masalah Umum 1: Jupyter Notebook macet selamanya (`[*]`) saat mengimpor pustaka (seperti `from data import ...`)**
> * **Penyebab:** Terlalu banyak kernel Python (`ipykernel_launcher`) yang menggantung aktif di latar belakang sistem Windows Anda, memicu *DLL/file lock deadlock* pada berkas PyTorch/Pandas.
> * **Solusi Cepat:** Buka PowerShell baru di VS Code, matikan seluruh proses Python secara paksa:
>   ```powershell
>   Stop-Process -Name python -Force
>   ```
>   Setelah itu, buka kembali notebook Anda di VS Code dan klik **"Restart Kernel"** di pojok kanan atas editor.

> [!WARNING]  
> **Masalah Umum 2: Simulasi FL (Flower) menggantung saat Server dijalankan**
> * **Penyebab:** 
>   1. Port gRPC simulasi (`8089`, `8090`, `8091`, atau `8092`) masih terkunci oleh sisa proses simulasi sebelumnya yang dihentikan paksa tanpa *clean-up*.
>   2. Windows Defender Firewall menahan koneksi soket jaringan `python.exe` lokal di latar belakang.
> * **Solusi Cepat:**
>   1. Periksa apakah port sedang terkunci:
>      ```powershell
>      Get-NetTCPConnection | Where-Object {$_.LocalPort -in @(8089, 8090, 8091, 8092)}
>      ```
>   2. Bersihkan total seluruh proses Python yang menggantung dengan perintah `Stop-Process -Name python -Force` untuk melepaskan port yang terkunci tersebut.
>   3. Izinkan hak akses jaringan (*Allow Access*) jika Windows Firewall menampilkan jendela peringatan keamanan saat simulasi Flower pertama kali dimulai.

---

## 📊 4. Daftar Luaran Hasil yang Diharapkan (Expected Outputs)

Setelah seluruh langkah di atas diselesaikan dengan sukses, pastikan berkas-berkas berikut telah terbentuk di dalam proyek Anda untuk verifikasi:

### A. Model Latih Checkpoints (`outputs/models/`)
* **`baseline_lstm.pt`**: Checkpoint model LSTM baseline terpusat.
* **`dp_lstm.pt`**: Checkpoint model LSTM terpusat dengan DP-SGD.
* **`fl_lstm.pt`**: Checkpoint model global hasil Federated Learning IID.
* **`fl_dp_lstm.pt`**: Checkpoint model global hasil Federated Learning + DP lokal.
* **`fl_non_iid_lstm.pt`**: Checkpoint model global FL Non-IID.
* **`fl_transfer_lstm.pt`**: Checkpoint model global FL dengan Transfer Learning skala Z-score (Akurasi melompat **>65%**!).

### B. Grafik Visualisasi Premium (`outputs/figures/`)
* **`keystroke_feature_importance.png`**: Heatmap kontribusi tombol hasil Integrated Gradients.
* **`gradient_reconstruction_leakage.png`**: Hasil perbandingan visual serangan rekonstruksi gradien pengetikan.
* **`privacy_utility_tradeoff.png`**: Kurva trade-off akurasi & EER vs Epsilon.
* **`mia_vs_epsilon.png`**: Grafik tingkat kerentanan privasi MIA.
* **`fl_transfer_comparison.png`**: Kurva perbandingan lonjakan konvergensi FL Baseline vs Transfer FL.

### C. Ringkasan Metrik Terpadu (`outputs/reports/`)
* **`final_summary_table.csv`** & **`final_summary_bundle.json`**: Gabungan seluruh metrik komparatif akhir proyek Anda.
