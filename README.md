# Keystroke Dynamics Biometrics: Privacy-Preserving Behavioral Biometrics

Proyek ini mengimplementasikan sistem identifikasi dan autentikasi pengguna berbasis biometrik perilaku mengetik (Keystroke Dynamics) menggunakan arsitektur Deep Learning (LSTM) di PyTorch. Sistem ini dirancang dengan mengintegrasikan kerangka keamanan privasi tingkat lanjut, yaitu Federated Learning dan Differential Privacy, serta mengevaluasi ketahanannya terhadap berbagai skenario serangan privasi.

Proyek ini terstruktur ke dalam 4 tingkatan pengembangan (Multi-Level Framework) untuk menyajikan solusi keamanan informasi yang komprehensif, akademis, dan dapat direproduksi secara penuh.

---

## Panduan Membaca dan Menjelajahi Repositori

Untuk membantu Anda memahami dan menavigasi repositori ini dengan cepat, berikut adalah berkas utama yang wajib dibaca terlebih dahulu sesuai dengan tujuan Anda:

1. **Memahami Hasil Penelitian dan Analisis Teoretis**  
   Buka direktori [docs/laporan/](docs/laporan/) yang berisi 12 bab laporan modular tersegmentasi secara mendalam. Disarankan untuk memulainya dari **[Daftar Isi Laporan](docs/laporan/00_daftar_isi.md)** yang menyajikan tautan navigasi ke seluruh bab, mulai dari pendahuluan, metodologi detail, hasil eksperimen privasi, skenario serangan, hingga analisis komparatif komprehensif.

2. **Melakukan Reproduksi Eksperimen Lokal**  
   Buka berkas **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)**. Berkas ini menyediakan panduan langkah demi langkah yang sangat detail untuk menyiapkan virtual environment, mengunduh dataset publik CMU, dan menjalankan seluruh skrip otomatisasi training maupun evaluasi tanpa kendala teknis pada sistem Windows.

3. **Melihat Kode Implementasi dan Visualisasi Interaktif**  
   Jelajahi direktori **[notebooks/modular/](notebooks/modular/)** yang berisi Jupyter Notebook berurutan dari langkah audit dataset (01), pemrosesan data (02), pelatihan baseline LSTM (03), integrasi Differential Privacy (04), simulasi Federated Learning (05), evaluasi serangan privasi (06), analisis ablasi dan pemahaman model (07), hingga Advanced Federated Transfer Learning (10).

4. **Memeriksa Rekapitulasi Data Hasil Akhir**  
   Hasil komparatif terpadu dari seluruh model dapat langsung diperiksa pada folder **[outputs/reports/](outputs/reports/)**, khususnya berkas `final_summary_table.csv` dan `final_summary_bundle.json`. Visualisasi kurva dan diagram premium tersimpan di **[outputs/figures/](outputs/figures/)**.

---

## Kerangka Kerja 4 Tingkat (Multi-Level Framework)

### Level 1: Baseline Machine Learning Terpusat
* **Tujuan**: Membangun landasan identifikasi pengguna berdasarkan karakteristik pengetikan.
* **Fitur Biometrik**: Ekstraksi key press timing, key release timing, dwell time, flight time, typing speed, inter-key latency, dan typing rhythm dari Keystroke Dynamics Benchmark Dataset (CMU Lab).
* **Model**: Recurrent Neural Network berbasis LSTM (Long Short-Term Memory) untuk mempelajari pola temporal ketukan tombol secara berurutan.
* **Capaian**: Akurasi Baseline LSTM mencapai 63.29% dengan EER (Equal Error Rate) sebesar 16.29%.

### Level 2: Desentralisasi dan Perlindungan Privasi (Federated Learning & Differential Privacy)
* **Tujuan**: Melindungi data sensitif pengguna agar tidak dikirim ke server pusat selama proses pelatihan.
* **Metode**: 
  * **Differential Privacy (DP-SGD)** menggunakan framework Opacus untuk menambahkan noise terkontrol pada gradien model (epsilon = 0.77).
  * **Federated Learning (FL)** menggunakan framework Flower dengan algoritma FedAvg untuk melatih model secara kolaboratif di antara 5 client lokal tanpa membagikan data mentah.
* **Skenario Kombinasi**: Pelatihan gabungan FL dan DP lokal untuk memberikan perlindungan privasi berlapis.

### Level 3: Evaluasi Keamanan Privasi dan Threat Modeling
* **Tujuan**: Menguji ketahanan model biometrik terhadap serangan siber aktif.
* **Metode Serangan**:
  * **Membership Inference Attack (MIA)** menggunakan Shadow Classifier untuk menebak apakah data pengguna tertentu digunakan selama proses pelatihan model.
  * **Gradient Reconstruction Leakage Attack** menggunakan optimasi L-BFGS untuk merekonstruksi interval waktu pengetikan sensitif dari gradien model yang dikirim selama pelatihan.

### Level 4: Optimalisasi Tingkat Lanjut dan Explainability
* **Tujuan**: Mengatasi penurunan akurasi pada lingkungan terdistribusi Non-IID dan menjelaskan keputusan model.
* **Metode**:
  * **Advanced Federated Transfer Learning**: Penerapan Z-score localization pada sisi client untuk menyeimbangkan variasi pola mengetik lokal pada distribusi Non-IID. Metode ini berhasil mengembalikan akurasi model FL ke angka optimal 63.29% dan menekan EER hingga 16.29%, sembari menjaga tingkat privasi tertinggi (MIA AUC mendekati tebakan acak, yaitu 0.5003).
  * **Model Explainability (Integrated Gradients)**: Menggunakan metode atribusi fitur untuk memetakan tombol mana saja yang paling berpengaruh terhadap identitas mengetik unik seseorang.

---

## Struktur Direktori Utama

* `docs/laporan/`: Berkas laporan modular tersegmentasi (Bab 00 sampai Bab 11).
* `notebooks/modular/`: Jupyter Notebook berurutan untuk eksperimen dan visualisasi.
* `src/`: Kode sumber utama bersama (data loaders, preprocessing, LSTM model, visualizer, metrics).
* `scripts/`: Skrip otomasi terminal untuk menjalankan eksperimen berat (Federated Learning, serangan MIA, rekonstruksi gradien).
* `outputs/figures/`: Seluruh grafik hasil evaluasi berkualitas tinggi (Premium HSL color palette).
* `outputs/reports/`: Rekapitulasi metrik performa model dalam format JSON dan CSV.
* `data/`: Folder penyimpanan dataset (data mentah diletakkan di `data/raw/`).

---

## Memulai Cepat (Quick Start) lokal

1. **Jalankan inisialisasi lingkungan virtual (virtual environment) dan dependensi**:
   Buka PowerShell di direktori proyek ini dan jalankan skrip bootstrap otomatis:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned; .\bootstrap_venv.ps1
   ```

2. **Unduh Dataset**:
   Letakkan berkas dataset resmi `DSL-StrongPasswordData.csv` ke dalam folder `data/raw/`.

3. **Jalankan Seluruh Pipeline**:
   Untuk melatih seluruh model dan memproses metrik secara berurutan, jalankan:
   ```powershell
   python scripts/run_pipeline.py
   ```

---

## Lisensi dan Kontak
Proyek ini dibuat untuk keperluan akademis keamanan informasi. Jika Anda menemukan kendala saat instalasi atau ingin mendiskusikan hasil eksperimen, silakan hubungi tim pengembang melalui sistem Issue di repositori GitHub ini.
