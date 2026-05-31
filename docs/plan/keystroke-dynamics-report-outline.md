# Outline Laporan Penelitian Keystroke Dynamics

Dokumen ini menjadi kerangka isi laporan `.docx` yang disusun berdasarkan plan dan task implementasi proyek.

## Bagian Awal
### Halaman Judul
- Judul penelitian.
- Nama kelompok.
- Nama mata kuliah.
- Nama dosen.
- Institusi.
- Tahun.

### Lembar Pengesahan
- Identitas penulis.
- Tanggal pengesahan.
- Tanda tangan pihak terkait.

### Abstrak
- Latar belakang singkat.
- Tujuan penelitian.
- Metode utama.
- Ringkasan hasil utama.
- Kesimpulan inti tentang utility dan privacy.

### Kata Pengantar
- Ucapan singkat dan formal.

### Daftar Isi
- Otomatis dari heading dokumen.

### Daftar Gambar dan Daftar Tabel
- Daftar visual dan tabel yang digunakan dalam laporan.

## Bab 1 - Pendahuluan
### 1.1 Latar Belakang
- Behavioral biometrics sebagai pendekatan identifikasi pengguna.
- Alasan keystroke dynamics relevan untuk keamanan informasi.
- Risiko privasi pada data perilaku pengetikan.
- Kebutuhan perlindungan melalui Differential Privacy dan Federated Learning.

### 1.2 Rumusan Masalah
- Bagaimana membangun identifikasi pengguna berbasis keystroke dynamics dengan LSTM.
- Bagaimana Differential Privacy memengaruhi utility model.
- Bagaimana Federated Learning memengaruhi privasi dan efisiensi komunikasi.
- Bagaimana kombinasi FL+DP memengaruhi performa dan keamanan.
- Seberapa besar risiko serangan privasi pada masing-masing skenario.

### 1.3 Tujuan Penelitian
- Membangun model baseline berbasis LSTM.
- Mengevaluasi DP, FL, dan FL+DP.
- Mengukur risiko kebocoran dan serangan privasi.
- Menilai privacy-utility trade-off.

### 1.4 Batasan Masalah
- Menggunakan dataset Keystroke Dynamics Benchmark.
- Fokus pada eksperimen terkontrol, bukan deployment produksi.
- Fokus pada skenario baseline, DP, FL, FL+DP, dan attack evaluation.

### 1.5 Manfaat Penelitian
- Menunjukkan potensi keystroke dynamics sebagai behavioral biometrics.
- Memberikan gambaran praktis trade-off privasi dan performa.
- Memberikan dasar analisis keamanan untuk sistem identifikasi pengguna.

### 1.6 Sistematika Penulisan
- Ringkasan isi tiap bab laporan.

## Bab 2 - Tinjauan Pustaka
### 2.1 Keystroke Dynamics
- Konsep behavioral biometrics.
- Fitur key press timing, key release timing, dwell time, flight time, typing speed, inter-key latency, dan typing rhythm.
- Tantangan pengolahan data pengetikan.

### 2.2 User Identification dan Authentication
- Perbedaan identifikasi multi-class dan autentikasi biner.
- Kesesuaian keystroke dynamics untuk kedua skenario.

### 2.3 LSTM untuk Data Temporal
- Dependency temporal.
- Kesesuaian LSTM untuk sequence keystroke.

### 2.4 Differential Privacy
- Konsep epsilon dan delta.
- DP-SGD.
- Gradient clipping dan Gaussian noise.

### 2.5 Federated Learning
- Konsep client-server training.
- FedAvg.
- Komunikasi model tanpa memindahkan data mentah.

### 2.6 Privacy Attacks
- Membership Inference Attack.
- Keystroke fingerprinting.
- User deanonymization.
- Gradient leakage.
- Model inversion.

### 2.7 Explainability dan Threat Modeling
- SHAP atau attention visualization.
- Threat model pada honest-but-curious server, malicious client, external attacker, dan insider threat.

## Bab 3 - Metodologi Penelitian
### 3.1 Desain Penelitian
- Alur eksperimen bertahap dari baseline hingga evaluasi privasi.
- Skema pembandingan antar metode.

### 3.2 Dataset
- Sumber dataset.
- Struktur data.
- Jumlah user dan sesi bila tersedia.
- Karakteristik fitur mentah.

### 3.3 Preprocessing Data
- Missing value handling.
- Outlier filtering.
- Session segmentation.
- Temporal sequence construction.
- Normalization.
- Sequence windowing.
- Padding/truncation.
- Train-test split.

### 3.4 Perancangan Fitur
- Pembentukan fitur turunan dari perilaku pengetikan.
- Representasi sequence multivariat.

### 3.5 Arsitektur Baseline LSTM
- Input layer.
- LSTM layer.
- Fully connected classification layer.
- Output multiclass atau binary.

### 3.6 Differential Privacy Implementation
- Integrasi Opacus.
- Clipping norm.
- Noise multiplier.
- Privacy accountant.

### 3.7 Federated Learning Implementation
- Pembagian data ke client.
- Client-local training.
- FedAvg aggregation.
- Communication rounds.

### 3.8 Federated Learning + Differential Privacy
- Penerapan DP pada local client.
- Agregasi parameter dengan FedAvg.

### 3.9 Privacy Attack Evaluation
- Setup Membership Inference Attack.
- Evaluasi leakage pada baseline, DP, FL, dan FL+DP.

### 3.10 Non-IID Simulation
- Strategi pembagian client heterogen.
- Sumber heterogenitas: typing speed, keyboard layout, device type, habits, session duration.

### 3.11 Ablation Study
- Variasi epsilon.
- Noise multiplier.
- Clipping norm.
- Sequence length.
- Hidden units.
- Jumlah client.
- Local epoch.
- Communication rounds.
- Learning rate.

### 3.12 Explainability dan Threat Model
- Interpretasi fitur penting.
- Analisis risiko per skenario.

## Bab 4 - Hasil dan Pembahasan
### 4.1 Hasil Preprocessing
- Statistik data sebelum dan sesudah preprocessing.
- Contoh sequence.
- Visualisasi distribusi fitur.

### 4.2 Hasil Baseline LSTM
- Accuracy.
- F1-score.
- FAR.
- FRR.
- EER.
- Confusion matrix.
- Analisis behavioral biometrics.

### 4.3 Hasil Differential Privacy
- Kinerja model DP.
- Nilai epsilon.
- Perubahan convergence dan stability.
- Trade-off utility versus privacy.

### 4.4 Hasil Federated Learning
- Local versus global performance.
- Communication efficiency.
- Convergence stability.
- Training latency.

### 4.5 Hasil FL + DP
- Perbandingan terhadap baseline, DP, dan FL.
- Utility loss dan privacy gain.

### 4.6 Hasil Privacy Attack Evaluation
- MIA accuracy, precision, recall, ROC-AUC.
- Perbandingan kebocoran antar skenario.
- Analisis keystroke fingerprinting dan leakage.

### 4.7 Hasil Non-IID FL
- Dampak heterogenitas client.
- Client drift.
- Fairness antar client.

### 4.8 Hasil Ablation Study
- Grafik accuracy versus epsilon.
- Grafik EER versus epsilon.
- Grafik FAR versus epsilon.
- Grafik FRR versus epsilon.
- Grafik communication efficiency versus performance.
- Grafik MIA attack accuracy versus epsilon.

### 4.9 Hasil Explainability dan Threat Model
- Fitur yang paling memengaruhi prediksi.
- Risiko serangan per skenario.
- Perbandingan tingkat keamanan baseline, DP, FL, dan FL+DP.

## Bab 5 - Kesimpulan dan Saran
### 5.1 Kesimpulan
- Ringkasan temuan utama.
- Jawaban terhadap rumusan masalah.
- Skema yang paling baik dari sisi utility dan privacy.

### 5.2 Keterbatasan Penelitian
- Keterbatasan dataset.
- Keterbatasan komputasi.
- Keterbatasan serangan privasi yang diuji.

### 5.3 Saran Pengembangan
- Penggunaan dataset yang lebih besar.
- Pengujian model lain.
- Penguatan serangan dan defense lanjutan.
- Implementasi sistem autentikasi yang lebih mendekati dunia nyata.

## Bagian Akhir
### Daftar Pustaka
- Semua sumber yang digunakan.

### Lampiran
- Konfigurasi eksperimen.
- Tabel hasil lengkap.
- Cuplikan kode penting.
- Tambahan visualisasi.

## Pemetaan Singkat ke Tahapan Proyek
- Bab 1 dan Bab 2 mendukung perumusan masalah dan landasan teori.
- Bab 3 mendokumentasikan Level 1 sampai Level 10 secara metodologis.
- Bab 4 menyajikan hasil eksperimen, analisis privasi, dan evaluasi serangan.
- Bab 5 merangkum kesimpulan, keterbatasan, dan arah lanjutan.

## Draft Lanjutan
- Draft isi yang lebih dekat ke teks final tersedia di [docs/plan/keystroke-dynamics-report-draft.md](docs/plan/keystroke-dynamics-report-draft.md).