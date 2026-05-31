# Project Plan: Keystroke Dynamics Behavioral Biometrics

## Outcome
Membangun sistem identifikasi dan autentikasi pengguna berbasis keystroke dynamics dengan LSTM, lalu mengevaluasi dampak Differential Privacy, Federated Learning, kombinasi FL+DP, dan serangan privasi terhadap utility model serta perlindungan behavioral biometrics.

Keberhasilan proyek terlihat jika seluruh level eksperimen dapat dijalankan secara konsisten, metrik utility dan privacy terdokumentasi dengan baik, hasil antar skenario dapat dibandingkan secara adil, dan laporan akhir, video penjelasan, serta notebook implementasi dapat diserahkan dalam bentuk yang rapi dan dapat direplikasi.

## Scope
### In Scope
- Akuisisi dan pemahaman dataset Keystroke Dynamics Benchmark.
- Preprocessing data keystroke end-to-end.
- Implementasi baseline LSTM untuk identifikasi atau autentikasi pengguna.
- Integrasi Differential Privacy menggunakan Opacus dan DP-SGD.
- Simulasi Federated Learning menggunakan Flower dengan FedAvg.
- Kombinasi FL + DP pada client-side training.
- Evaluasi performance, privacy, communication efficiency, dan convergence.
- Implementasi dan evaluasi Membership Inference Attack.
- Analisis leakage lanjutan, non-IID FL, ablation study, explainability, dan threat modeling.
- Penyusunan laporan .docx, video .mp4, dan notebook .ipynb.

### Out of Scope
- Deployment produksi ke layanan publik.
- Integrasi dengan sistem autentikasi nyata di perangkat pengguna.
- Pengumpulan data keystroke baru di luar dataset yang disediakan.
- Optimasi skala besar untuk infrastruktur terdistribusi produksi.

## Assumptions
- Dataset tersedia dan dapat diakses dalam format yang cukup untuk dieksplorasi secara programatik.
- Implementasi utama menggunakan Python, PyTorch, Opacus, dan Flower.
- Eksperimen dapat dijalankan pada satu environment lokal dengan pengaturan seed yang konsisten.
- Perbandingan antar skenario harus menggunakan pipeline preprocessing yang sama agar hasil adil.
- Jika data mentah memiliki struktur yang berbeda dari asumsi awal, pipeline akan disesuaikan tanpa mengubah tujuan utama eksperimen.
- Referensi struktur dataset dan implikasi teknis tersedia di [docs/plan/keystroke-dataset-notes.md](docs/plan/keystroke-dataset-notes.md).

## Work Breakdown
Rincian task implementasi per level tersedia di [docs/plan/keystroke-dynamics-tasks.md](docs/plan/keystroke-dynamics-tasks.md).

1. Identifikasi struktur dataset, skema label, sesi, dan fitur mentah.
2. Bangun pipeline preprocessing yang mencakup missing value handling, outlier filtering, session segmentation, sequence construction, normalization, windowing, padding/truncation, dan train-test split.
3. Implementasikan baseline LSTM untuk klasifikasi pengguna atau autentikasi.
4. Tambahkan evaluasi utility lengkap: accuracy, F1-score, FAR, FRR, EER, dan confusion matrix.
5. Integrasikan Differential Privacy dengan Opacus dan ukur epsilon privacy budget.
6. Simulasikan Federated Learning dengan Flower menggunakan FedAvg.
7. Gabungkan FL + DP pada client-side training.
8. Jalankan Membership Inference Attack dan analisis leakage pada baseline, DP, FL, dan FL+DP.
9. Simulasikan non-IID client distribution dan ukur dampaknya pada federated training.
10. Lakukan ablation study terhadap parameter utama model, DP, dan FL.
11. Tambahkan explainability dan threat modeling untuk menjelaskan fitur penting dan risiko keamanan.
12. Susun laporan, video penjelasan, dan notebook final.

## Implementation Notes
### Data Pipeline
- Gunakan pipeline terpusat agar seluruh skenario memakai preprocessing yang identik.
- Simpan transformasi penting seperti scaling, windowing policy, dan split strategy secara eksplisit.
- Hindari kebocoran data antar train, validation, dan test.

### Baseline Model
- Gunakan LSTM sebagai model utama untuk mempelajari dependency temporal.
- Siapkan output multiclass untuk user identification dan opsi biner untuk authentication jika diperlukan.
- Dokumentasikan konfigurasi sequence length, hidden units, batch size, optimizer, dan learning rate.

### Differential Privacy
- Terapkan DP-SGD dengan clipping per-sample gradient dan Gaussian noise.
- Catat epsilon, clipping norm, noise multiplier, dan target delta untuk setiap eksperimen.
- Bandingkan penurunan utility terhadap baseline.

### Federated Learning
- Gunakan simulasi client-server lokal dengan FedAvg.
- Catat jumlah client, local epochs, communication rounds, dan latency training.
- Evaluasi convergence global dan perbedaan kinerja local versus global model.

### FL + DP
- Terapkan DP pada local client sebelum agregasi.
- Evaluasi apakah perlindungan privasi meningkat tanpa menurunkan utility secara ekstrem.

### Privacy Attacks
- Fokus pada Membership Inference Attack sebagai evaluasi utama.
- Jika memungkinkan, tambahkan analisis konseptual atau empiris untuk fingerprinting, gradient leakage, dan model inversion.

### Explainability dan Threat Model
- Gunakan SHAP atau attention visualization untuk fitur penting seperti dwell time, flight time, typing speed, inter-key latency, dan typing rhythm.
- Petakan ancaman untuk honest-but-curious server, malicious client, external attacker, dan insider threat.

## Validation
### Functional Validation
- Pastikan preprocessing menghasilkan sequence yang valid dan konsisten.
- Pastikan baseline LSTM dapat dilatih dan menghasilkan prediksi pada test set.
- Pastikan DP, FL, dan FL+DP dapat berjalan tanpa error integrasi.

### Quantitative Validation
- Ukur accuracy, F1-score, FAR, FRR, EER, dan confusion matrix pada baseline dan varian privasi.
- Ukur epsilon privacy budget untuk eksperimen DP.
- Ukur communication efficiency, convergence, dan latency untuk eksperimen FL.
- Ukur MIA attack accuracy, precision, recall, dan ROC-AUC untuk evaluasi leakage.

### Comparative Validation
- Bandingkan baseline, DP, FL, dan FL+DP menggunakan konfigurasi yang setara.
- Bandingkan hasil pada skenario IID dan non-IID.
- Bandingkan hasil sebelum dan sesudah ablation untuk melihat sensitivitas parameter.

## Deliverables
- Notebook implementasi lengkap dalam format `.ipynb`.
- Laporan penelitian dalam format `.docx`.
- Video penjelasan dalam format `.mp4`.
- Tabel hasil eksperimen dan visualisasi utama.
- Ringkasan konfigurasi eksperimen untuk replikasi.
- Outline laporan tersedia di [docs/plan/keystroke-dynamics-report-outline.md](docs/plan/keystroke-dynamics-report-outline.md).

## Risks
- Struktur dataset dapat berbeda dari ekspektasi awal sehingga preprocessing perlu disesuaikan.
- DP dapat menurunkan utility secara signifikan jika noise terlalu besar atau clipping terlalu ketat.
- FL pada non-IID data dapat menyebabkan konvergensi lambat atau tidak stabil.
- Evaluasi serangan privasi dapat memerlukan penyesuaian jika data atau resource terbatas.
- Waktu komputasi dapat membesar karena banyaknya skenario eksperimen dan ablation.

## Status
Not started