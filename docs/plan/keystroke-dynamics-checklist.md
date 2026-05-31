# Keystroke Dynamics Granular Checklist

Dokumen ini memecah setiap level menjadi checklist implementasi yang lebih granular dan siap dipakai saat eksekusi bertahap.

## Cross-Cutting Checklist
- [ ] Tetapkan outcome final dan success criteria untuk seluruh proyek.
- [ ] Tetapkan metrik utama yang wajib dilaporkan di semua skenario.
- [ ] Siapkan struktur folder untuk data, notebook, source code, figure, dan report.
- [ ] Siapkan environment Python yang memuat PyTorch, Opacus, Flower, pandas, numpy, scikit-learn, matplotlib, dan seaborn.
- [ ] Tetapkan seed, logging, dan format output eksperimen.
- [ ] Standarkan penamaan eksperimen, checkpoint, dan tabel hasil.
- [ ] Siapkan template visualisasi untuk perbandingan antar skenario.

## Level 1 - Baseline Machine Learning
### Dataset Audit dan Preprocessing
- [ ] Identifikasi format file dan struktur kolom dataset.
- [ ] Petakan kolom user, session, key press, dan key release.
- [ ] Catat missing value, duplikasi, dan anomali format.
- [ ] Definisikan fitur dwell time, flight time, typing speed, inter-key latency, dan typing rhythm.
- [ ] Tentukan representasi sequence multivariat untuk model.
- [ ] Implementasikan missing value handling.
- [ ] Implementasikan outlier filtering.
- [ ] Implementasikan session segmentation.
- [ ] Implementasikan temporal sequence construction.
- [ ] Implementasikan normalization.
- [ ] Implementasikan sequence windowing.
- [ ] Implementasikan padding dan truncation.
- [ ] Implementasikan train-validation-test split yang bebas leakage.

### Baseline Model
- [ ] Bentuk dataset train, validation, dan test.
- [ ] Verifikasi panjang sequence dan dimensi fitur.
- [ ] Pastikan label distribusi masuk akal.
- [ ] Bangun model LSTM baseline.
- [ ] Tambahkan layer klasifikasi akhir.
- [ ] Siapkan mode multiclass identification.
- [ ] Siapkan mode binary authentication bila diperlukan.
- [ ] Jalankan training loop baseline.
- [ ] Simpan checkpoint model terbaik.
- [ ] Catat loss curve dan akurasi per epoch.

### Evaluasi Baseline
- [ ] Hitung accuracy.
- [ ] Hitung F1-score.
- [ ] Hitung FAR.
- [ ] Hitung FRR.
- [ ] Hitung EER.
- [ ] Buat confusion matrix.
- [ ] Tulis ringkasan pola keystroke yang paling membedakan pengguna.

## Level 2 - Differential Privacy
### Integrasi DP-SGD
- [ ] Integrasikan Opacus ke pipeline baseline.
- [ ] Pastikan model, optimizer, dan dataloader kompatibel dengan DP-SGD.
- [ ] Tetapkan clipping norm.
- [ ] Tetapkan noise multiplier.
- [ ] Tetapkan target delta.
- [ ] Aktifkan privacy accountant.
- [ ] Simpan epsilon untuk setiap run atau epoch.

### Training dan Evaluasi DP
- [ ] Jalankan training DP dengan konfigurasi yang setara dengan baseline.
- [ ] Catat convergence loss dan training stability.
- [ ] Simpan checkpoint model privat.
- [ ] Hitung accuracy, F1-score, FAR, FRR, dan EER.
- [ ] Bandingkan hasil dengan baseline.
- [ ] Dokumentasikan penurunan utility akibat noise.
- [ ] Tulis analisis trade-off privacy versus utility.

## Level 3 - Federated Learning
### Client Partitioning
- [ ] Tentukan strategi pembagian client.
- [ ] Distribusikan data secara IID sebagai baseline FL.
- [ ] Pastikan tidak ada leakage antar client.

### Federated Training
- [ ] Bangun fungsi training lokal per client.
- [ ] Pastikan client mengirim parameter model, bukan data mentah.
- [ ] Implementasikan FedAvg untuk agregasi bobot.
- [ ] Tentukan jumlah client.
- [ ] Tentukan local epochs.
- [ ] Tentukan communication rounds.
- [ ] Jalankan simulasi FL end-to-end.

### Evaluasi FL
- [ ] Hitung global accuracy.
- [ ] Hitung local versus global performance.
- [ ] Hitung convergence stability.
- [ ] Hitung communication efficiency.
- [ ] Hitung training latency.
- [ ] Catat model global per communication round.
- [ ] Tulis analisis privasi FL dan risiko leakage parameter.

## Level 4 - Federated Learning + Differential Privacy
### DP pada Client
- [ ] Terapkan DP-SGD pada setiap client training loop.
- [ ] Verifikasi clipping dan noise berjalan per client.
- [ ] Pastikan privacy budget dapat dicatat untuk tiap client.

### Agregasi dan Evaluasi
- [ ] Integrasikan update client DP dengan FedAvg.
- [ ] Jalankan training FL+DP.
- [ ] Simpan model global per round.
- [ ] Hitung accuracy, FAR, FRR, EER, dan F1-score.
- [ ] Hitung epsilon privacy budget pada setting federated.
- [ ] Ukur communication efficiency, convergence, dan training stability.
- [ ] Bandingkan FL+DP dengan baseline, DP saja, dan FL saja.
- [ ] Tulis kesimpulan apakah kombinasi ini sepadan.

## Level 5 - Privacy Attack Evaluation
### Attack Setup
- [ ] Tentukan target serangan untuk baseline, DP, FL, dan FL+DP.
- [ ] Tentukan apakah attack dilakukan pada output, gradient, atau parameter.
- [ ] Siapkan pipeline attack yang konsisten antar skenario.

### Membership Inference Attack
- [ ] Implementasikan attack pipeline untuk membedakan sampel train dan non-train.
- [ ] Hitung attack confidence.
- [ ] Hitung attack prediction untuk membership.
- [ ] Hitung attack accuracy.
- [ ] Hitung precision.
- [ ] Hitung recall.
- [ ] Hitung ROC-AUC.

### Leakage Analysis
- [ ] Tulis analisis fingerprinting.
- [ ] Tulis analisis user deanonymization.
- [ ] Tulis analisis gradient leakage.
- [ ] Tulis analisis model inversion.
- [ ] Bandingkan tingkat kebocoran pada baseline, DP, FL, dan FL+DP.

## Level 6 - Comparative Analysis and Utility vs Privacy
- [ ] Konsolidasikan seluruh metrik baseline, DP, FL, dan FL+DP.
- [ ] Pastikan definisi metrik konsisten di semua skenario.
- [ ] Buat tabel perbandingan utama.
- [ ] Plot accuracy, FAR, FRR, EER, dan F1-score.
- [ ] Plot epsilon dan MIA attack accuracy.
- [ ] Tulis analisis privacy-utility trade-off.
- [ ] Jelaskan skenario paling efisien untuk proteksi biometrik.

## Level 7 - Non-IID Federated Learning
### Non-IID Design
- [ ] Rancang pembagian client berdasarkan typing speed.
- [ ] Rancang pembagian client berdasarkan keyboard layout.
- [ ] Rancang pembagian client berdasarkan device type.
- [ ] Rancang pembagian client berdasarkan typing habits.
- [ ] Rancang pembagian client berdasarkan session duration.

### Non-IID Experiment
- [ ] Terapkan pembagian data heterogen antar client.
- [ ] Jalankan FL dengan konfigurasi yang sama seperti skenario IID.
- [ ] Ukur global accuracy.
- [ ] Ukur convergence stability.
- [ ] Ukur client drift.
- [ ] Ukur communication efficiency.
- [ ] Ukur fairness antar client.
- [ ] Tulis dampak non-IID terhadap FL dan DP.

## Level 8 - Ablation Study and Privacy-Utility Trade-off
### Parameter Selection
- [ ] Pilih epsilon.
- [ ] Pilih noise multiplier.
- [ ] Pilih clipping norm.
- [ ] Pilih sequence length.
- [ ] Pilih hidden units.
- [ ] Pilih jumlah client.
- [ ] Pilih local epoch.
- [ ] Pilih communication rounds.
- [ ] Pilih learning rate.

### Ablation Execution
- [ ] Uji satu parameter pada satu waktu jika memungkinkan.
- [ ] Catat hasil untuk setiap konfigurasi.
- [ ] Plot accuracy versus epsilon.
- [ ] Plot EER versus epsilon.
- [ ] Plot FAR versus epsilon.
- [ ] Plot FRR versus epsilon.
- [ ] Plot communication efficiency versus model performance.
- [ ] Plot MIA attack accuracy versus epsilon.
- [ ] Identifikasi parameter paling berpengaruh terhadap utility.
- [ ] Identifikasi parameter paling berpengaruh terhadap privacy.
- [ ] Identifikasi parameter paling berpengaruh terhadap convergence.
- [ ] Identifikasi parameter paling berpengaruh terhadap communication cost.

## Level 9 - Advanced Leakage Attack using Behavioral Reconstruction
### Attack Preparation
- [ ] Pilih apakah attack diarahkan ke gradient leakage atau behavioral reconstruction.
- [ ] Tetapkan input attack dan output yang ingin direkonstruksi.
- [ ] Siapkan attack pada FL tanpa DP.
- [ ] Siapkan attack pada FL+DP.

### Evaluation
- [ ] Jalankan attack pada FL tanpa DP.
- [ ] Catat kualitas rekonstruksi.
- [ ] Jalankan attack pada FL+DP.
- [ ] Bandingkan penurunan kemampuan rekonstruksi.
- [ ] Ukur typing reconstruction similarity.
- [ ] Ukur keystroke leakage rate.
- [ ] Tulis pengaruh DP terhadap kebocoran.

## Level 10 - Explainability and Threat Modeling
### Explainability
- [ ] Pilih SHAP atau attention visualization.
- [ ] Siapkan pipeline visualisasi untuk fitur keystroke.
- [ ] Analisis dwell time.
- [ ] Analisis flight time.
- [ ] Analisis typing speed.
- [ ] Analisis inter-key latency.
- [ ] Analisis typing rhythm.

### Threat Model
- [ ] Petakan honest-but-curious server.
- [ ] Petakan malicious client.
- [ ] Petakan external attacker.
- [ ] Petakan insider threat.
- [ ] Tulis risiko keystroke fingerprinting untuk baseline.
- [ ] Tulis risiko membership inference untuk baseline, DP, FL, dan FL+DP.
- [ ] Tulis risiko gradient leakage untuk FL dan FL+DP.
- [ ] Tulis ringkasan skenario paling aman secara relatif.

## Finalization Checklist
- [ ] Rapikan notebook final agar dapat dijalankan ulang.
- [ ] Susun laporan penelitian lengkap.
- [ ] Susun video penjelasan yang runtut.
- [ ] Cek konsistensi angka dan narasi antara notebook, laporan, dan video.
- [ ] Pastikan semua deliverable siap dikumpulkan.

## Milestone Checklist
- [ ] Milestone 1: Data audit dan preprocessing siap.
- [ ] Milestone 2: Baseline LSTM selesai dan tervalidasi.
- [ ] Milestone 3: DP selesai dan epsilon tercatat.
- [ ] Milestone 4: FL selesai dan global model berjalan.
- [ ] Milestone 5: FL+DP dan attack evaluation selesai.
- [ ] Milestone 6: Non-IID, ablation, explainability, dan threat model selesai.
- [ ] Milestone 7: Laporan, video, dan notebook final siap dikumpulkan.

## Reminder Implementasi Modular Notebook
- [x] Struktur notebook modular tersedia: 01, 02, 03, 04, 05, 06, 07, 99.
- [x] Penamaan notebook mengikuti format yang disarankan.
- [x] Tiap notebook menulis artefak ke folder standar (`data/processed`, `outputs/models`, `outputs/figures`, `outputs/reports`) sesuai kebutuhan tahap.
- [x] Chaining artefak antar notebook sudah aktif (audit -> preprocessing -> baseline -> dp/fl/attack -> final report bundle).
- [x] Shared logic utama menggunakan modul di `src` (data/models/evaluation), notebook berperan sebagai orchestration.
- [x] Runner otomatis tersedia di `scripts/run_modular_notebooks.py` (berbasis `nbclient`).
- [ ] Integrasi Flower end-to-end menggantikan placeholder simulasi pada notebook federated.
- [ ] Integrasi attack model MIA riil (shadow model/threshold attack formal) menggantikan placeholder skor proxy.
- [ ] Integrasi explainability riil (SHAP/Integrated Gradients/attention map) pada notebook ablation-explainability.
- [ ] Harmonisasi path fallback untuk lingkungan cloud vs lokal jika kernel tidak berada pada root project.
- [ ] Menambahkan validasi artefak wajib sebelum notebook `99_report_artifacts.ipynb` digabungkan ke laporan final.