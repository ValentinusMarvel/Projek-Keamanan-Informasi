# Keystroke Dynamics Task Breakdown

Dokumen ini memecah plan formal menjadi task implementasi yang sangat spesifik, terurut, dan siap dikerjakan per level.

Checklist granular per level tersedia di [docs/plan/keystroke-dynamics-checklist.md](docs/plan/keystroke-dynamics-checklist.md).

## Execution Order
1. Persiapan project dan environment.
2. Audit dataset dan desain preprocessing.
3. Baseline LSTM.
4. Differential Privacy.
5. Federated Learning.
6. FL + DP.
7. Privacy attack evaluation.
8. Non-IID FL.
9. Ablation study.
10. Explainability dan threat modeling.
11. Finalisasi laporan, video, dan notebook.

## Cross-Cutting Tasks
### CT-01 - Tetapkan target outcome dan success criteria
- Finalkan definisi output untuk klasifikasi pengguna, autentikasi, dan privacy evaluation.
- Tetapkan metrik utama yang wajib dilaporkan pada semua skenario.
- Pastikan skenario baseline, DP, FL, dan FL+DP memakai definisi evaluasi yang konsisten.

### CT-02 - Siapkan struktur project dan environment
- Buat struktur folder untuk data, notebooks, source code, figures, dan reports.
- Siapkan environment Python yang memuat PyTorch, Opacus, Flower, scikit-learn, pandas, numpy, matplotlib, dan seaborn.
- Tetapkan seed, logging strategy, dan format output hasil eksperimen.

### CT-03 - Standarkan artefak eksperimen
- Tetapkan format penamaan eksperimen, model checkpoints, dan tabel hasil.
- Buat template visualisasi agar setiap skenario mudah dibandingkan.
- Simpan konfigurasi eksperimen agar hasil dapat direplikasi.

## Level 1 - Baseline Machine Learning
### L1-01 - Audit dataset mentah
- Identifikasi file, format, delimiter, dan struktur kolom dataset.
- Petakan atribut yang merepresentasikan user, session, key press, dan key release.
- Catat missing value, duplikasi, dan anomali format.

### L1-02 - Definisikan skema fitur keystroke
- Turunkan dwell time, flight time, typing speed, inter-key latency, dan typing rhythm dari data mentah.
- Tentukan fitur yang dipakai sebagai input sequence model.
- Dokumentasikan bentuk tensor input dan label target.

### L1-03 - Bangun pipeline preprocessing dasar
- Implementasikan missing value handling.
- Implementasikan outlier filtering.
- Implementasikan session segmentation.
- Implementasikan sequence construction.
- Implementasikan normalization.
- Implementasikan sequence windowing.
- Implementasikan padding dan truncation.
- Implementasikan train-test split yang bebas leakage.

### L1-04 - Siapkan dataset model
- Ubah hasil preprocessing menjadi dataset train, validation, dan test.
- Verifikasi panjang sequence, dimensi fitur, dan distribusi label.
- Pastikan satu user tidak bocor ke split yang tidak semestinya.

### L1-05 - Implementasikan baseline LSTM
- Bangun model LSTM dengan input multivariat.
- Tambahkan layer klasifikasi akhir.
- Siapkan mode multiclass identification atau binary authentication.

### L1-06 - Latih baseline dan simpan checkpoint
- Jalankan training loop baseline.
- Simpan model terbaik berdasarkan validation metric.
- Catat loss curve dan perubahan akurasi per epoch.

### L1-07 - Evaluasi baseline
- Hitung accuracy.
- Hitung F1-score.
- Hitung FAR.
- Hitung FRR.
- Hitung EER.
- Buat confusion matrix.

### L1-08 - Analisis behavioral biometrics
- Identifikasi fitur keystroke yang paling membedakan pengguna.
- Tulis ringkasan pola pengetikan yang terlihat dari hasil baseline.

## Level 2 - Differential Privacy
### L2-01 - Tambahkan wrapper training DP
- Integrasikan Opacus ke pipeline training baseline.
- Pastikan model, optimizer, dan dataloader kompatibel dengan DP-SGD.

### L2-02 - Konfigurasikan clipping dan noise
- Tetapkan clipping norm.
- Tetapkan noise multiplier.
- Tetapkan target delta.
- Catat dampak parameter DP terhadap training.

### L2-03 - Aktifkan privacy accountant
- Hitung epsilon untuk setiap run.
- Simpan nilai epsilon per epoch atau per konfigurasi akhir.
- Validasi bahwa privacy budget tercatat konsisten.

### L2-04 - Latih model DP
- Jalankan training DP dengan konfigurasi yang setara dengan baseline.
- Catat convergence loss dan stability.
- Simpan checkpoint model privat.

### L2-05 - Evaluasi utility dan privasi
- Bandingkan accuracy, F1-score, FAR, FRR, dan EER terhadap baseline.
- Laporkan epsilon privacy budget.
- Dokumentasikan penurunan utility akibat noise.

### L2-06 - Analisis trade-off DP
- Tentukan konfigurasi DP yang paling seimbang antara utility dan privacy.
- Tulis interpretasi dampak privacy protection pada behavioral biometrics.

## Level 3 - Federated Learning
### L3-01 - Pecah data menjadi client dataset
- Tentukan strategi pembagian client.
- Distribusikan data secara IID terlebih dahulu sebagai baseline FL.
- Pastikan tidak ada leakage antar client.

### L3-02 - Siapkan client-local training loop
- Bangun fungsi training per client.
- Pastikan client mengirim parameter model, bukan data mentah.
- Catat local loss dan local accuracy.

### L3-03 - Implementasikan FedAvg
- Buat mekanisme agregasi bobot model dari seluruh client.
- Pastikan server menghitung model global dengan benar.
- Simpan model global per communication round.

### L3-04 - Jalankan simulasi FL
- Tentukan jumlah client.
- Tentukan local epochs.
- Tentukan communication rounds.
- Jalankan pelatihan federated end-to-end.

### L3-05 - Evaluasi FL
- Ukur global accuracy.
- Ukur convergence stability.
- Ukur communication efficiency.
- Ukur training latency.
- Bandingkan local versus global performance.

### L3-06 - Analisis privasi FL
- Tulis alasan mengapa FL mengurangi pemindahan data mentah.
- Catat risiko bahwa FL tetap mungkin bocor melalui parameter atau gradient.

## Level 4 - Federated Learning + Differential Privacy
### L4-01 - Terapkan DP di client lokal
- Tambahkan DP-SGD pada setiap client training loop.
- Verifikasi clipping dan noise berjalan per client.

### L4-02 - Integrasikan DP client dengan FedAvg
- Pastikan model dari client DP dapat di-aggregate oleh server.
- Validasi kompatibilitas update parameter antar client.

### L4-03 - Jalankan training FL+DP
- Gunakan konfigurasi client dan round yang sama dengan FL tanpa DP.
- Simpan model global per round.

### L4-04 - Evaluasi FL+DP
- Hitung accuracy, FAR, FRR, EER, dan F1-score.
- Hitung epsilon privacy budget pada setting federated.
- Ukur communication efficiency, convergence, dan training stability.

### L4-05 - Analisis utility versus privacy
- Bandingkan FL+DP dengan baseline, DP saja, dan FL saja.
- Tulis kesimpulan apakah kombinasi ini memberikan perlindungan yang sepadan.

## Level 5 - Privacy Attack Evaluation
### L5-01 - Definisikan skenario serangan
- Tentukan target serangan untuk baseline, DP, FL, dan FL+DP.
- Tetapkan apakah attack dilakukan pada output model, gradient, atau parameter.

### L5-02 - Implementasikan Membership Inference Attack
- Bangun attack pipeline untuk membedakan sampel train dan non-train.
- Ukur attack confidence dan prediksi membership.

### L5-03 - Evaluasi MIA
- Hitung attack accuracy.
- Hitung precision.
- Hitung recall.
- Hitung ROC-AUC.

### L5-04 - Analisis leakage lain
- Tulis analisis konseptual untuk fingerprinting.
- Tulis analisis konseptual untuk user deanonymization.
- Tulis analisis konseptual untuk gradient leakage.
- Tulis analisis konseptual untuk model inversion.

### L5-05 - Bandingkan tingkat kebocoran
- Bandingkan hasil serangan pada baseline, DP, FL, dan FL+DP.
- Identifikasi skenario paling rentan.

## Level 6 - Analisis Perbandingan dan Utility vs Privacy
### L6-01 - Konsolidasikan metrik utama
- Gabungkan hasil baseline, DP, FL, dan FL+DP dalam satu tabel perbandingan.
- Pastikan metrik yang dibandingkan memakai definisi yang sama.

### L6-02 - Buat grafik perbandingan
- Plot accuracy, FAR, FRR, EER, dan F1-score.
- Plot epsilon dan attack accuracy untuk tiap skenario.

### L6-03 - Analisis trade-off utama
- Jelaskan perubahan utility saat privasi meningkat.
- Jelaskan skenario yang paling efisien untuk proteksi biometrik.

## Level 7 - Non-IID Federated Learning
### L7-01 - Rancang skenario non-IID
- Bagi client berdasarkan typing speed.
- Bagi client berdasarkan keyboard layout.
- Bagi client berdasarkan device type.
- Bagi client berdasarkan typing habits.
- Bagi client berdasarkan session duration.

### L7-02 - Jalankan simulasi non-IID
- Terapkan pembagian data heterogen antar client.
- Jalankan FL dengan konfigurasi yang sama seperti skenario IID.

### L7-03 - Evaluasi dampak non-IID
- Ukur global accuracy.
- Ukur convergence stability.
- Ukur client drift.
- Ukur communication efficiency.
- Ukur fairness antar client.

### L7-04 - Analisis efek non-IID terhadap DP
- Tulis apakah heterogenitas client memperburuk atau memperbaiki hasil saat DP diterapkan.

## Level 8 - Ablation Study dan Privacy-Utility Trade-off
### L8-01 - Tentukan parameter ablation
- Pilih epsilon.
- Pilih noise multiplier.
- Pilih clipping norm.
- Pilih sequence length.
- Pilih hidden units.
- Pilih jumlah client.
- Pilih local epoch.
- Pilih communication rounds.
- Pilih learning rate.

### L8-02 - Jalankan eksperimen ablation
- Uji satu parameter pada satu waktu jika memungkinkan.
- Catat hasil untuk setiap konfigurasi.

### L8-03 - Visualisasikan hasil ablation
- Buat accuracy versus epsilon.
- Buat EER versus epsilon.
- Buat FAR versus epsilon.
- Buat FRR versus epsilon.
- Buat communication efficiency versus model performance.
- Buat MIA attack accuracy versus epsilon.

### L8-04 - Analisis sensitivitas parameter
- Identifikasi parameter yang paling memengaruhi utility.
- Identifikasi parameter yang paling memengaruhi privacy.
- Identifikasi parameter yang paling memengaruhi convergence.
- Identifikasi parameter yang paling memengaruhi communication cost.

## Level 9 - Advanced Leakage Attack menggunakan Behavioral Reconstruction
### L9-01 - Tentukan attack target
- Pilih apakah serangan diarahkan ke gradient leakage atau behavioral reconstruction.
- Tetapkan input attack dan output yang ingin direkonstruksi.

### L9-02 - Jalankan attack pada FL tanpa DP
- Gunakan model atau gradient dari skenario FL murni.
- Catat kualitas rekonstruksi.

### L9-03 - Jalankan attack pada FL+DP
- Ulangi attack pada skenario dengan DP.
- Bandingkan penurunan kemampuan rekonstruksi.

### L9-04 - Evaluasi leakage
- Ukur typing reconstruction similarity.
- Ukur keystroke leakage rate.
- Jelaskan pengaruh DP terhadap kebocoran.

## Level 10 - Explainability dan Threat Modeling
### L10-01 - Siapkan metode explainability
- Pilih SHAP atau attention visualization.
- Siapkan pipeline visualisasi untuk fitur keystroke.

### L10-02 - Interpretasikan fitur penting
- Analisis dwell time.
- Analisis flight time.
- Analisis typing speed.
- Analisis inter-key latency.
- Analisis typing rhythm.

### L10-03 - Susun threat model
- Petakan honest-but-curious server.
- Petakan malicious client.
- Petakan external attacker.
- Petakan insider threat.

### L10-04 - Dokumentasikan risiko per skenario
- Tulis risiko keystroke fingerprinting untuk baseline.
- Tulis risiko membership inference untuk baseline, DP, FL, dan FL+DP.
- Tulis risiko gradient leakage untuk FL dan FL+DP.

### L10-05 - Buat ringkasan final explainability dan keamanan
- Jelaskan fitur yang paling berpengaruh pada prediksi.
- Jelaskan skenario yang paling aman secara relatif.

## Finalization Tasks
### F-01 - Rapikan notebook final
- Satukan alur eksekusi yang dapat dijalankan ulang.
- Pastikan output penting tersimpan dan diberi narasi.

### F-02 - Susun laporan penelitian
- Susun pendahuluan, metodologi, hasil, pembahasan, dan kesimpulan.
- Masukkan tabel, grafik, dan analisis per level.

### F-03 - Susun video penjelasan
- Buat alur presentasi dari problem hingga hasil final.
- Tampilkan visualisasi yang paling representatif.

### F-04 - Validasi akhir artefak
- Cek konsistensi angka dan narasi antara notebook, laporan, dan video.
- Cek apakah semua deliverable siap dikumpulkan.

## Suggested Milestone Mapping
1. Milestone 1: Data audit dan preprocessing siap.
2. Milestone 2: Baseline LSTM selesai dan tervalidasi.
3. Milestone 3: DP selesai dan epsilon tercatat.
4. Milestone 4: FL selesai dan global model berjalan.
5. Milestone 5: FL+DP dan attack evaluation selesai.
6. Milestone 6: Non-IID, ablation, explainability, dan threat model selesai.
7. Milestone 7: Laporan, video, dan notebook final siap dikumpulkan.