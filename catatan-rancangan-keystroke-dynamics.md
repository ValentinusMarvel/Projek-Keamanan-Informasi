# Catatan Rancangan Tugas Besar KI

## Judul Kerja
Sistem Identifikasi Pengguna Berbasis Keystroke Dynamics dengan LSTM, Differential Privacy, Federated Learning, dan Evaluasi Serangan Privasi

## Tujuan Utama
Membangun alur riset dan implementasi yang utuh untuk menilai hubungan antara utility model dan privasi pada behavioral biometrics pengetikan. Sistem akan dimulai dari baseline machine learning berbasis LSTM, lalu dikembangkan dengan Differential Privacy, Federated Learning, kombinasi FL+DP, evaluasi serangan privasi, studi ablation, hingga explainability dan threat modeling.

## Ruang Lingkup
Rancangan ini mencakup:
1. Pengolahan dataset keystroke dynamics dari Kaggle.
2. Pembangunan baseline klasifikasi/identifikasi pengguna dengan LSTM.
3. Integrasi Differential Privacy menggunakan Opacus.
4. Simulasi Federated Learning menggunakan Flower.
5. Kombinasi Federated Learning + Differential Privacy.
6. Evaluasi Membership Inference Attack dan attack leakage lain.
7. Analisis non-IID, ablation study, dan privacy-utility trade-off.
8. Explainability dan threat model pada tiap skenario.
9. Penyusunan laporan, video presentasi, dan notebook artefak.

## Asumsi Awal
1. Dataset tersedia dalam bentuk log keystroke per pengguna dan per sesi.
2. Eksperimen utama dilakukan di Python dengan PyTorch.
3. Opsi pengelompokan tugas dapat dibuat per level agar pipeline tetap rapi.
4. Evaluasi dilakukan pada beberapa skenario yang konsisten agar perbandingan valid.
5. Fokus utama adalah perilaku pengetikan sebagai biometrik, bukan identitas statis.

## Prinsip Desain Sistem
1. Pipeline harus modular agar baseline, DP, FL, dan FL+DP dapat dibandingkan secara adil.
2. Semua preprocessing harus ditetapkan secara deterministik dan terdokumentasi.
3. Split data harus mencegah kebocoran antara train, validation, dan test.
4. Semua pengukuran privasi dan utility harus dicatat pada konfigurasi eksperimen yang sama.
5. Hasil akhir harus mudah direplikasi dari notebook.

## Arsitektur Tingkat Tinggi
### Lapisan Data
- Dataset mentah keystroke.
- Data cleaning dan preprocessing.
- Segmentasi sesi dan pembentukan sequence.
- Dataset siap latih untuk baseline, client FL, dan analisis serangan.

### Lapisan Model
- LSTM untuk klasifikasi pengguna atau autentikasi.
- Versi baseline tanpa proteksi privasi.
- Versi DP-SGD dengan Opacus.
- Versi FL dengan FedAvg.
- Versi FL+DP pada local client training.

### Lapisan Evaluasi
- Kinerja identifikasi.
- Kinerja privasi.
- Kinerja komunikasi FL.
- Kinerja serangan privasi.
- Interpretabilitas model.

### Lapisan Pelaporan
- Tabel hasil per level.
- Grafik komparatif.
- Analisis trade-off.
- Threat model.
- Kesimpulan dan rekomendasi.

## Definisi Output Akhir
1. Laporan penelitian dalam format .docx.
2. Video penjelasan dalam format .mp4.
3. Notebook implementasi dalam format .ipynb.
4. Artefak visualisasi hasil eksperimen.
5. Ringkasan eksperimen dan konfigurasi untuk replikasi.

## Pipeline End-to-End
### Tahap 1: Akuisisi dan Pemahaman Data
- Unduh dataset dari Kaggle.
- Identifikasi format file, struktur kolom, jumlah user, jumlah sesi, dan panjang sequence.
- Petakan fitur mentah ke fitur turunan yang dibutuhkan.
- Catat potensi label, sesi, dan user boundary.

### Tahap 2: Preprocessing
- Missing value handling.
- Outlier filtering.
- Session segmentation.
- Temporal sequence construction.
- Normalization.
- Sequence windowing.
- Padding/truncation.
- Train-test split.

### Tahap 3: Baseline Modeling
- Bentuk dataset supervised untuk identifikasi pengguna.
- Latih model LSTM.
- Evaluasi baseline pada test set.

### Tahap 4: Privasi dan Federasi
- Terapkan DP-SGD dengan Opacus.
- Simulasikan Federated Learning dengan Flower dan FedAvg.
- Gabungkan FL dan DP pada client-side training.

### Tahap 5: Analisis Serangan
- Jalankan Membership Inference Attack.
- Uji potensi fingerprinting dan leakage lain secara konseptual dan empiris.

### Tahap 6: Analisis Lanjutan
- Non-IID simulation.
- Ablation study.
- Explainability.
- Threat model.

### Tahap 7: Dokumentasi
- Susun laporan.
- Rekam video presentasi.
- Rapikan notebook.

## Spesifikasi Fitur Keystroke
Fitur utama yang dipakai untuk merepresentasikan perilaku pengetikan:
1. Key press timing.
2. Key release timing.
3. Dwell time.
4. Flight time.
5. Typing speed.
6. Inter-key latency.
7. Typing rhythm.

## Rekayasa Fitur Turunan
Fitur mentah dapat diperkaya menjadi:
1. Durasi tekan tiap tombol.
2. Jarak waktu antar tombol berurutan.
3. Statistik per sesi seperti mean, variance, dan tempo.
4. Sequence multivariat per window.
5. Normalized rhythm profile per user.

## Desain Preprocessing
### 1. Missing Value Handling
- Identifikasi nilai kosong atau event yang tidak lengkap.
- Strategi: drop jika sangat sedikit, imputasi jika konsisten dan aman, atau segmentasi ulang bila data rusak.

### 2. Outlier Filtering
- Deteksi interval yang tidak realistis.
- Gunakan batas statistik atau quantile-based filtering.
- Pastikan outlier tidak menghapus variasi alami perilaku pengguna.

### 3. Session Segmentation
- Pisahkan data berdasarkan sesi pengetikan.
- Hindari sequence yang melintasi batas sesi bila itu mengaburkan pola temporal.

### 4. Temporal Sequence Construction
- Susun fitur dalam urutan waktu.
- Bentuk sequence multivariat yang mempertahankan dependency temporal.

### 5. Normalization
- Terapkan standardization atau min-max scaling sesuai karakter distribusi.
- Fit scaler hanya pada train set.

### 6. Sequence Windowing
- Bagi sequence menjadi window tetap.
- Atur overlap jika diperlukan untuk menjaga kontinuitas temporal.

### 7. Padding/Truncation
- Samakan panjang sequence untuk input LSTM.
- Gunakan padding mask bila diperlukan.

### 8. Train-Test Split
- Split harus dilakukan secara user-aware atau session-aware sesuai tujuan eksperimen.
- Untuk skenario identifikasi, hindari kebocoran sesi pengguna yang sama ke train dan test bila ingin menguji generalisasi.

## Desain Baseline ML
### Tujuan
Membangun model referensi yang menjadi pembanding untuk seluruh skenario privasi.

### Model Inti
- Input: sequence keystroke multivariat.
- Layer LSTM satu atau beberapa lapis.
- Fully connected layer untuk klasifikasi user.
- Softmax output untuk multi-class user identification.

### Variasi Output
1. User identification multiclass.
2. User authentication biner per target user.
3. Behavioral biometric classification untuk pola pengetikan.

### Output yang Harus Dicatat
- Loss train dan validation.
- Accuracy.
- F1-score.
- Confusion matrix.
- FAR.
- FRR.
- EER.

## Level 1 — Baseline Machine Learning
### Target
Mendapatkan performa dasar model LSTM pada data keystroke tanpa perlindungan privasi.

### Langkah Kerja
1. Pahami struktur data dan label.
2. Lakukan preprocessing lengkap.
3. Bentuk sequence input dan label target.
4. Latih LSTM baseline.
5. Evaluasi pada test set.
6. Analisis pola keystroke yang paling dominan.

### Output Level 1
- Model baseline.
- Tabel metrik klasifikasi.
- Confusion matrix.
- Analisis behavioral biometrics awal.

## Level 2 — Differential Privacy
### Target
Melindungi model dari kebocoran perilaku pengetikan melalui training privat.

### Implementasi
1. Gunakan Opacus untuk DP-SGD.
2. Terapkan per-sample gradient clipping.
3. Tambahkan Gaussian noise pada gradient.
4. Hitung epsilon privacy budget dengan privacy accountant.

### Parameter Penting
- Noise multiplier.
- Clipping norm.
- Batch size.
- Learning rate.
- Target delta.

### Evaluasi Tambahan
- Epsilon privacy budget.
- Convergence loss.
- Training stability.
- Penurunan utility dibanding baseline.

### Analisis
- Bandingkan baseline vs DP pada utility dan privasi.
- Identifikasi titik di mana privasi meningkat tetapi performa masih dapat diterima.

## Level 3 — Federated Learning
### Target
Mensimulasikan pembelajaran kolaboratif tanpa memindahkan data mentah ke server pusat.

### Implementasi
1. Gunakan Flower sebagai framework federated.
2. Bagi data ke beberapa client.
3. Terapkan FedAvg untuk agregasi model.
4. Jalankan beberapa communication rounds.

### Evaluasi Tambahan
- Communication efficiency.
- Number of rounds.
- Federated convergence.
- Local vs global model performance.
- Training latency.

### Analisis
- Apakah FL mempertahankan utility yang cukup baik?
- Seberapa stabil model global terhadap variasi client?

## Level 4 — Federated Learning + Differential Privacy
### Target
Menggabungkan DP-SGD pada local client dengan agregasi FedAvg.

### Implementasi
1. Setiap client melatih model lokal dengan DP-SGD.
2. Bobot model dikirim ke server, bukan data mentah.
3. Server melakukan FedAvg.
4. Evaluasi privacy budget pada level local training.

### Evaluasi
- Accuracy.
- FAR.
- FRR.
- EER.
- F1-score.
- Epsilon privacy budget.
- Communication efficiency.
- Federated convergence.
- Training stability.

### Analisis
- Seberapa besar penurunan utility akibat kombinasi proteksi ganda.
- Apakah FL+DP memberi perlindungan yang lebih kuat daripada DP atau FL saja.

## Level 5 — Privacy Attack Evaluation
### Target
Mengukur seberapa besar kebocoran informasi yang masih bisa dimanfaatkan penyerang.

### Serangan yang Dianalisis
1. Membership Inference Attack.
2. Keystroke fingerprinting.
3. User deanonymization.
4. Gradient leakage attack.
5. Model inversion attack.
6. Typing behavior profiling.

### Evaluasi MIA
- Attack accuracy.
- Precision.
- Recall.
- ROC-AUC.

### Analisis Kebocoran
- Bandingkan baseline, DP, FL, dan FL+DP.
- Lihat apakah parameter model masih mengandung sinyal identitas yang kuat.

### Output
- Tabel serangan per skenario.
- Grafik ROC bila relevan.
- Ringkasan tingkat kebocoran data.

## Level 6 — Analisis Perbandingan dan Utility vs Privacy
### Target
Menetapkan perbandingan komprehensif antar metode.

### Skema Perbandingan
1. Baseline ML.
2. Differential Privacy.
3. Federated Learning.
4. Federated Learning + Differential Privacy.

### Metrik Perbandingan
- Accuracy.
- FAR.
- FRR.
- EER.
- F1-score.
- Communication efficiency.
- Federated convergence.
- MIA attack accuracy.
- Epsilon privacy budget.

### Analisis Utama
- Jelaskan privacy-utility trade-off.
- Tentukan metode yang paling aman namun masih layak secara performa.

## Level 7 — Non-IID Federated Learning
### Target
Mengukur ketahanan FL ketika distribusi data antar client berbeda.

### Bentuk Non-IID
1. Typing speed berbeda.
2. Keyboard layout berbeda.
3. Device type berbeda.
4. Typing habits berbeda.
5. Session duration berbeda.

### Evaluasi
- Global model accuracy.
- Federated convergence stability.
- Client drift.
- Communication efficiency.
- Fairness antar client.

### Analisis
- Apakah heterogenitas perilaku pengetikan memperlambat konvergensi?
- Apakah DP memperburuk atau memperbaiki stabilitas pada non-IID setting?

## Level 8 — Ablation Study dan Privacy-Utility Trade-off
### Target
Menentukan parameter paling berpengaruh terhadap performa dan privasi.

### Parameter yang Diuji
1. Epsilon privacy budget.
2. Noise multiplier.
3. Clipping norm.
4. Sequence length.
5. LSTM hidden units.
6. Jumlah client.
7. Local epoch.
8. Communication rounds.
9. Learning rate.

### Visualisasi yang Wajib Ada
- Accuracy vs epsilon.
- EER vs epsilon.
- FAR vs epsilon.
- FRR vs epsilon.
- Communication efficiency vs model performance.
- MIA attack accuracy vs epsilon.

### Analisis
- Parameter mana yang paling memengaruhi utility.
- Parameter mana yang paling memengaruhi privacy.
- Parameter mana yang paling memengaruhi stabilitas federated training.
- Parameter mana yang paling memengaruhi efisiensi komunikasi.

## Level 9 — Advanced Leakage Attack menggunakan Behavioral Reconstruction
### Target
Menilai apakah pola pengetikan dapat direkonstruksi dari gradient atau parameter model.

### Implementasi dan Analisis
1. Jalankan gradient leakage attack atau behavioral reconstruction attack.
2. Bandingkan hasil pada FL tanpa DP dan FL+DP.
3. Ukur similarity antara data rekonstruksi dan data asli.
4. Catat leakage rate.

### Evaluasi
- Typing reconstruction similarity.
- Keystroke leakage rate.
- Dampak DP pada penurunan kebocoran.

### Output
- Contoh hasil rekonstruksi bila memungkinkan.
- Tabel perbandingan risiko leakage.

## Level 10 — Explainability dan Threat Modeling
### Explainability
Gunakan SHAP atau attention visualization untuk menilai kontribusi fitur:
1. Dwell time.
2. Flight time.
3. Typing speed.
4. Inter-key latency.
5. Typing rhythm pattern.

### Threat Model Analysis
Susun threat model untuk skenario:
1. Honest-but-curious server.
2. Malicious client.
3. External attacker.
4. Insider threat.

### Risiko yang Dibahas
- Keystroke fingerprinting risk.
- User deanonymization risk.
- Gradient leakage risk.
- Membership inference risk.

### Per Skenario
Analisis threat model untuk:
1. Baseline ML.
2. DP.
3. FL.
4. FL+DP.

## Rencana Eksperimen yang Disarankan
### Eksperimen Inti
1. Baseline LSTM.
2. LSTM + DP.
3. FL FedAvg tanpa DP.
4. FL + DP.
5. MIA pada keempat skenario.
6. Non-IID FL.
7. Ablation study.
8. Leakage attack lanjutan.
9. Explainability.

### Prioritas Urutan Pengerjaan
1. Dataset understanding.
2. Preprocessing final.
3. Baseline model.
4. DP extension.
5. FL extension.
6. FL+DP integration.
7. Attack evaluation.
8. Non-IID and ablation.
9. Explainability and threat modeling.
10. Report and presentation.

## Struktur Notebook yang Disarankan
1. Intro dan objective.
2. Dataset loading and inspection.
3. Preprocessing pipeline.
4. Baseline training.
5. DP training.
6. Federated learning simulation.
7. FL+DP.
8. Attack evaluation.
9. Non-IID study.
10. Ablation study.
11. Explainability.
12. Threat model summary.
13. Final comparison tables and plots.

## Struktur Laporan .docx
### Bab 1 — Pendahuluan
- Latar belakang.
- Rumusan masalah.
- Tujuan.
- Batasan.

### Bab 2 — Tinjauan Pustaka
- Keystroke dynamics.
- LSTM.
- Differential Privacy.
- Federated Learning.
- Membership inference dan leakage attacks.

### Bab 3 — Metodologi
- Dataset.
- Preprocessing.
- Arsitektur model.
- DP-SGD.
- FL FedAvg.
- Experimental design.

### Bab 4 — Hasil dan Pembahasan
- Hasil baseline.
- Hasil DP.
- Hasil FL.
- Hasil FL+DP.
- Hasil serangan.
- Ablation.
- Explainability.

### Bab 5 — Kesimpulan
- Ringkasan temuan.
- Trade-off utama.
- Keterbatasan.
- Rekomendasi lanjutan.

## Desain Video Penjelasan .mp4
### Urutan Isi Video
1. Masalah dan tujuan tugas besar.
2. Dataset dan fitur keystroke.
3. Preprocessing pipeline.
4. Baseline LSTM.
5. Differential Privacy.
6. Federated Learning.
7. FL+DP.
8. Privacy attacks.
9. Non-IID dan ablation.
10. Explainability dan threat model.
11. Kesimpulan akhir.

### Prinsip Penyampaian
- Singkat, runtut, dan berbasis visual.
- Tampilkan tabel dan grafik utama.
- Jelaskan trade-off utility vs privacy secara jelas.

## Visualisasi yang Perlu Disiapkan
1. Distribusi fitur keystroke.
2. Contoh sequence sebelum dan sesudah padding.
3. Learning curve baseline.
4. Confusion matrix tiap skenario.
5. FAR, FRR, dan EER.
6. Privacy budget epsilon.
7. Loss convergence FL.
8. Accuracy per communication round.
9. MIA ROC curve.
10. Grafik ablation.
11. Grafik trade-off privacy-utility.
12. Explainability plot atau attention heatmap.

## Kriteria Keberhasilan
1. Baseline berjalan dan menghasilkan metrik klasifikasi yang valid.
2. DP berhasil menurunkan risiko kebocoran dengan epsilon yang terukur.
3. FL berhasil melakukan pelatihan terdistribusi tanpa data mentah berpindah.
4. FL+DP menunjukkan perlindungan privasi lebih kuat dibanding baseline.
5. Serangan privasi dapat diukur dan dibandingkan lintas skenario.
6. Laporan, video, dan notebook konsisten satu sama lain.

## Risiko Teknis
1. Dataset tidak seragam atau format tidak sesuai ekspektasi.
2. Sequence terlalu pendek atau terlalu panjang.
3. DP menyebabkan penurunan performa yang terlalu besar.
4. FL pada non-IID membuat konvergensi tidak stabil.
5. Serangan privasi sulit direplikasi penuh karena keterbatasan data atau resource.
6. Waktu komputasi tinggi untuk eksperimen berulang.

## Mitigasi Risiko
1. Buat pipeline preprocessing yang terdokumentasi ketat.
2. Uji beberapa panjang sequence dan pilih yang paling stabil.
3. Gunakan grid kecil untuk tuning DP dan FL.
4. Catat seed eksperimen agar hasil konsisten.
5. Simpan hasil per eksperimen dalam tabel ringkas.
6. Jika compute terbatas, prioritaskan baseline, DP, FL, FL+DP, dan MIA terlebih dahulu.

## Rencana Kerja Bertahap
### Tahap A — Persiapan
- Unduh dataset.
- Baca data mentah.
- Finalisasi skema preprocessing.

### Tahap B — Implementasi Inti
- Bangun baseline.
- Tambahkan evaluasi lengkap.

### Tahap C — Privasi
- Tambahkan DP.
- Jalankan pengukuran epsilon.

### Tahap D — Federasi
- Implementasikan FL.
- Uji non-IID.

### Tahap E — Kombinasi dan Serangan
- Bangun FL+DP.
- Jalankan MIA dan leakage evaluation.

### Tahap F — Analisis Lanjutan
- Ablation.
- Explainability.
- Threat model.

### Tahap G — Finalisasi
- Laporan.
- Video.
- Notebook final.

## Format Tabel Hasil yang Disarankan
### Tabel 1 — Baseline dan Varian Privasi
Kolom: metode, accuracy, FAR, FRR, EER, F1-score, epsilon, catatan.

### Tabel 2 — FL Metrics
Kolom: metode, rounds, local accuracy, global accuracy, latency, communication efficiency, convergence note.

### Tabel 3 — Attack Metrics
Kolom: metode, MIA accuracy, precision, recall, ROC-AUC, leakage note.

### Tabel 4 — Ablation
Kolom: parameter, nilai, accuracy, EER, epsilon, attack accuracy, kesimpulan.

## Catatan Analisis yang Harus Ditulis di Pembahasan
1. Mengapa perilaku pengetikan dapat digunakan sebagai biometrik.
2. Mengapa LSTM sesuai untuk dependency temporal.
3. Bagaimana DP menambah noise tetapi menekan leakage.
4. Bagaimana FL mengurangi kebutuhan sentralisasi data.
5. Mengapa FL belum tentu aman dari attack berbasis gradient.
6. Bagaimana FL+DP menyeimbangkan privasi dan utility.
7. Mengapa non-IID menjadi tantangan besar pada keystroke biometrics.
8. Fitur apa yang paling berpengaruh terhadap keputusan model.

## Checklist Final Pengumpulan
### Laporan
- Struktur lengkap.
- Tabel dan gambar konsisten.
- Semua metrik utama ada.
- Analisis trade-off jelas.

### Video
- Narasi runtut.
- Menunjukkan pipeline dan hasil utama.
- Durasi sesuai kebutuhan tugas.

### Notebook
- Bisa dijalankan ulang.
- Seluruh cell penting diberi penjelasan.
- Hasil tersimpan dan terorganisasi.

## Kesimpulan Rancangan
Rancangan ini menyusun eksperimen secara bertahap dari baseline hingga skenario paling kompleks. Urutan ini penting agar setiap kenaikan level punya pembanding yang jelas, sehingga dampak Differential Privacy, Federated Learning, non-IID, dan serangan privasi dapat diukur secara adil. Fokus utama tetap pada keseimbangan antara akurasi identifikasi pengguna dan perlindungan behavioral biometrics.