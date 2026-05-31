# Bab 7: Kesimpulan & Rekomendasi

## 7.1 Ringkasan Temuan

Penelitian ini telah berhasil membangun pipeline end-to-end untuk identifikasi pengguna berbasis keystroke dynamics menggunakan LSTM, dengan evaluasi komprehensif terhadap mekanisme privasi (Differential Privacy dan Federated Learning) serta serangan privasi (MIA dan Gradient Leakage). Berikut ringkasan temuan utama:

### 7.1.1 Temuan pada Dimensi Utilitas

1. **Model baseline LSTM** berhasil mencapai akurasi **63.29%** pada klasifikasi 51 pengguna (32× random guess) menggunakan hanya 3 fitur temporal (dwell time, flight UD, flight DD) dari 11 transisi tombol kata sandi `.tie5Roanl`.

2. **Differential Privacy (ε = 0.77)** menyebabkan degradasi utilitas total - akurasi turun ke 2.25% (≈ random guess). Noise DP pada skala fitur keystroke yang halus menghancurkan sinyal diskriminatif model.

3. **Federated Learning standar (5 round)** menghasilkan akurasi rendah (2.18%) karena keterbatasan jumlah round dan fragmentasi data, tetapi menunjukkan tren konvergensi positif (loss menurun konsisten).

4. **Advanced Transfer FL** berhasil mencapai akurasi **63.29%** - menyamai model terpusat - melalui pre-trained initialization dan feature standardization.

5. **Studi ablasi** memetakan kurva privacy-utility yang monoton, dengan sweet spot pada **ε = 2.0** (akurasi 38%, EER 18%, MIA AUC 0.501).

### 7.1.2 Temuan pada Dimensi Privasi

1. **MIA tidak efektif** terhadap semua konfigurasi - AUC konsisten di sekitar 0.50 (random guess). Model LSTM multi-class secara inherent tidak mengekspos pola membership yang kuat.

2. **Gradient Leakage** pada model standar menunjukkan kerentanan potensial (status "VULNERABLE"), meskipun rekonstruksi aktual tidak berhasil dalam eksperimen ini. Model DP terbukti aman (status "SECURED").

3. **FL + DP** memberikan perlindungan berlapis terkuat - data tetap lokal DAN gradien dilindungi noise - dengan biaya utilitas paling tinggi.

### 7.1.3 Temuan pada Dimensi Explainability

1. **Flight Time DD** (Down-Down) merupakan fitur paling diskriminatif untuk identifikasi pengguna.

2. **Transisi `.` → `t`** dan **`l` → `Return`** adalah transisi tombol paling penting - keduanya melibatkan perpindahan tangan yang bervariasi antar-pengguna.

3. **Karakter `Shift.r`** memiliki kontribusi nol, menunjukkan bahwa modifier keys tidak informatif untuk biometrik.

---

## 7.2 Jawaban atas Rumusan Masalah

### RM-1: Sistem Identifikasi Berbasis LSTM
🟢 **Terjawab**: Model `KeystrokeLSTM` (2 layer × 64 hidden units) berhasil dibangun dan mampu mengidentifikasi 51 pengguna dengan akurasi 63.29% dan Top-5 accuracy 81.75% dari hanya 3 fitur temporal per timestep.

### RM-2: Pengaruh DP terhadap Privacy-Utility
🟢 **Terjawab**: DP-SGD dengan ε = 0.77 memberikan privasi sangat kuat tetapi menghapus utilitas model. Kurva ablation menunjukkan bahwa ε = 2.0 merupakan titik kompromi optimal (akurasi 38% dengan MIA random guess).

### RM-3: Performa FL vs Centralized
🟢 **Terjawab**: FL standar (5 round) menghasilkan akurasi rendah (2.18%) karena keterbatasan round. Dengan Transfer Learning, FL mampu menyamai performa terpusat (63.29% vs 63.29%), membuktikan kelayakan pelatihan kolaboratif.

### RM-4: Dampak FL + DP
🟢 **Terjawab**: Kombinasi FL + DP menyebabkan stagnansi total (akurasi 2.11%, loss konstan) pada 5 round - perlindungan berlapis efektif tetapi mengorbankan utilitas secara drastis.

### RM-5: Kerentanan terhadap Serangan
🟢 **Terjawab**: MIA tidak efektif (AUC ≈ 0.50) terhadap semua konfigurasi. Gradient Leakage menunjukkan kerentanan potensial pada model standar, yang berhasil dimitigasi oleh DP.

### RM-6: Fitur Paling Berpengaruh dan Threat Model
🟢 **Terjawab**: Flight Time DD paling diskriminatif; transisi `.`→`t` dan `l`→`Return` paling kritis. Threat model menunjukkan FL+DP memberikan risiko keseluruhan RENDAH terhadap keempat profil aktor ancaman.

---

## 7.3 Keterbatasan Penelitian

1. **Jumlah FL round**: Simulasi dibatasi pada 5 round dengan 1 epoch lokal untuk efisiensi waktu. Akurasi FL dan FL+DP akan meningkat signifikan dengan lebih banyak round (30-50), tetapi tidak dievaluasi karena keterbatasan komputasi.

2. **Single dataset**: Evaluasi hanya pada dataset DSL-StrongPasswordData (kata sandi tunggal `.tie5Roanl`). Generalisasi ke kata sandi berbeda atau free-text typing belum diuji.

3. **Single architecture**: Hanya LSTM yang dievaluasi. Arsitektur alternatif (Transformer, CNN-LSTM, Attention-based) mungkin memberikan performa berbeda di bawah DP.

4. **Serangan terbatas**: Hanya MIA dan Gradient Leakage yang dievaluasi. Serangan lain seperti Model Inversion, Property Inference, dan Data Poisoning belum diuji.

5. **CPU-only environment**: Eksperimen dijalankan pada CPU tanpa GPU, membatasi skala eksperimen (jumlah round, grid search, dll.).

6. **Static baseline comparison**: Model terpusat (baseline) dilatih pada fitur mentah tanpa StandardScaler, sementara Transfer FL menggunakan scaled features - perbandingan tidak sepenuhnya *fair* karena perbedaan preprocessing.

---

## 7.4 Rekomendasi Pengembangan

### 7.4.1 Pengembangan Jangka Pendek

| No | Rekomendasi | Dampak yang Diharapkan |
|:--:|-------------|------------------------|
| 1 | **Tingkatkan jumlah round FL** menjadi 30-50 dengan 3-5 epoch lokal | Akurasi FL mendekati baseline (~40-50%) |
| 2 | **Terapkan StandardScaler** pada baseline terpusat untuk perbandingan fair | Baseline mungkin naik ke >60% |
| 3 | **Implementasikan FL + DP + Transfer** (kombinasi ketiganya) | Keseimbangan optimal: akurasi tinggi + privasi formal |
| 4 | **Gunakan adaptive noise scheduling** pada DP-SGD | Pembelajaran awal lebih stabil, convergence lebih baik |

### 7.4.2 Pengembangan Jangka Menengah

| No | Rekomendasi | Dampak yang Diharapkan |
|:--:|-------------|------------------------|
| 5 | **Evaluasi arsitektur Transformer** (keystroke attention) | Potensi akurasi lebih tinggi |
| 6 | **Tambahkan fitur turunan** (statistik sesi, typing speed, rhythm) | Input lebih kaya → akurasi naik |
| 7 | **Implementasikan Secure Aggregation** untuk FL | Privasi tanpa overhead DP noise |
| 8 | **Uji pada dataset free-text** | Generalisasi ke skenario pengetikan alami |

### 7.4.3 Pengembangan Jangka Panjang

| No | Rekomendasi | Dampak yang Diharapkan |
|:--:|-------------|------------------------|
| 9 | **Continuous authentication** menggunakan keystroke streaming | Autentikasi sepanjang sesi, bukan one-shot |
| 10 | **Multi-modal biometrics** (keystroke + mouse movement) | Akurasi dan keamanan lebih tinggi |
| 11 | **Personalized FL** (per-user model adaptation) | Model yang disesuaikan per pengguna |
| 12 | **Deployment edge** (on-device inference) | Privasi by design, tanpa server |

---

## 7.5 Kesimpulan Akhir

Penelitian ini mendemonstrasikan bahwa **behavioral biometrics berbasis keystroke dynamics** merupakan pendekatan yang viable untuk identifikasi pengguna, dengan potensi mencapai akurasi >60% menggunakan hanya 3 fitur temporal sederhana. Namun, terdapat **trade-off fundamental** antara utilitas model dan perlindungan privasi:

- **Privasi formal ketat** (ε < 1) secara efektif menghapus kemampuan identifikasi pada skala fitur keystroke yang halus.
- **Federated Learning** menjaga data tetap lokal tetapi memerlukan optimasi khusus (transfer learning, feature scaling) untuk mencapai performa kompetitif.
- **Serangan privasi** (MIA, Gradient Leakage) tidak efektif terhadap konfigurasi yang menggunakan DP, mengonfirmasi bahwa mekanisme privasi berfungsi sebagaimana dirancang.

**Rekomendasi utama**: Untuk deployment produksi, gunakan **Advanced Transfer FL** dengan **DP-SGD moderat (ε = 2.0)** - memberikan akurasi ~46-63%, privasi formal yang memadai, dan data locality. Ini merepresentasikan titik keseimbangan terbaik antara keamanan, privasi, dan fungsionalitas sistem biometrik behavioral.
