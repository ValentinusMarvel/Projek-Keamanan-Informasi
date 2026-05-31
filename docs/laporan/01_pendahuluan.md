# Bab 1: Pendahuluan

## 1.1 Latar Belakang

Kemajuan teknologi informasi mendorong kebutuhan autentikasi yang lebih kuat dan adaptif. Metode autentikasi konvensional berbasis *knowledge factor* (kata sandi) rentan terhadap pencurian, serangan *brute-force*, dan teknik *social engineering*. Sementara itu, autentikasi berbasis *inherence factor* seperti sidik jari dan pengenalan wajah memerlukan perangkat keras khusus yang tidak selalu tersedia di seluruh perangkat. Dalam konteks ini, **behavioral biometrics** muncul sebagai solusi alternatif yang menjanjikan - mengidentifikasi pengguna berdasarkan *pola perilaku* yang unik dan sulit dipalsukan.

**Keystroke dynamics** merupakan salah satu modalitas behavioral biometrics yang memanfaatkan karakteristik temporal pengetikan pengguna - mencakup durasi penekanan tombol (*dwell time*), jeda antar-tombol (*flight time*), dan ritme pengetikan secara keseluruhan - sebagai fitur biometrik. Setiap individu memiliki pola mengetik yang berbeda dan relatif konsisten, sehingga pola tersebut dapat digunakan untuk identifikasi atau verifikasi identitas.

Namun, pemanfaatan data biometrik perilaku menimbulkan tantangan serius dari sisi **privasi**. Data keystroke mengandung informasi sensitif yang, jika bocor, dapat digunakan untuk *profiling* perilaku pengguna atau serangan *deanonymization*. Permasalahan ini menjadi semakin kritis ketika model machine learning dilatih secara terpusat pada data dari banyak pengguna - pola temporal yang tersimpan dalam bobot model berpotensi diekstraksi oleh pihak tidak berwenang.

Dua paradigma modern yang menjawab tantangan privasi ini adalah:

1. **Differential Privacy (DP)**: Memberikan jaminan matematis bahwa kehadiran atau ketiadaan satu data individu tidak memengaruhi distribusi output model secara signifikan. Implementasi melalui mekanisme DP-SGD (Differentially Private Stochastic Gradient Descent) menambahkan noise terukur pada proses pelatihan.

2. **Federated Learning (FL)**: Memungkinkan pelatihan model kolaboratif tanpa mengumpulkan data mentah ke satu server sentral. Setiap klien melatih model secara lokal dan hanya mengirimkan pembaruan bobot (*model updates*) ke server, menjaga data tetap di perangkat masing-masing.

Meskipun kedua mekanisme ini meningkatkan perlindungan privasi, masing-masing memiliki konsekuensi terhadap performa (*utility*) model. DP menambahkan noise yang dapat menurunkan akurasi, sementara FL menghadapi tantangan konvergensi terutama ketika distribusi data antar-klien bersifat heterogen (*non-IID*). Pertanyaan fundamental yang muncul adalah: **sejauh mana privasi dan utility dapat diseimbangkan pada sistem biometrik behavioral?**

## 1.2 Rumusan Masalah

Berdasarkan latar belakang tersebut, penelitian ini merumuskan masalah-masalah berikut:

1. Bagaimana membangun sistem identifikasi pengguna berbasis keystroke dynamics menggunakan arsitektur LSTM yang mampu menangkap dependensi temporal dalam pola pengetikan?

2. Bagaimana penerapan Differential Privacy (DP-SGD via Opacus) memengaruhi keseimbangan antara privasi dan utilitas model identifikasi?

3. Bagaimana Federated Learning (FedAvg via Flower) memungkinkan pelatihan kolaboratif tanpa sentralisasi data, dan bagaimana performanya dibandingkan model terpusat?

4. Bagaimana kombinasi FL + DP memberikan perlindungan berlapis, dan apa dampaknya terhadap metrik utilitas dan keamanan?

5. Seberapa rentan masing-masing konfigurasi model terhadap serangan privasi - khususnya Membership Inference Attack (MIA) dan Gradient Leakage Attack?

6. Fitur keystroke mana yang paling berpengaruh terhadap keputusan klasifikasi model, dan bagaimana threat model yang tepat untuk setiap skenario?

## 1.3 Tujuan Penelitian

Penelitian ini bertujuan untuk:

1. **Membangun pipeline end-to-end** untuk identifikasi pengguna berbasis keystroke dynamics, mulai dari preprocessing data mentah hingga evaluasi komprehensif.

2. **Mengembangkan model baseline LSTM** yang mencapai performa identifikasi optimal sebagai titik referensi perbandingan.

3. **Mengimplementasikan dan mengevaluasi** empat konfigurasi perlindungan privasi: Baseline (tanpa proteksi), DP-only, FL-only, dan FL+DP.

4. **Mengukur kerentanan privasi** setiap konfigurasi melalui Membership Inference Attack (MIA) dan Gradient Leakage Attack berbasis optimasi LBFGS.

5. **Melaksanakan studi ablasi** terhadap parameter privasi utama (epsilon, noise multiplier, clipping norm) untuk memetakan kurva *privacy-utility trade-off*.

6. **Menerapkan explainability** melalui Integrated Gradients untuk mengidentifikasi fitur keystroke paling diskriminatif.

7. **Menyusun threat model** yang komprehensif untuk setiap skenario pelatihan.

## 1.4 Batasan Penelitian

1. **Dataset**: Eksperimen menggunakan dataset DSL-StrongPasswordData dari Carnegie Mellon University (Killourhy & Maxion, 2009), berisi pengetikan kata sandi `.tie5Roanl` dari 51 subjek dengan 400 repetisi per subjek.

2. **Arsitektur Model**: Evaluasi dibatasi pada arsitektur LSTM (*Long Short-Term Memory*) dengan konfigurasi 2 layer × 64 hidden units.

3. **Simulasi FL**: Federated Learning disimulasikan menggunakan framework Flower dengan 5 klien dan 5 communication rounds untuk efisiensi komputasi pada lingkungan lokal.

4. **Mekanisme DP**: Differential Privacy diimplementasikan melalui Opacus (DP-SGD) dengan target privacy budget $\epsilon = 0.77$ pada $\delta = 10^{-5}$.

5. **Lingkungan Komputasi**: Seluruh eksperimen dijalankan pada mesin Windows dengan CPU-only (tanpa GPU), yang memengaruhi pilihan jumlah round dan epoch untuk menjaga kelayakan waktu eksekusi.

6. **Serangan Privasi**: Evaluasi serangan terbatas pada Membership Inference Attack (shadow classifier) dan Gradient Leakage Attack (LBFGS reconstruction).

## 1.5 Sistematika Penulisan

Laporan ini disusun dengan sistematika sebagai berikut:

- **Bab 1: Pendahuluan**: Memaparkan latar belakang, rumusan masalah, tujuan, batasan, dan sistematika penulisan.

- **Bab 2: Tinjauan Pustaka**: Mengulas teori dan penelitian terkait keystroke dynamics, LSTM, Differential Privacy, Federated Learning, serta serangan privasi pada model machine learning.

- **Bab 3: Metodologi**: Menjelaskan dataset, arsitektur sistem, pipeline preprocessing, desain eksperimen, dan konfigurasi parameter.

- **Bab 4: Hasil dan Pembahasan**: Menyajikan dan menganalisis hasil eksperimen pada setiap level, meliputi:
  - 4A: Baseline LSTM dan Differential Privacy
  - 4B: Federated Learning dan FL+DP
  - 4C: Evaluasi Serangan Privasi (MIA & Gradient Leakage)
  - 4D: Non-IID, Ablation Study, dan Advanced Transfer FL

- **Bab 5: Explainability & Threat Modeling**: Analisis interpretabilitas model dan pemodelan ancaman per skenario.

- **Bab 6: Analisis Komparatif**: Dashboard perbandingan menyeluruh dan analisis privacy-utility trade-off.

- **Bab 7: Kesimpulan & Rekomendasi**: Ringkasan temuan, keterbatasan, dan arah pengembangan selanjutnya.

- **Lampiran**: Konfigurasi teknis, daftar artefak, dan panduan reproduksi.
