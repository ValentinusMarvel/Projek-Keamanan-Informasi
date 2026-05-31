# Bab 4A: Hasil: Baseline LSTM & Differential Privacy

## 4A.1 Hasil Baseline LSTM (Level 1)

### 4A.1.1 Performa Klasifikasi

Model baseline LSTM dilatih secara terpusat (centralized) pada seluruh data tanpa mekanisme perlindungan privasi. Hasil evaluasi pada test set:

| Metrik | Nilai |
|--------|:-----:|
| **Accuracy** | **47.40%** |
| **Precision** (macro) | 53.54% |
| **Recall** (macro) | 47.40% |
| **F1-Score** (macro) | 47.48% |
| **Top-3 Accuracy** | 72.53% |
| **Top-5 Accuracy** | 81.75% |
| **Loss** | 1.990 |

### 4A.1.2 Analisis Performa Baseline

**Konteks interpretasi**: Dengan 51 kelas subjek, *random guess baseline* adalah $\frac{1}{51} \approx 1.96\%$. Akurasi **47.40%** menunjukkan bahwa model berhasil mempelajari pola pengetikan yang diskriminatif - sekitar **24× lipat** lebih baik dari tebakan acak.

**Mengapa tidak >90%?** Beberapa faktor yang membatasi akurasi:

1. **Jumlah kelas tinggi (51)**: Multi-class classification dengan banyak kelas secara inheren lebih sulit dibandingkan klasifikasi biner.
2. **Fitur terbatas (3 per timestep)**: Hanya menggunakan dwell time, flight UD, dan flight DD, tanpa fitur turunan tambahan (statistik sesi, pressure, dll.).
3. **Sequence pendek (11 timestep)**: Kata sandi `.tie5Roanl` hanya menghasilkan 11 transisi - lebih pendek dari threshold optimal untuk LSTM.
4. **Intra-user variability**: Pola pengetikan satu pengguna bervariasi antar sesi dan repetisi.

**Kekuatan model**:
- Top-5 accuracy **81.75%** menunjukkan bahwa model hampir selalu menempatkan subjek yang benar di antara 5 prediksi teratas.
- Precision (53.54%) > Accuracy (47.40%) menunjukkan bahwa ketika model yakin, prediksinya cenderung benar.

### 4A.1.3 Equal Error Rate (EER)

| Metrik Biometrik | Nilai |
|-----------------|:-----:|
| **EER** | **10.0%** |

EER 10% menunjukkan bahwa pada threshold optimal, sistem menolak 10% pengguna sah (FRR) dan menerima 10% impostor (FAR). Ini merupakan performa yang memadai untuk sistem berbasis kata sandi tunggal.

### 4A.1.4 Confusion Matrix

Confusion matrix 51×51 menunjukkan pola diagonal yang jelas - mayoritas prediksi benar terkonsentrasi pada diagonal utama. Beberapa pola kesalahan yang teramati:

- **Cluster confusion**: Beberapa pasangan subjek saling tertukar karena memiliki pola pengetikan serupa.
- **Subject dominance**: Beberapa subjek (misalnya s012 dengan 38 prediksi benar dari 39 sampel) memiliki pola yang sangat distinktif.
- **Underrepresented subjects**: Subjek dengan sampel lebih sedikit setelah outlier filtering cenderung memiliki recall lebih rendah.

---

## 4A.2 Hasil Centralized Differential Privacy (Level 2)

### 4A.2.1 Konfigurasi DP-SGD

| Parameter | Nilai |
|-----------|:-----:|
| **Framework** | Opacus |
| **Noise multiplier** ($\sigma$) | 1.1 |
| **Clipping norm** ($C$) | 1.0 |
| **Target delta** ($\delta$) | $10^{-5}$ |
| **Achieved epsilon** ($\epsilon$) | **0.77** |

### 4A.2.2 Performa Model DP

| Metrik | Nilai DP | Nilai Baseline | Perubahan |
|--------|:--------:|:--------------:|:---------:|
| **Accuracy** | **2.56%** | 47.40% | ↓ 44.84 pp |
| **Precision** | 0.07% | 53.54% | ↓ 53.47 pp |
| **Recall** | 2.56% | 47.40% | ↓ 44.84 pp |
| **F1-Score** | 0.13% | 47.48% | ↓ 47.35 pp |
| **Top-3 Accuracy** | 7.02% | 72.53% | ↓ 65.51 pp |
| **Top-5 Accuracy** | 11.55% | 81.75% | ↓ 70.20 pp |
| **Loss** | 3.904 | 1.990 | ↑ 1.914 |

### 4A.2.3 Privacy Budget

| Parameter | Nilai |
|-----------|:-----:|
| **Final epsilon** ($\epsilon$) | **0.77** |
| **Delta** ($\delta$) | $10^{-5}$ |
| **Interpretasi** | Privasi **sangat kuat** ($\epsilon < 1$) |

Epsilon 0.77 berarti: *kemungkinan penyerang membedakan dua dataset tetangga dari output model dibatasi hingga faktor $e^{0.77} \approx 2.16\times$*. Ini merupakan tingkat privasi yang sangat ketat menurut standar literatur DP.

### 4A.2.4 Analisis Trade-off Privasi-Utilitas

**Penurunan drastis** akurasi dari 47.40% ke 2.56% (≈ random guess) disebabkan oleh:

1. **Noise multiplier tinggi ($\sigma = 1.1$)**: Gaussian noise yang ditambahkan ke gradient sangat besar relatif terhadap sinyal gradient yang tipis pada data keystroke.

2. **Clipping norm ketat ($C = 1.0$)**: Per-sample gradient di-clip ke norma maksimum 1.0, memangkas informasi diskriminatif yang besar.

3. **Signal-to-noise ratio rendah**: Fitur keystroke memiliki variasi kecil antar-pengguna (dalam satuan detik/milidetik). Setelah clipping dan noising, sinyal yang membedakan pengguna tenggelam dalam noise.

4. **Jumlah kelas tinggi**: Dengan 51 kelas, model membutuhkan gradient yang sangat presisi untuk membedakan batas keputusan antar kelas. Noise DP menghancurkan presisi ini.

**Implikasi penting**: Pada setting $\epsilon < 1$ dengan 51 kelas dan fitur temporal yang halus, DP-SGD secara efektif **menghapus kemampuan diskriminatif** model - tetapi justru inilah yang membuktikan bahwa *privasi data individu benar-benar terlindungi*.

### 4A.2.5 Perbandingan Visual

Kurva loss pelatihan menunjukkan:
- **Baseline**: Konvergen stabil dari loss ~3.9 ke ~1.2 dalam 30 epoch.
- **DP**: Loss tetap plateau di ~3.9 sepanjang training, menunjukkan model gagal belajar di bawah noise DP.

---

## 4A.3 Rangkuman Bab 4A

| Aspek | Baseline | Centralized DP |
|-------|:--------:|:--------------:|
| **Accuracy** | 47.40% | 2.56% |
| **EER** | 10.0% | 18.0% |
| **Privacy ($\epsilon$)** | $\infty$ (tanpa proteksi) | **0.77** (sangat kuat) |
| **Kerentanan MIA (AUC)** | 0.5182 | 0.4986 |
| **Status** | Utility tinggi, privasi nol | Privasi maksimal, utility minimal |

**Temuan kunci**: Terdapat trade-off fundamental antara privasi dan utilitas. Pada skala fitur keystroke yang sangat halus, penerapan DP ketat ($\epsilon < 1$) menyebabkan degradasi utilitas yang parah. Ini menjadi motivasi untuk mengeksplorasi pendekatan alternatif - Federated Learning - yang memberikan bentuk perlindungan privasi berbeda tanpa menambahkan noise langsung ke gradient.
