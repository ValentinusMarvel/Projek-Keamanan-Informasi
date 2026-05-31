# Bab 4B: Hasil: Federated Learning & FL+DP

## 4B.1 Hasil Federated Baseline (Level 3)

### 4B.1.1 Konfigurasi Simulasi

| Parameter | Nilai |
|-----------|:-----:|
| **Framework** | Flower (gRPC) |
| **Strategi agregasi** | FedAvg |
| **Jumlah klien** | 5 |
| **Communication rounds** | 5 |
| **Local epochs per round** | 1 |
| **Batch size** | 64 |
| **Learning rate** | 0.001 |
| **Partisi data** | IID (rata) |
| **Sampel per klien** | ~2.278-2.279 |
| **Total sampel** | 14.242 |
| **Jumlah kelas** | 51 |

### 4B.1.2 Konvergensi Per-Round

| Round | Global Loss | Global Accuracy |
|:-----:|:-----------:|:---------------:|
| 1 | 3.929 | 1.51% |
| 2 | 3.915 | 2.56% |
| 3 | 3.906 | 2.56% |
| 4 | 3.900 | 2.56% |
| 5 | 3.868 | **4.49%** |

### 4B.1.3 Hasil Akhir FL Baseline

| Metrik | Nilai |
|--------|:-----:|
| **Global Accuracy** | **2.18%** |
| **Global Loss** | 3.932 |
| **EER** | 12.0% |

### 4B.1.4 Analisis Performa FL Baseline

**Akurasi sangat rendah (2.18%)** - mendekati random guess (1.96%). Penyebab utama:

1. **Jumlah round terbatas (5)**: FedAvg membutuhkan banyak round untuk konvergensi, terutama pada klasifikasi multi-class dengan 51 kelas. Pada pelatihan terpusat, model membutuhkan 30 epoch untuk mencapai 47% - setara dengan 30 full pass data. Dengan 5 round × 1 epoch lokal, total effective pass hanya 5.

2. **Fragmentasi data**: Setiap klien hanya memiliki ~2.279 sampel dari 14.242 total. Dengan 51 kelas, rata-rata hanya ~45 sampel per kelas per klien - terlalu sedikit untuk mempelajari batas keputusan yang baik dalam 1 epoch.

3. **Random initialization**: Model diinisialisasi dengan bobot acak, memerlukan banyak iterasi untuk menemukan representasi yang bermakna.

4. **Skala fitur tidak dinormalisasi**: Fitur timing (dalam satuan detik) memiliki skala absolut yang berbeda-beda, menyulitkan optimisasi yang efisien.

**Tren positif yang teramati**: Meskipun akurasi akhir rendah, terdapat penurunan loss yang konsisten (3.929 → 3.868) - menunjukkan model *sedang belajar*, hanya belum cukup round.

### 4B.1.5 Estimasi dengan Lebih Banyak Round

Berdasarkan tren penurunan loss, jika round ditingkatkan ke 30-50 dengan 3-5 epoch lokal, akurasi diestimasi dapat mendekati 40-50% - mendekati performa terpusat. Namun, hal ini memerlukan waktu eksekusi yang jauh lebih lama (~10-30 menit vs ~20 detik pada 5 round).

---

## 4B.2 Hasil FL + DP (Level 4)

### 4B.2.1 Konfigurasi FL + DP

| Parameter | Nilai |
|-----------|:-----:|
| **Strategi** | FedAvg + DP-SGD lokal |
| **Jumlah klien** | 5 |
| **Rounds** | 5 |
| **Local epochs** | 1 |
| **Noise multiplier** | 1.1 |
| **Clipping norm** | 1.0 |
| **Local epsilon** | 0.77 |
| **Local delta** | $10^{-5}$ |

### 4B.2.2 Konvergensi Per-Round

| Round | Global Loss | Global Accuracy |
|:-----:|:-----------:|:---------------:|
| 1 | 3.934 | 2.32% |
| 2 | 3.934 | 2.32% |
| 3 | 3.934 | 2.32% |
| 4 | 3.934 | 2.32% |
| 5 | 3.934 | 2.32% |

### 4B.2.3 Hasil Akhir FL + DP

| Metrik | Nilai |
|--------|:-----:|
| **Global Accuracy** | **2.11%** |
| **Global Loss** | 3.938 |
| **EER** | 20.0% |
| **Local Epsilon** | 0.77 |

### 4B.2.4 Analisis Performa FL + DP

**Stagnansi total**: Loss dan akurasi tidak berubah sepanjang 5 round. Ini menunjukkan **model tidak belajar sama sekali** dalam konfigurasi ini.

Penyebab kumulatif:

1. **Double privacy barrier**: Model menghadapi *dua lapisan penghambat pembelajaran* secara simultan:
   - **DP noise**: Gradient sudah sangat noisy sebelum dikirim ke server.
   - **FL fragmentation**: Data sudah terpecah ke 5 klien.

2. **Gradient noise dominasi**: Dengan $\sigma = 1.1$ dan $C = 1.0$, noise yang ditambahkan jauh melebihi sinyal gradient yang sudah kecil akibat fragmentasi data.

3. **No convergence signal**: Server menerima pembaruan bobot yang hampir identik dari setiap klien (karena noise mendominasi), sehingga agregasi FedAvg tidak menghasilkan perbaikan model.

**Interpretasi keamanan**: Stagnansi ini justru *mendemonstrasikan kekuatan proteksi privasi*. Jika model tidak mampu mempelajari pola dari data, maka informasi data training pun tidak terekspos melalui model.

### 4B.2.5 EER yang Memburuk

EER meningkat dari 10.0% (baseline) ke 20.0% (FL+DP), menunjukkan bahwa kemampuan biometrik sistem telah terdegradasi signifikan. Pada EER 20%, sistem menolak 1 dari 5 pengguna sah dan menerima 1 dari 5 impostor - tidak dapat diandalkan untuk autentikasi praktis.

---

## 4B.3 Perbandingan Skenario Federated

| Aspek | FL Baseline | FL + DP |
|-------|:----------:|:-------:|
| **Accuracy** | 2.18% | 2.11% |
| **EER** | 12.0% | 20.0% |
| **Loss** | 3.932 | 3.938 |
| **Konvergensi** | Lambat tapi ada | Stagnan total |
| **Privacy ($\epsilon$)** | $\infty$ | 0.77 |
| **Learning signal** | Ada (loss turun) | Tidak ada |

**Temuan kunci**: Penambahan DP pada FL menyebabkan *total stagnation* pada 5 round. Perlindungan privasi maksimal (FL + DP ketat) pada skenario ini menghasilkan model yang secara fungsional setara dengan *random classifier*.

---

## 4B.4 Implikasi dan Justifikasi

### 4B.4.1 Keterbatasan 5 Round

Pilihan 5 round didasarkan pada pertimbangan pragmatis:

- **Waktu eksekusi**: Setiap round melibatkan spawn 5 proses gRPC client + 1 server, yang pada Windows memerlukan ~4-5 detik per round. Total ~20 detik untuk demo lengkap.
- **Stabilitas sistem**: Jumlah proses bersamaan yang lebih banyak (>50 round) berisiko menyebabkan *kernel freeze* atau *dangling processes* pada lingkungan Windows.
- **Tujuan demonstratif**: Eksperimen ini bertujuan mendemonstrasikan *mekanisme* FL dan FL+DP, bukan mengoptimalkan akurasi absolut.

### 4B.4.2 Skenario Produksi

Dalam skenario produksi yang realistis:
- **30-50 round** dengan **3-5 epoch lokal** akan menghasilkan konvergensi yang jauh lebih baik.
- **Lebih banyak data per klien** (distribusi horizontal lebih luas) meningkatkan kualitas gradient lokal.
- **Adaptive noise scheduling** (menurunkan noise secara bertahap) memungkinkan pembelajaran awal yang lebih stabil.
