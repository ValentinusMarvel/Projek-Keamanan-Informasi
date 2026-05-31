# Bab 2: Tinjauan Pustaka

## 2.1 Keystroke Dynamics sebagai Behavioral Biometrics

### 2.1.1 Definisi dan Konsep Dasar

Keystroke dynamics adalah teknik biometrik perilaku yang mengidentifikasi pengguna berdasarkan pola temporal pengetikan. Tidak seperti biometrik fisiologis (sidik jari, iris), keystroke dynamics memanfaatkan karakteristik *perilaku* yang bersifat:

- **Non-intrusive**: Pengumpulan data tidak memerlukan tindakan eksplisit dari pengguna selain mengetik secara normal.
- **Continuous**: Verifikasi dapat dilakukan secara berkelanjutan selama sesi pengetikan.
- **Software-based**: Tidak memerlukan perangkat keras khusus - cukup keyboard standar dan perekam event.

### 2.1.2 Fitur Temporal Keystroke

Fitur yang diekstraksi dari keystroke dynamics meliputi:

| Fitur | Notasi | Definisi |
|-------|--------|----------|
| **Dwell Time (Hold Time)** | $H_i$ | Durasi penekanan tombol ke-$i$, dihitung dari waktu *key press* hingga *key release*. |
| **Flight Time (Up-Down)** | $UD_{i,i+1}$ | Jeda antara *release* tombol ke-$i$ dan *press* tombol ke-$(i+1)$. Dapat bernilai negatif jika overlap. |
| **Flight Time (Down-Down)** | $DD_{i,i+1}$ | Jeda antara *press* tombol ke-$i$ dan *press* tombol ke-$(i+1)$. Selalu positif. |

Hubungan matematis ketiga fitur:

$$DD_{i,i+1} = H_i + UD_{i,i+1}$$

Dataset yang digunakan dalam penelitian ini memvalidasi relasi ini dengan toleransi numerik $< 10^{-15}$ (floating-point precision), mengonfirmasi konsistensi internal data.

### 2.1.3 Aplikasi dan Penelitian Terkait

Penelitian awal oleh **Gaines et al. (1980)** menunjukkan bahwa individu memiliki ritme pengetikan yang dapat dibedakan secara statistik. **Killourhy dan Maxion (2009)** mengembangkan benchmark dataset DSL-StrongPasswordData yang menjadi standar evaluasi di bidang ini, mengukur kemampuan berbagai algoritma (Euclidean, Manhattan, Mahalanobis, Neural Network) dalam membedakan 51 pengguna berdasarkan pengetikan kata sandi `.tie5Roanl`.

Pendekatan modern menggunakan deep learning - khususnya arsitektur berbasis recurrent neural network - menunjukkan peningkatan signifikan dibandingkan metode statistik klasik, karena kemampuannya menangkap dependensi temporal kompleks dalam urutan penekanan tombol.

## 2.2 Long Short-Term Memory (LSTM)

### 2.2.1 Arsitektur dan Mekanisme

LSTM (Hochreiter & Schmidhuber, 1997) adalah varian Recurrent Neural Network (RNN) yang dirancang untuk mengatasi masalah *vanishing gradient* pada sekuens panjang. LSTM menggunakan mekanisme **gating** untuk mengontrol aliran informasi:

1. **Forget Gate** ($f_t$): Menentukan informasi mana dari *cell state* sebelumnya yang harus dibuang.

   $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

2. **Input Gate** ($i_t$): Menentukan informasi baru mana yang disimpan ke *cell state*.

   $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

3. **Output Gate** ($o_t$): Menentukan output berdasarkan *cell state* yang diperbarui.

   $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

4. **Cell State Update**:

   $$C_t = f_t \odot C_{t-1} + i_t \odot \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

5. **Hidden State Output**:

   $$h_t = o_t \odot \tanh(C_t)$$

### 2.2.2 Kesesuaian dengan Keystroke Dynamics

LSTM sangat cocok untuk keystroke dynamics karena:

- **Dependensi temporal**: Pola pengetikan bersifat sekuensial - ritme tombol ke-$i$ dipengaruhi oleh tombol-tombol sebelumnya.
- **Variabel-length context**: LSTM dapat menangkap konteks dari awal hingga akhir urutan kata sandi.
- **Multi-variate input**: Setiap timestep memiliki beberapa fitur (dwell, flight UD, flight DD) yang diproses secara simultan.

## 2.3 Differential Privacy (DP)

### 2.3.1 Definisi Formal

Differential Privacy (Dwork et al., 2006) memberikan jaminan privasi matematis yang ketat. Sebuah mekanisme acak $\mathcal{M}$ memenuhi $(\epsilon, \delta)$-differential privacy jika untuk semua dataset tetangga $D$ dan $D'$ (berbeda satu record), dan untuk semua himpunan output $S$:

$$Pr[\mathcal{M}(D) \in S] \leq e^{\epsilon} \cdot Pr[\mathcal{M}(D') \in S] + \delta$$

Parameter kunci:
- **$\epsilon$ (epsilon)**: *Privacy budget* - semakin kecil semakin privat. $\epsilon < 1$ dianggap privasi kuat.
- **$\delta$ (delta)**: Probabilitas kegagalan jaminan privasi. Umumnya diset $\delta < \frac{1}{N}$ (N = jumlah sampel).

### 2.3.2 DP-SGD (Differentially Private Stochastic Gradient Descent)

Abadi et al. (2016) memperkenalkan DP-SGD yang memodifikasi SGD standar dengan dua mekanisme:

1. **Per-sample gradient clipping**: Membatasi norma gradient setiap sampel ke nilai maksimum $C$ (*clipping norm*), mencegah satu sampel mendominasi pembaruan model.

   $$\bar{g}_i = g_i \cdot \min\left(1, \frac{C}{\|g_i\|_2}\right)$$

2. **Gaussian noise addition**: Menambahkan noise Gaussian dengan standar deviasi $\sigma \cdot C$ ke rata-rata gradient batch:

   $$\tilde{g} = \frac{1}{B}\left(\sum_{i=1}^{B} \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 I)\right)$$

### 2.3.3 Privacy Accounting

**Rényi Differential Privacy (RDP)** accountant digunakan untuk melacak akumulasi privacy budget selama training. Setiap iterasi "menghabiskan" sebagian budget, dan total epsilon dihitung melalui komposisi:

$$\epsilon_{total} = f(\sigma, B, N, T)$$

di mana $\sigma$ = noise multiplier, $B$ = batch size, $N$ = dataset size, $T$ = total iterasi.

### 2.3.4 Implementasi dengan Opacus

Opacus (Meta, 2021) adalah pustaka PyTorch yang mengotomatiskan konversi model standar ke versi DP-compatible:

- Wrapping optimizer dengan `DPOptimizer` untuk per-sample gradient clipping dan noising.
- Konversi layer yang tidak kompatibel (e.g., `BatchNorm` → `GroupNorm`).
- Tracking otomatis privacy budget melalui `PrivacyEngine`.

## 2.4 Federated Learning (FL)

### 2.4.1 Paradigma Pembelajaran Terdistribusi

Federated Learning (McMahan et al., 2017) memungkinkan pelatihan model machine learning secara kolaboratif tanpa mengumpulkan data mentah ke server sentral. Prinsip utamanya:

- **Data tetap di perangkat lokal**: Setiap klien melatih model pada data pribadinya.
- **Hanya model updates yang dikirim**: Server menerima dan mengagregasi pembaruan bobot, bukan data.
- **Iterasi berulang (rounds)**: Proses pelatihan lokal dan agregasi global diulang hingga konvergensi.

### 2.4.2 FedAvg (Federated Averaging)

Algoritma FedAvg yang digunakan dalam penelitian ini mengikuti prosedur:

1. Server menginisialisasi model global $w_0$.
2. Pada setiap round $t$:
   - Server mengirim $w_t$ ke $K$ klien terpilih.
   - Setiap klien $k$ melatih model lokal selama $E$ epoch lokal: $w_t^k \leftarrow \text{LocalSGD}(w_t, D_k)$.
   - Server mengagregasi: $w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_t^k$, di mana $n_k$ = jumlah sampel klien $k$.

### 2.4.3 Tantangan: Non-IID Data

Dalam skenario realistis, data antar-klien bersifat heterogen (*non-IID*):

- **Label skew**: Beberapa klien hanya memiliki subset label tertentu.
- **Feature skew**: Distribusi fitur berbeda antar-klien (misalnya kecepatan mengetik yang berbeda).
- **Quantity skew**: Jumlah sampel per klien tidak merata.

Non-IID menyebabkan *client drift* - model lokal bergerak ke arah berbeda-beda, memperlambat atau menggagalkan konvergensi global.

### 2.4.4 Implementasi dengan Flower

Flower (Beutel et al., 2020) adalah framework FL yang mendukung simulasi dan produksi. Penelitian ini menggunakan Flower dengan arsitektur gRPC untuk komunikasi server-client.

## 2.5 Kombinasi FL + DP

Penggabungan FL dan DP memberikan perlindungan berlapis:

- **FL** mencegah server mengakses data mentah klien.
- **DP lokal** mencegah server mempelajari informasi individu dari gradient yang dikirim klien.

Dalam skema ini, setiap klien menerapkan DP-SGD secara lokal sebelum mengirim pembaruan model, sehingga gradien yang diterima server sudah di-*noise*. Ini memberikan jaminan formal bahwa server - bahkan jika bersifat *honest-but-curious* - tidak dapat merekonstruksi data individu.

## 2.6 Serangan Privasi pada Model ML

### 2.6.1 Membership Inference Attack (MIA)

MIA (Shokri et al., 2017) bertujuan menentukan apakah sebuah data sampel digunakan dalam pelatihan model. Prosedur:

1. **Shadow model training**: Melatih model "bayangan" pada data dengan distribusi serupa.
2. **Attack classifier**: Melatih classifier biner pada output shadow model untuk membedakan *member* vs *non-member*.
3. **Evaluasi**: Mengukur seberapa baik attack classifier membedakan data pelatihan target model asli.

Metrik utama: **AUC-ROC** - AUC mendekati 0.5 berarti model aman (penyerang tidak lebih baik dari tebakan acak).

### 2.6.2 Gradient Leakage Attack

Serangan rekonstruksi gradient (Zhu et al., 2019) berusaha memulihkan data pelatihan asli dari gradient yang dibagikan:

1. Inisialisasi data dummy $x^*$ secara acak.
2. Hitung gradient dummy: $g^* = \nabla_w \mathcal{L}(f(x^*), y^*)$.
3. Optimalkan $x^*$ untuk meminimumkan jarak antara gradient dummy dan gradient asli:

   $$x^* = \arg\min_{x^*} \|g^* - g_{real}\|^2$$

Optimasi dilakukan menggunakan LBFGS. Keberhasilan diukur melalui **Cosine Similarity** antara data rekonstruksi dan data asli.

## 2.7 Explainability pada Deep Learning

### 2.7.1 Integrated Gradients

Integrated Gradients (Sundararajan et al., 2017) menghitung kontribusi setiap fitur input terhadap prediksi model:

$$IG_i(x) = (x_i - x'_i) \cdot \int_{\alpha=0}^{1} \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$

di mana $x'$ adalah baseline (biasanya input nol) dan $F$ adalah fungsi model. Metode ini memenuhi dua aksioma penting:

- **Sensitivity**: Jika mengubah fitur $i$ mengubah prediksi, maka atribusi fitur $i$ harus non-nol.
- **Implementation Invariance**: Atribusi tidak bergantung pada detail implementasi internal model.

### 2.7.2 Relevansi untuk Keystroke Dynamics

Explainability pada keystroke dynamics menjawab pertanyaan: *transisi tombol mana* dan *fitur temporal mana* (dwell, flight UD, flight DD) yang paling membedakan antar-pengguna? Informasi ini penting untuk:

- Memahami keputusan model.
- Mengidentifikasi fitur yang rentan terhadap pemalsuan.
- Merancang pertahanan yang lebih terarah.
