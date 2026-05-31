# Bab 5: Explainability & Threat Modeling

## 5.1 Explainability dengan Integrated Gradients (Level 10)

### 5.1.1 Konfigurasi

| Parameter | Nilai |
|-----------|-------|
| **Metode** | Integrated Gradients (PyTorch) |
| **Model target** | Baseline LSTM (`baseline_lstm.pt`) |
| **Baseline input** | Tensor nol (zero baseline) |
| **Jumlah interpolation steps** | Default (50) |
| **Kata sandi** | `.tie5Roanl` (10 karakter + Return) |

### 5.1.2 Karakter Kata Sandi dan Mapping

| Timestep | Karakter | Fitur 1: Dwell | Fitur 2: Flight UD | Fitur 3: Flight DD |
|:--------:|:--------:|:--------------:|:-------------------:|:-------------------:|
| 0 | `.` (period) | $H_{period}$ | 0 (initial) | 0 (initial) |
| 1 | `t` | $H_t$ | $UD_{period \rightarrow t}$ | $DD_{period \rightarrow t}$ |
| 2 | `i` | $H_i$ | $UD_{t \rightarrow i}$ | $DD_{t \rightarrow i}$ |
| 3 | `e` | $H_e$ | $UD_{i \rightarrow e}$ | $DD_{i \rightarrow e}$ |
| 4 | `5` | $H_5$ | $UD_{e \rightarrow 5}$ | $DD_{e \rightarrow 5}$ |
| 5 | `Shift.r` | $H_{Shift.r}$ | $UD_{5 \rightarrow Shift.r}$ | $DD_{5 \rightarrow Shift.r}$ |
| 6 | `o` | $H_o$ | $UD_{Shift.r \rightarrow o}$ | $DD_{Shift.r \rightarrow o}$ |
| 7 | `a` | $H_a$ | $UD_{o \rightarrow a}$ | $DD_{o \rightarrow a}$ |
| 8 | `n` | $H_n$ | $UD_{a \rightarrow n}$ | $DD_{a \rightarrow n}$ |
| 9 | `l` | $H_l$ | $UD_{n \rightarrow l}$ | $DD_{n \rightarrow l}$ |
| 10 | `Return` | $H_{Return}$ | $UD_{l \rightarrow Return}$ | $DD_{l \rightarrow Return}$ |

### 5.1.3 Hasil Atribusi (Absolute Values)

Tabel berikut menunjukkan **magnitude atribusi absolut** per karakter per fitur. Nilai lebih tinggi menunjukkan kontribusi lebih besar terhadap keputusan klasifikasi:

| Karakter | Dwell Time | Flight UD | Flight DD | **Total** |
|:--------:|:----------:|:---------:|:---------:|:---------:|
| `.` | **0.1033** | 0.0000 | 0.0000 | 0.1033 |
| `t` | 0.0468 | **0.1061** | **0.2172** | **0.3701** |
| `i` | 0.0066 | 0.0586 | 0.1460 | 0.2112 |
| `e` | 0.0095 | 0.0419 | 0.1137 | 0.1651 |
| `5` | 0.0088 | 0.0616 | 0.1118 | 0.1822 |
| `Shift.r` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `o` | 0.0555 | 0.0000 | 0.0000 | 0.0555 |
| `a` | **0.0891** | 0.0079 | 0.0206 | 0.1176 |
| `n` | 0.0500 | 0.0234 | 0.0526 | 0.1260 |
| `l` | 0.0339 | 0.0694 | 0.0905 | 0.1938 |
| `Return` | 0.0067 | **0.1579** | **0.1867** | **0.3513** |

### 5.1.4 Temuan Explainability

#### Karakter Paling Penting

1. **`t` (Total = 0.3701)** - Karakter paling diskriminatif. Transisi `.` → `t` menghasilkan atribusi flight DD tertinggi (0.2172). Ini konsisten secara logis: transisi dari tombol `.` (pinky/ring finger kanan) ke `t` (index finger kiri) memerlukan perpindahan tangan yang besar dan bervariasi antar-pengguna.

2. **`Return` (Total = 0.3513)** - Penekanan tombol Enter di akhir kata sandi juga sangat diskriminatif. Flight UD (0.1579) dan DD (0.1867) yang tinggi menunjukkan bahwa *cara pengguna mengakhiri pengetikan* merupakan penanda biometrik yang kuat.

3. **`i`, `l` (Total > 0.19)** - Transisi berurutan pada bagian tengah dan akhir kata sandi juga berkontribusi signifikan.

#### Fitur Paling Penting

1. **Flight Time DD (Down-Down)**: Fitur paling informatif secara keseluruhan, dengan atribusi tertinggi pada hampir semua karakter. DD menangkap *ritme keseluruhan* pengetikan karena mengukur interval press-to-press.

2. **Flight Time UD (Up-Down)**: Fitur kedua terpenting, khususnya pada transisi `Return` dan `t`. UD menangkap *kelincahan jari* - jeda antara melepas satu tombol dan menekan tombol berikutnya.

3. **Dwell Time (Hold)**: Paling penting pada karakter awal (`.` = 0.1033, `a` = 0.0891). Dwell time mengindikasikan *kebiasaan tekanan* individual.

#### Anomali: Shift.r

Karakter `Shift.r` memiliki atribusi **nol** pada semua fitur. Ini kemungkinan karena:
- Shift ditekan bersamaan dengan `r`, sehingga timing-nya sangat konsisten antar-pengguna (modifikasi kecil).
- Model menganggap transisi ke/dari Shift tidak informatif dibandingkan transisi antar huruf.

### 5.1.5 Implikasi Praktis

- **Pertahanan biometrik**: Transisi `.` → `t` dan `l` → `Return` adalah *fitur paling sensitif*. Jika penyerang ingin meniru pola ketik, mereka harus terutama mencocokkan timing pada transisi ini.
- **Desain kata sandi**: Kata sandi dengan transisi jari yang lebih bervariasi (cross-hand movements) akan menghasilkan biometrik yang lebih kuat.
- **Feature selection**: Dalam skenario komputasi terbatas, fitur DD dan UD sudah cukup - dwell time memberikan kontribusi relatif kecil.

---

## 5.2 Threat Modeling

### 5.2.1 Profil Aktor Ancaman

#### Threat Actor 1: Honest-but-Curious Server

| Aspek | Detail |
|-------|--------|
| **Identitas** | Server agregasi FL |
| **Motivasi** | Mengumpulkan informasi perilaku pengguna dari gradien |
| **Kapabilitas** | Memiliki akses penuh ke gradien/pembaruan bobot yang dikirim klien |
| **Vektor serangan** | Gradient Matching (LBFGS), analisis statistik gradien |
| **Aset target** | Pola timing keystroke → profil perilaku pengguna |

**Evaluasi risiko per konfigurasi**:

| Konfigurasi | Risiko | Justifikasi |
|-------------|:------:|-------------|
| Baseline | N/A | Tidak ada komponen federated |
| FL | 🟡 MEDIUM | Gradien dapat dianalisis; Cosine Sim = −0.019 |
| FL + DP | 🟢 LOW | Gradien sudah di-noise; Cosine Sim = 0.038 |
| Transfer FL | 🟡 MEDIUM | Gradien tersedia tanpa noise |

#### Threat Actor 2: Malicious Client

| Aspek | Detail |
|-------|--------|
| **Identitas** | Klien FL yang berperilaku jahat |
| **Motivasi** | Mempengaruhi model global atau mencuri data klien lain |
| **Kapabilitas** | Dapat mengirim pembaruan bobot berbahaya (model poisoning) |
| **Vektor serangan** | Gradient poisoning, backdoor attack, free-riding |
| **Aset target** | Integritas model global; data klien lain (melalui model global) |

**Evaluasi risiko**:

| Konfigurasi | Risiko | Justifikasi |
|-------------|:------:|-------------|
| FL | 🟡 MEDIUM | Tidak ada verifikasi pembaruan klien |
| FL + DP | 🟢 LOW | Gradient clipping membatasi dampak poisoning |

**Mitigasi yang diterapkan**: Per-sample gradient clipping ($C = 1.0$) pada DP-SGD secara otomatis membatasi magnitude pembaruan dari setiap klien, mengurangi dampak poisoning.

#### Threat Actor 3: External Attacker

| Aspek | Detail |
|-------|--------|
| **Identitas** | Penyerang eksternal yang mengintersepsi komunikasi |
| **Motivasi** | Mencuri data biometrik dari transmisi jaringan |
| **Kapabilitas** | Penyadapan paket jaringan (man-in-the-middle) |
| **Vektor serangan** | Packet sniffing pada komunikasi gRPC; MIA pada model yang dipublikasikan |
| **Aset target** | Gradien dalam transit; model checkpoint yang tersimpan |

**Evaluasi risiko**:

| Konfigurasi | Risiko | Justifikasi |
|-------------|:------:|-------------|
| Baseline | 🟡 MEDIUM | Model checkpoint rentan terhadap MIA (AUC=0.52) |
| DP | 🟢 LOW | MIA tidak efektif (AUC=0.50) |
| FL | 🟢 LOW | Gradien dalam transit; MIA AUC=0.50 |
| FL + DP | 🟢 VERY LOW | Double protection; MIA AUC=0.50 |

#### Threat Actor 4: Insider / Administrator

| Aspek | Detail |
|-------|--------|
| **Identitas** | Administrator sistem dengan akses ke server |
| **Motivasi** | Memprofiling pengguna berdasarkan model yang tersimpan |
| **Kapabilitas** | Akses offline ke checkpoint model dan data pelatihan |
| **Vektor serangan** | Model inversion; analisis pola bobot; akses langsung ke data |
| **Aset target** | Profil perilaku individual; database biometrik |

**Evaluasi risiko**:

| Konfigurasi | Risiko | Justifikasi |
|-------------|:------:|-------------|
| Baseline | 🔴 HIGH | Data terpusat + model tanpa proteksi |
| DP | 🟡 MEDIUM | Model di-noise tapi data tetap terpusat |
| FL | 🟡 MEDIUM | Data tidak terpusat tapi model global ada |
| FL + DP | 🟢 LOW | Data desentralisasi + model di-noise |

### 5.2.2 Matriks Risiko Terpadu

| Aktor Ancaman | Baseline | DP | FL | FL+DP | Transfer FL |
|---------------|:--------:|:--:|:--:|:-----:|:-----------:|
| Honest-but-Curious Server | N/A | N/A | 🟡 | 🟢 | 🟡 |
| Malicious Client | N/A | N/A | 🟡 | 🟢 | 🟡 |
| External Attacker | 🟡 | 🟢 | 🟢 | 🟢 | 🟢 |
| Insider/Admin | 🔴 | 🟡 | 🟡 | 🟢 | 🟡 |
| **Risiko Keseluruhan** | **🔴 HIGH** | **🟡 MED** | **🟡 MED** | **🟢 LOW** | **🟡 MED** |

### 5.2.3 Analisis Risiko Per-Konfigurasi

#### Baseline: Risiko TINGGI
- Semua data terpusat, tidak ada mekanisme privasi.
- Model menyimpan informasi tentang pola pelatihan (MIA AUC = 0.52).
- Insider dapat mengakses data dan model tanpa hambatan.

#### DP: Risiko SEDANG
- Noise DP melindungi pola individu dalam model (MIA AUC = 0.50).
- **Kelemahan**: Data tetap terpusat - insider masih dapat mengakses data mentah.
- DP hanya melindungi *model*, bukan *data*.

#### FL: Risiko SEDANG
- Data tidak pernah meninggalkan perangkat klien.
- **Kelemahan**: Gradien yang dikirim masih potensial dianalisis (Gradient Leakage).
- Server tidak memiliki data mentah, tapi memiliki informasi gradien setiap klien.

#### FL+DP: Risiko RENDAH
- **Perlindungan berlapis**: Data lokal (FL) + gradien di-noise (DP).
- Server tidak memiliki data mentah DAN gradien sudah dilindungi noise.
- MIA dan Gradient Leakage keduanya tidak efektif.
- **Trade-off**: Utilitas model sangat rendah (2.11%) - mungkin tidak praktis tanpa optimasi lebih lanjut.

---

## 5.3 Rekomendasi Keamanan

| Prioritas | Rekomendasi | Konfigurasi Target |
|:---------:|-------------|:-------------------:|
| **1** | Gunakan FL+DP untuk skenario sensitivitas tinggi | FL+DP |
| **2** | Terapkan Transfer FL + DP untuk keseimbangan utility-privacy | Transfer FL + DP |
| **3** | Gunakan $\epsilon \geq 2.0$ untuk trade-off yang masuk akal | Semua DP |
| **4** | Enkripsi komunikasi gRPC (TLS/mTLS) | Semua FL |
| **5** | Batasi akses admin ke model checkpoint | Semua konfigurasi |
| **6** | Implementasi secure aggregation untuk pencegahan gradient analysis | FL, FL+DP |
