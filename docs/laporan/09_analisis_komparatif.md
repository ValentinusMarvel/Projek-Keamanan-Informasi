# Bab 6: Analisis Komparatif & Privacy-Utility Trade-off

## 6.1 Dashboard Metrik Terpadu

### 6.1.1 Tabel Perbandingan Lengkap

| Konfigurasi | Accuracy (%) | F1-Score (%) | EER (%) | MIA AUC | Gradient Leak (Cos Sim) | Epsilon ($\epsilon$) | Bandwidth (MB) | Latency (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Centralized Baseline LSTM | **63.29** | **62.03** | **16.29** | 0.5182 | N/A | ∞ | N/A | **30.0** |
| Centralized DP LSTM (Opacus) | 2.25 | 0.10 | 34.01 | **0.4986** | 0.0383 🟢 | **0.77** | N/A | 45.0 |
| Federated Baseline FL (Flower) | 2.18 | 0.09 | 33.75 | 0.5003 | −0.0186 | ∞ | 10.94 | 62.5 |
| Joint FL + DP (Flower + Opacus) | 2.11 | 0.09 | 33.91 | 0.5010 | 0.0383 🟢 | **0.77** | 10.94 | 62.5 |
| Non-IID Federated Learning | 1.83 | 0.07 | 34.94 | 0.5005 | N/A | ∞ | 10.94 | 62.5 |
| **Advanced Transfer FL** | **63.29** | **62.03** | **16.29** | 0.5003 | −0.0186 | ∞ | **10.94** | 62.5 |

### 6.1.2 Peringkat Per-Dimensi

**Peringkat Utilitas (Accuracy & F1-Score)**:
1. 🥇 Advanced Transfer FL & Centralized Baseline LSTM - **63.29%**
2. 🥈 Centralized DP - 2.25%
3. 🥉 FL Baseline - 2.18%
4. FL + DP - 2.11%
5. Non-IID FL - 1.83%

**Peringkat Biometrik (EER, semakin rendah semakin baik)**:
1. 🥇 Centralized Baseline & Advanced Transfer FL - **16.29%**
2. 🥈 FL Baseline - 33.75%
3. 🥉 FL + DP - 33.91%
4. Centralized DP - 34.01%
5. Non-IID FL - 34.94%

**Peringkat Privasi (Privacy, semakin rendah epsilon semakin baik)**:
1. 🥇 Centralized DP & FL+DP - **ε = 0.77**
2. 🥉 FL Baseline - Data lokal, tanpa DP formal
3. Non-IID FL - Data lokal, heterogen
4. Transfer FL - Data lokal, tanpa DP
5. Centralized Baseline - **ε = ∞ (tanpa privasi)**

**Peringkat Keamanan Serangan (MIA AUC, semakin dekat 0.50 semakin baik)**:
1. 🥇 Centralized DP - **0.4986** (di bawah random guess)
2. 🥈 FL Baseline & Transfer FL - 0.5003
3. 🥉 FL + DP - 0.5010
4. Centralized Baseline - 0.5182 (sedikit di atas random guess)

---

## 6.2 Analisis Privacy-Utility Trade-off

### 6.2.1 Kurva Trade-off

Berdasarkan data ablation dan konfigurasi utama, kurva privacy-utility dapat dipetakan:

```
Accuracy (%)
      │
  65 ─┤                                          ★ Transfer FL
      │
  54 ─┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ○ Baseline (ε=∞)
  51 ─┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ● ε=5
  46 ─┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ● ε=2 (optimal)
  39 ─┤─ ─ ─ ─ ─ ─ ─ ─ ● ε=1
  28 ─┤─ ─ ─ ─ ● ε=0.5
      │
  12 ─┤─ ● ε=0.1
      │
   2 ─┤● DP (ε=0.77) ─ ─ ─ ▲ FL ─ ─ ▲ FL+DP
      │
      └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──→
      0.1  0.5  1  2  5  10    ∞
                 Epsilon (ε) →
         ←── Lebih Privat ───── Kurang Privat ──→
```

### 6.2.2 Zona Trade-off

Dari kurva di atas, tiga zona dapat diidentifikasi:

#### Zona 1: Privasi Maksimal ($\epsilon \leq 0.5$)
- **Accuracy**: 12-28%
- **MIA AUC**: ≤ 0.50
- **Cocok untuk**: Skenario di mana privasi mutlak diprioritaskan di atas utilitas (data sangat sensitif, regulasi ketat).
- **Trade-off**: Model hampir tidak berguna untuk identifikasi praktis.

#### Zona 2: Keseimbangan Optimal ($1.0 \leq \epsilon \leq 5.0$)
- **Accuracy**: 39-51%
- **MIA AUC**: 0.50-0.51
- **Cocok untuk**: Produksi di mana privasi dan utilitas harus seimbang.
- **Rekomendasi**: **$\epsilon = 2.0$** - akurasi 46% (85% baseline) dengan MIA random guess.

#### Zona 3: Utilitas Maksimal ($\epsilon > 5.0$ atau tanpa DP)
- **Accuracy**: >51% (hingga 63% dengan Transfer FL)
- **MIA AUC**: 0.51-0.52
- **Cocok untuk**: Skenario di mana akurasi biometrik kritis dan privasi dijamin secara infrastruktur (encryption, access control).
- **Catatan**: Transfer FL dalam zona ini memberikan akurasi tertinggi (63%) sambil tetap menjaga data lokal (FL).

### 6.2.3 Pareto Frontier

Konfigurasi yang berada di *Pareto frontier* (tidak ada konfigurasi lain yang lebih baik di semua dimensi sekaligus):

| Konfigurasi | Utility | Privacy | Security | **Status Pareto** |
|-------------|:-------:|:-------:|:--------:|:-----------------:|
| FL + DP | Low | Best | Best | 🟢 Pareto (privacy-focused) |
| DP ε=2.0 | Medium | Good | Good | 🟢 Pareto (balanced) |
| Transfer FL | Best | Medium | Good | 🟢 Pareto (utility-focused) |

Konfigurasi **Non-IID FL** dan **FL Baseline** (tanpa transfer) **tidak berada di Pareto frontier** - mereka memiliki utility rendah tanpa keuntungan privacy formal.

---

## 6.3 Perbandingan Dimensi Keamanan

### 6.3.1 Heatmap Keamanan

| Dimensi | Baseline | DP | FL | FL+DP | Transfer FL |
|---------|:--------:|:--:|:--:|:-----:|:-----------:|
| **Data Locality** | 🔴 Terpusat | 🔴 Terpusat | 🟢 Lokal | 🟢 Lokal | 🟢 Lokal |
| **Gradient Protection** | N/A | 🟢 Noised | 🔴 Plain | 🟢 Noised | 🔴 Plain |
| **MIA Resistance** | 🟡 Marginal | 🟢 Strong | 🟢 Strong | 🟢 Strong | 🟢 Strong |
| **Leakage Resistance** | 🔴 Vulnerable | 🟢 Secured | 🔴 Vulnerable | 🟢 Secured | 🔴 Vulnerable |
| **Formal Privacy** | 🔴 None | 🟢 ε=0.77 | 🔴 None | 🟢 ε=0.77 | 🔴 None |

### 6.3.2 Analisis Trade-off Multi-Dimensi

**1. Utilitas vs Privasi Formal**: Hubungan *monoton negatif* yang jelas - semakin ketat privasi (ε kecil), semakin rendah akurasi. Tidak ada "magic bullet" yang memberikan keduanya secara sempurna.

**2. Utilitas vs Keamanan Serangan**: Hubungan *lemah* - bahkan model dengan akurasi tertinggi (Transfer FL, 63%) memiliki MIA AUC = 0.50, menunjukkan bahwa keamanan terhadap MIA *tidak selalu* harus mengorbankan utilitas.

**3. Data Locality vs Formal Privacy**: Keduanya independen - FL menjaga data lokal tapi tidak memberikan jaminan formal; DP memberikan jaminan formal tapi data bisa tetap terpusat. Kombinasi FL+DP memberikan keduanya.

**4. Transfer Learning sebagai "Free Lunch"**: Transfer FL mencapai akurasi tertinggi (63%) sambil mempertahankan data locality. Biaya tambahannya adalah kebutuhan model pre-trained - yang memerlukan *satu kali* pelatihan terpusat awal.

---

## 6.4 Implikasi Praktis

### 6.4.1 Rekomendasi Per-Skenario

| Skenario Penggunaan | Konfigurasi Direkomendasikan | Alasan |
|----------------------|:----------------------------:|--------|
| **Banking/Financial** | FL + DP (ε=2.0) | Regulasi ketat; privasi prioritas |
| **Enterprise SSO** | Transfer FL | Accuracy tinggi; data tetap lokal |
| **Research/Academic** | Baseline + DP (ε=5.0) | Keseimbangan utility-privacy |
| **IoT/Edge Device** | FL (lightweight) | Resource terbatas; data lokal |
| **Government/Military** | FL + DP (ε=0.5) | Privasi maksimal wajib |

### 6.4.2 Pertimbangan Deployment

1. **Pre-training requirement**: Transfer FL memerlukan checkpoint model yang sudah dilatih. Dalam produksi, ini bisa dilakukan pada data publik/sintetis sebelum deployment.

2. **Communication overhead**: Setiap round FL memerlukan transmisi ~224KB (ukuran model .pt). Pada 5 klien × 50 round = 56 MB total bandwidth.

3. **Scalability**: Arsitektur current dengan gRPC mendukung simulasi lokal. Untuk deployment nyata, diperlukan infrastruktur FL yang lebih robust (Flower production mode, Kubernetes, dll.).
