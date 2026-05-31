# Draft Laporan Penelitian Keystroke Dynamics

Dokumen ini merupakan draft isi laporan yang lebih dekat ke bentuk final `.docx`. Fokus utama draft ini adalah membantu penyusunan narasi laporan, memperjelas isi Bab 3 dan Bab 4, serta menyiapkan template tabel dan daftar gambar untuk pengisian hasil eksperimen.

## Bagian Awal
### Judul Penelitian
Sistem Identifikasi Pengguna Berbasis Keystroke Dynamics Menggunakan LSTM dengan Differential Privacy dan Federated Learning

### Abstrak Draft
Penelitian ini membangun sistem identifikasi pengguna berbasis keystroke dynamics sebagai bentuk behavioral biometrics dengan memanfaatkan model LSTM untuk mempelajari pola temporal pengetikan. Eksperimen disusun bertahap dari baseline machine learning, Differential Privacy, Federated Learning, kombinasi FL+DP, hingga evaluasi serangan privasi seperti Membership Inference Attack. Hasil penelitian diharapkan menunjukkan bahwa perlindungan privasi dapat ditingkatkan melalui DP dan FL, namun tetap harus dianalisis terhadap dampaknya pada utility model, stabilitas pelatihan, serta efisiensi komunikasi. Draft ini menempatkan privacy-utility trade-off sebagai fokus utama pembahasan.

### Kata Kunci
Keystroke dynamics, behavioral biometrics, LSTM, Differential Privacy, Federated Learning, Membership Inference Attack, privacy-utility trade-off.

## Bab 1 - Pendahuluan
### 1.1 Latar Belakang
Keystroke dynamics adalah salah satu bentuk behavioral biometrics yang memanfaatkan pola pengetikan pengguna sebagai karakteristik identifikasi. Berbeda dari biometrik fisiologis, keystroke dynamics menangkap perilaku yang muncul selama interaksi pengguna dengan perangkat, sehingga relevan untuk skenario autentikasi dan identifikasi pengguna. Data pengetikan mengandung sinyal temporal yang kaya, seperti dwell time, flight time, typing speed, inter-key latency, dan typing rhythm, yang dapat dipelajari oleh model sekuensial seperti LSTM.

Di sisi lain, data perilaku pengetikan juga membawa risiko privasi yang signifikan. Pola tersebut dapat bertindak sebagai fingerprint perilaku yang mengungkap identitas, kebiasaan, dan preferensi pengguna. Dalam konteks pembelajaran mesin terdistribusi, risiko kebocoran tidak hanya muncul dari data mentah, tetapi juga dari gradien, parameter model, dan hasil agregasi. Oleh karena itu, penelitian ini memerlukan pendekatan yang tidak hanya mengejar akurasi, tetapi juga perlindungan privasi melalui Differential Privacy dan Federated Learning.

### 1.2 Rumusan Masalah
Rumusan masalah dalam penelitian ini mencakup empat pertanyaan utama. Pertama, bagaimana membangun model LSTM yang mampu mengidentifikasi pengguna berdasarkan pola keystroke dynamics. Kedua, bagaimana Differential Privacy memengaruhi performa model dan besarnya privacy budget yang dibutuhkan. Ketiga, bagaimana Federated Learning dapat digunakan untuk melatih model tanpa memindahkan data mentah ke server pusat. Keempat, bagaimana kombinasi FL dan DP memengaruhi utility, privacy, dan risiko serangan privasi.

### 1.3 Tujuan Penelitian
Penelitian ini bertujuan untuk merancang dan mengevaluasi sistem identifikasi pengguna berbasis keystroke dynamics dengan tahapan eksperimen yang sistematis. Tujuan teknisnya adalah membangun baseline LSTM, menerapkan DP-SGD, mensimulasikan Federated Learning dengan FedAvg, menguji kombinasi FL+DP, dan mengukur potensi kebocoran menggunakan Membership Inference Attack serta analisis leakage lanjutan. Dari sisi analitis, penelitian ini bertujuan memperlihatkan trade-off antara privacy dan utility dalam skenario behavioral biometrics.

### 1.4 Batasan Penelitian
Penelitian ini dibatasi pada dataset Keystroke Dynamics Benchmark dan tidak mencakup deployment pada sistem produksi nyata. Fokus evaluasi berada pada perbandingan baseline, DP, FL, FL+DP, non-IID setting, ablation study, dan analisis serangan privasi. Model utama yang digunakan adalah LSTM berbasis PyTorch, sedangkan integrasi privasi dan federasi dilakukan dengan Opacus dan Flower.

### 1.5 Kontribusi Penelitian
Kontribusi utama penelitian ini adalah menyediakan pipeline eksperimen yang berlapis dan dapat direplikasi untuk mengevaluasi perilaku pengetikan sebagai biometrik. Selain itu, penelitian ini menyajikan perbandingan eksplisit antara utility model, privacy budget, efisiensi komunikasi, dan ketahanan terhadap serangan privasi pada beberapa skenario pelatihan.

## Bab 2 - Tinjauan Pustaka
### 2.1 Keystroke Dynamics sebagai Behavioral Biometrics
Keystroke dynamics memanfaatkan karakteristik temporal pengetikan untuk membedakan pengguna. Fitur seperti key press timing, key release timing, dwell time, dan flight time membentuk pola yang sering kali konsisten per individu. Dengan mengubah event pengetikan menjadi sequence terstruktur, pola ini dapat digunakan untuk klasifikasi pengguna.

### 2.2 LSTM untuk Pemodelan Urutan Temporal
LSTM dipilih karena kemampuannya menangkap dependency temporal pada sequence yang panjang dan tidak seragam. Dalam konteks keystroke dynamics, urutan event pengetikan penting karena identitas perilaku tidak hanya tercermin pada nilai fitur statis, tetapi juga pada transisi antar event.

### 2.3 Differential Privacy
Differential Privacy digunakan untuk membatasi kontribusi setiap sampel terhadap model. Mekanisme DP-SGD menerapkan clipping pada gradien individual dan menambahkan noise Gaussian untuk menekan risiko inferensi terhadap data asli. Parameter penting yang dianalisis adalah epsilon, noise multiplier, clipping norm, dan target delta.

### 2.4 Federated Learning
Federated Learning memungkinkan pelatihan model secara terdistribusi tanpa memindahkan data mentah ke server. Dengan FedAvg, parameter dari tiap client di-aggregate menjadi model global. Pendekatan ini penting untuk skenario yang sensitif terhadap privasi, tetapi tetap menyisakan risiko leakage dari parameter atau gradien.

### 2.5 Privacy Attacks
Membership Inference Attack digunakan untuk menilai apakah suatu sampel termasuk data training. Selain itu, keystroke fingerprinting, user deanonymization, gradient leakage, dan model inversion dibahas sebagai ancaman yang relevan pada sistem berbasis behavioral biometrics.

## Bab 3 - Metodologi Penelitian
### 3.1 Desain Penelitian
Penelitian dirancang sebagai eksperimen komparatif bertahap. Tahap awal membangun baseline LSTM sebagai pembanding utama. Tahap berikutnya menambahkan Differential Privacy, kemudian Federated Learning, kemudian kombinasi FL+DP. Setelah itu dilakukan evaluasi serangan privasi, non-IID simulation, ablation study, dan analisis explainability. Seluruh skenario menggunakan pipeline preprocessing yang sama agar perbandingan valid.

### 3.2 Dataset
Dataset yang digunakan adalah Keystroke Dynamics Benchmark Dataset. Pada tahap ini dilakukan identifikasi struktur file, format kolom, jumlah pengguna, jumlah sesi, dan karakteristik data mentah. Struktur dataset dibaca untuk mengetahui bagaimana event pengetikan direpresentasikan dan bagaimana sequence dapat dibangun dari log tersebut.

### 3.3 Preprocessing Data
Preprocessing dilakukan untuk menyiapkan sequence yang layak dipakai oleh LSTM. Tahap ini dimulai dengan missing value handling agar data tidak kehilangan informasi penting akibat kolom kosong. Selanjutnya dilakukan outlier filtering untuk mengurangi nilai ekstrem yang tidak realistis atau berpotensi mengganggu model. Setelah itu data disegmentasi berdasarkan sesi agar sequence tidak melintasi batas interaksi yang berbeda.

Sequence kemudian disusun secara temporal untuk mempertahankan urutan event pengetikan. Fitur numerik dinormalisasi agar skala antar fitur setara. Setelah itu dilakukan sequence windowing untuk menyamakan panjang sequence, lalu padding atau truncation agar input model seragam. Tahap akhir adalah train-validation-test split yang didesain untuk meminimalkan leakage antar subset.

### 3.4 Perancangan Fitur
Fitur yang dipakai mencerminkan perilaku pengetikan pengguna, meliputi dwell time, flight time, typing speed, inter-key latency, dan typing rhythm. Fitur-fitur ini dirancang dalam bentuk multivariat sequence sehingga model tidak hanya mempelajari nilai statis, tetapi juga perubahan temporal antar event.

### 3.5 Arsitektur Baseline LSTM
Baseline menggunakan LSTM sebagai encoder temporal. Input sequence diproses melalui satu atau beberapa layer LSTM, kemudian hasil representasi akhir diteruskan ke fully connected layer untuk menghasilkan klasifikasi pengguna. Dalam skenario identifikasi, output berupa multiclass classification, sedangkan untuk skenario autentikasi output dapat dipandang sebagai klasifikasi biner.

### 3.6 Differential Privacy Implementation
DP diintegrasikan menggunakan Opacus dengan DP-SGD. Mekanisme ini meliputi clipping pada gradien per sampel, injeksi noise Gaussian, dan pemantauan privacy budget menggunakan privacy accountant. Konfigurasi DP dievaluasi untuk melihat keseimbangan antara proteksi privasi dan penurunan utility.

### 3.7 Federated Learning Implementation
FedAvg digunakan untuk mensimulasikan pelatihan terdistribusi. Dataset dibagi menjadi beberapa client, lalu masing-masing client melatih model lokal pada data miliknya. Server mengagregasi bobot model dari semua client untuk membentuk model global. Evaluasi mencakup local performance, global performance, communication rounds, dan training latency.

### 3.8 Federated Learning + Differential Privacy
Pada skenario FL+DP, DP-SGD diterapkan di sisi client sebelum parameter di-aggregate oleh server. Dengan pendekatan ini, data lokal tetap berada di client, sementara kontribusi tiap client terhadap model juga dibatasi oleh mekanisme privasi. Skenario ini menjadi kandidat utama untuk menilai apakah privasi dapat ditingkatkan tanpa penurunan utility yang berlebihan.

### 3.9 Privacy Attack Evaluation
Membership Inference Attack digunakan untuk mengukur apakah sampel tertentu dipakai dalam training. Evaluasi dilakukan pada baseline, DP, FL, dan FL+DP. Jika memungkinkan, analisis diperluas ke fingerprinting, deanonymization, gradient leakage, dan model inversion untuk menggambarkan risiko kebocoran secara lebih komprehensif.

### 3.10 Non-IID Simulation
Eksperimen non-IID dirancang dengan memecah data client berdasarkan typing speed, keyboard layout, device type, typing habits, dan session duration. Tujuannya adalah melihat dampak heterogenitas pada stabilitas konvergensi, fairness antar client, dan efisiensi komunikasi.

### 3.11 Ablation Study
Ablation study memeriksa pengaruh parameter-parameter utama seperti epsilon, noise multiplier, clipping norm, sequence length, hidden units, jumlah client, local epoch, communication rounds, dan learning rate. Hasil ablation digunakan untuk mengidentifikasi faktor yang paling memengaruhi utility, privacy, dan stabilitas pelatihan.

### 3.12 Explainability dan Threat Model
Explainability dilakukan untuk melihat fitur apa yang paling berpengaruh terhadap keputusan model. SHAP atau attention visualization dapat dipakai untuk menginterpretasikan kontribusi dwell time, flight time, typing speed, inter-key latency, dan typing rhythm. Threat model disusun untuk empat aktor utama: honest-but-curious server, malicious client, external attacker, dan insider threat.

## Bab 4 - Hasil dan Pembahasan
### 4.1 Hasil Preprocessing
Bab ini diawali dengan penjelasan kondisi data sebelum dan sesudah preprocessing. Statistik data, contoh sequence, dan distribusi fitur ditampilkan untuk menunjukkan bahwa pipeline preprocessing berhasil menghasilkan input yang sesuai untuk model. Pada bagian ini juga dijelaskan keputusan teknis yang diambil saat menangani missing value, outlier, segmentasi sesi, dan padding sequence.

### 4.2 Hasil Baseline LSTM
Hasil baseline digunakan sebagai referensi utama. Metrik yang dibahas meliputi accuracy, F1-score, FAR, FRR, EER, dan confusion matrix. Pembahasan tidak hanya berfokus pada angka, tetapi juga pada pola kesalahan klasifikasi dan bagaimana perilaku pengetikan tertentu cenderung lebih mudah atau lebih sulit dikenali oleh model.

### 4.3 Hasil Differential Privacy
Bagian ini membahas perubahan utility setelah DP diterapkan. Selain metrik klasifikasi, nilai epsilon privacy budget dilaporkan untuk menilai seberapa kuat proteksi privasi yang diperoleh. Pembahasan juga menyoroti perubahan convergence dan stability sebagai konsekuensi dari clipping dan penambahan noise.

### 4.4 Hasil Federated Learning
Hasil FL dibahas dari sisi local versus global performance, communication efficiency, training latency, dan convergence stability. Analisis ini penting untuk menunjukkan apakah pelatihan terdistribusi memberikan utility yang kompetitif sekaligus mengurangi kebutuhan sentralisasi data.

### 4.5 Hasil FL + DP
Pada skenario kombinasi, hasil dibandingkan langsung dengan baseline, DP, dan FL. Fokus pembahasan adalah utility loss, privacy gain, serta apakah kombinasi ini memberikan trade-off yang lebih seimbang dibandingkan metode tunggal.

### 4.6 Hasil Privacy Attack Evaluation
Bab ini menyajikan hasil Membership Inference Attack dan, bila tersedia, hasil analisis leakage lain. Metrik yang dibahas mencakup attack accuracy, precision, recall, dan ROC-AUC. Perbandingan antar skenario menunjukkan metode mana yang paling rentan terhadap kebocoran informasi.

### 4.7 Hasil Non-IID FL
Skenario non-IID digunakan untuk melihat bagaimana heterogenitas client memengaruhi pelatihan federated. Pembahasan mencakup global accuracy, client drift, fairness antar client, dan stabilitas konvergensi. Hasil ini juga dikaitkan dengan efek DP pada lingkungan data yang tidak homogen.

### 4.8 Hasil Ablation Study
Bagian ini merangkum hubungan antara parameter dan performa. Visualisasi accuracy versus epsilon, EER versus epsilon, FAR versus epsilon, FRR versus epsilon, communication efficiency versus performance, dan MIA attack accuracy versus epsilon digunakan untuk menjelaskan trade-off utama yang ditemukan selama eksperimen.

### 4.9 Hasil Explainability dan Threat Model
Fitur yang paling memengaruhi keputusan model diinterpretasikan menggunakan metode explainability. Hasil interpretasi kemudian dipadukan dengan threat model untuk menggambarkan risiko privasi pada tiap skenario. Pada bagian ini dijelaskan bahwa baseline cenderung paling rentan, sementara DP dan FL memberikan lapisan perlindungan tambahan dengan tingkat efektivitas yang berbeda.

## Bab 5 - Kesimpulan dan Saran
### 5.1 Kesimpulan
Penelitian ini menunjukkan bahwa keystroke dynamics dapat digunakan sebagai dasar identifikasi pengguna berbasis behavioral biometrics. Namun, model yang baik secara akurasi belum tentu aman dari serangan privasi. Differential Privacy dan Federated Learning sama-sama membantu menekan risiko kebocoran, tetapi masing-masing membawa trade-off pada utility, komunikasi, dan stabilitas.

### 5.2 Keterbatasan
Keterbatasan penelitian mencakup ketergantungan pada dataset yang tersedia, keterbatasan komputasi, dan ruang lingkup serangan privasi yang diuji. Selain itu, eksperimen yang lebih luas pada model dan dataset yang berbeda masih diperlukan untuk memperkuat generalisasi temuan.

### 5.3 Saran
Penelitian lanjutan dapat mengeksplorasi dataset yang lebih besar, model sekuensial lain, metode defense tambahan, dan skenario autentikasi yang lebih mendekati aplikasi dunia nyata.

## Template Tabel Hasil
### Tabel 4.1 - Statistik Dataset dan Preprocessing
| Komponen | Nilai | Keterangan |
|---|---:|---|
| Jumlah file mentah |  |  |
| Jumlah user |  |  |
| Jumlah sesi |  |  |
| Missing value awal |  |  |
| Outlier terfilter |  |  |
| Sequence akhir |  |  |

### Tabel 4.2 - Hasil Baseline dan Varian Privasi
| Metode | Accuracy | F1-score | FAR | FRR | EER | Epsilon | Catatan |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline |  |  |  |  |  |  |  |
| DP |  |  |  |  |  |  |  |
| FL |  |  |  |  |  |  |  |
| FL+DP |  |  |  |  |  |  |  |

### Tabel 4.3 - Hasil Federated Learning
| Metode | Client | Rounds | Local Accuracy | Global Accuracy | Latency | Communication Efficiency | Catatan |
|---|---:|---:|---:|---:|---:|---:|---|
| FL IID |  |  |  |  |  |  |  |
| FL non-IID |  |  |  |  |  |  |  |
| FL+DP IID |  |  |  |  |  |  |  |
| FL+DP non-IID |  |  |  |  |  |  |  |

### Tabel 4.4 - Hasil Membership Inference Attack
| Metode | Attack Accuracy | Precision | Recall | ROC-AUC | Leakage Note |
|---|---:|---:|---:|---:|---|
| Baseline |  |  |  |  |  |
| DP |  |  |  |  |  |
| FL |  |  |  |  |  |
| FL+DP |  |  |  |  |  |

### Tabel 4.5 - Ablation Study
| Parameter | Nilai | Accuracy | EER | FAR | FRR | Attack Accuracy | Catatan |
|---|---|---:|---:|---:|---:|---:|---|
| Epsilon |  |  |  |  |  |  |  |
| Noise multiplier |  |  |  |  |  |  |  |
| Clipping norm |  |  |  |  |  |  |  |
| Sequence length |  |  |  |  |  |  |  |
| Hidden units |  |  |  |  |  |  |  |

## Template Daftar Gambar
### Gambar 4.1 - Contoh Struktur Data Mentah
- Lokasi penyimpanan: hasil audit dataset.
- Isi: preview kolom dan baris awal.

### Gambar 4.2 - Distribusi Fitur Keystroke
- Lokasi penyimpanan: analisis preprocessing.
- Isi: histogram atau boxplot dwell time, flight time, typing speed, inter-key latency.

### Gambar 4.3 - Ilustrasi Sequence Sebelum dan Sesudah Padding
- Lokasi penyimpanan: preprocessing sequence.
- Isi: visualisasi panjang sequence awal dan hasil padding/truncation.

### Gambar 4.4 - Learning Curve Baseline
- Lokasi penyimpanan: hasil training baseline.
- Isi: loss dan accuracy per epoch.

### Gambar 4.5 - Confusion Matrix Baseline
- Lokasi penyimpanan: evaluasi baseline.
- Isi: heatmap prediksi antar kelas pengguna.

### Gambar 4.6 - Perbandingan Metrik Baseline, DP, FL, dan FL+DP
- Lokasi penyimpanan: analisis komparatif.
- Isi: bar chart atau radar chart untuk accuracy, FAR, FRR, EER, F1-score.

### Gambar 4.7 - Privacy Budget Epsilon
- Lokasi penyimpanan: hasil DP.
- Isi: grafik epsilon terhadap epoch atau konfigurasi.

### Gambar 4.8 - Convergence Federated Learning
- Lokasi penyimpanan: hasil FL.
- Isi: loss atau accuracy per communication round.

### Gambar 4.9 - ROC Curve MIA
- Lokasi penyimpanan: evaluasi serangan privasi.
- Isi: kurva ROC untuk baseline, DP, FL, dan FL+DP.

### Gambar 4.10 - Ablation Study Plots
- Lokasi penyimpanan: hasil ablation.
- Isi: accuracy versus epsilon, EER versus epsilon, FAR versus epsilon, FRR versus epsilon.

### Gambar 4.11 - Explainability Plot
- Lokasi penyimpanan: hasil interpretabilitas model.
- Isi: SHAP summary atau attention heatmap.

## Catatan Penyusunan ke .docx
- Gunakan heading yang konsisten agar daftar isi otomatis terbentuk.
- Tempelkan tabel dan gambar pada bab yang sesuai.
- Ganti placeholder angka setelah eksperimen selesai.
- Pertahankan gaya bahasa formal, ringkas, dan analitis.

## Pemetaan ke Artefak Implementasi
- Notebook menjadi sumber angka dan visualisasi.
- File laporan ini menjadi kerangka penulisan naratif.
- Tabel dan gambar diisi dari output eksperimen terakhir.