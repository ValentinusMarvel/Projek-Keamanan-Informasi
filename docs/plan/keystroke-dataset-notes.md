# Keystroke Dataset Notes and Evaluation Anchors

Dokumen ini menyimpan konteks dataset benchmark sebagai acuan tetap selama pengembangan, validasi, dan pembahasan hasil.

## Ringkasan Dataset
- Nama: Keystroke Dynamics - Benchmark Data Set.
- Referensi studi: "Comparing Anomaly-Detection Algorithms for Keystroke Dynamics" (DSN-2009).
- Jumlah subjek: 51 typists.
- Tugas: setiap subjek mengetik password `.tie5Roanl` sebanyak 400 kali.
- Total sesi per subjek: 8 sesi.
- Total repetisi per sesi: 50 repetisi.
- Struktur tabel: 34 kolom per baris.

## Struktur Kolom Utama
- `subject`: identitas unik subjek (contoh `s002`, `s057`).
- `sessionIndex`: sesi pengetikan, rentang 1 sampai 8.
- `rep`: urutan repetisi dalam sesi, rentang 1 sampai 50.
- 31 kolom sisanya adalah fitur timing pengetikan.

## Arti Fitur Timing
- `H.key`: hold time dari tombol tertentu (keydown sampai keyup).
- `DD.key1.key2`: keydown-keydown time antar dua tombol.
- `UD.key1.key2`: keyup-keydown time antar dua tombol.

## Hubungan Penting Antar Fitur
- Nilai `UD` dapat bernilai negatif.
- Secara konsep, `H + UD = DD` untuk pasangan key yang sesuai.

## Implikasi untuk Pipeline Preprocessing
1. Validasi bahwa kolom metadata (`subject`, `sessionIndex`, `rep`) selalu tersedia.
2. Gunakan `subject` sebagai kandidat utama untuk grouping sequence berbasis user.
3. Gunakan `sessionIndex` untuk session segmentation agar urutan temporal antar sesi tidak tercampur.
4. Gunakan `rep` untuk menjaga urutan repetisi dalam sesi.
5. Jangan menghapus nilai `UD` negatif secara buta karena bisa valid secara domain.
6. Tambahkan pemeriksaan konsistensi sederhana untuk relasi `H + UD` terhadap `DD` pada subset pasangan fitur.
7. Pisahkan fitur metadata dan fitur timing sebelum normalisasi.

## Implikasi untuk Evaluasi
1. Pastikan split data mencegah leakage antar repetisi/sesi yang terlalu mirip.
2. Laporkan hasil per skenario dengan basis fitur yang sama agar perbandingan adil.
3. Saat menganalisis error, cek apakah kesalahan terkonsentrasi pada sesi tertentu atau subjek tertentu.
4. Dalam pembahasan privasi, tegaskan bahwa pola timing dapat bertindak sebagai behavioral fingerprint.

## Catatan Implementasi Notebook
- Pada tahap audit dataset, tampilkan:
  - Jumlah unik `subject`.
  - Distribusi `sessionIndex`.
  - Distribusi `rep`.
  - Daftar kolom `H.*`, `DD.*`, dan `UD.*`.
- Pada tahap preprocessing, simpan ringkasan jumlah baris sebelum/sesudah filtering.

## Catatan Penulisan Laporan
- Gunakan bagian ini sebagai sumber deskripsi dataset pada Bab 3.2.
- Gunakan implikasi domain (`UD` bisa negatif dan relasi `H + UD = DD`) untuk memperkuat validasi metodologi.