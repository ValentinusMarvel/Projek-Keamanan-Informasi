# Keystroke Dynamics — Modular Notebooks

Ringkasan singkat untuk menjalankan dan berkontribusi pada proyek ini.

## Quick start

1. Jalankan bootstrap untuk membuat virtual environment, menginstal dependensi minimum, dan mendaftarkan kernel Jupyter:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned; .\bootstrap_venv.ps1
```

2. Pilih kernel `projek-keystroke (.venv)` di VS Code / Jupyter.

3. Jalankan pipeline modular (mengeksekusi notebook 01→99):

```powershell
python scripts/run_pipeline.py
```

4. Atau jalankan satu notebook saja (contoh: preprocessing):

```powershell
# menjalankan notebook 02 dengan nbclient (alternatif manual)
python -c "from pathlib import Path; import nbformat; from nbclient import NotebookClient; p=Path('notebooks/modular/02_preprocessing.ipynb'); nb=nbformat.read(p,as_version=4); NotebookClient(nb,kernel_name='projek-keystroke').execute(cwd=Path('.').resolve()); nbformat.write(nb,p)"
```

## Struktur penting
- `notebooks/modular/` — modul-notebook berurutan (01..07,99)
- `src/` — kode bersama (data loaders, preprocessing, models)
- `data/raw/` — letakkan `DSL-StrongPasswordData.csv` di sini
- `data/processed/` — artefak preprocessing (mis. `sequence_bundle.npz`)
- `outputs/models/` — model tersimpan
- `outputs/reports/` — ringkasan metrik (mis. `fl_metrics.json`)

## Jika pipeline gagal
- Pastikan kernel yang dipakai oleh runner sama dengan `.venv` yang berisi dependensi (lihat di atas).
- Jika `sequence_bundle.npz` tidak ada: jalankan `02_preprocessing.ipynb` dulu.
- Jika notebook berhenti karena dataset terlalu kecil, jalankan `03_baseline_lstm.ipynb` menggunakan dataset demo (`src.data.create_demo_dataset`) atau sesuaikan preprocessing.

## Pengembangan selanjutnya
- `06_privacy_attack_eval.ipynb` masih placeholder — ada task untuk menambahkan MIA (membership inference attack) nyata.

## Kontak
Jika ada error install atau runtime, salin log dan file `outputs/reports/*.json` lalu buka issue atau minta saya bantu debug.
