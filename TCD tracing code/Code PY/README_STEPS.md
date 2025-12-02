# Panduan Eksekusi Pipeline TCD SQA & Klasifikasi (End-to-End)

Dokumen ini menjelaskan langkah-langkah lengkap untuk menjalankan seluruh pipeline proyek, mulai dari pemrosesan data mentah hingga klasifikasi akhir, termasuk semua visualisasi.

## 1. Persiapan Lingkungan (Google Colab)

Pastikan struktur folder Anda di Google Colab (atau Google Drive yang di-mount) terlihat seperti ini:

```
/content/ (atau path project Anda)
├── TCD_Data/                   <-- Folder Data Mentah
│   ├── Healthy Subjects/
│   │   ├── Healthy_Subjects_Recording_1.txt
│   │   └── ...
│   └── ICU Patients/
│       ├── ICU_Patients_Recording_22_MCA.txt  (Pastikan ada '_MCA' di nama file ICU)
│       └── ...
└── TCD tracing code/
    └── Code PY/                <-- Folder Kode Python
        ├── ... (semua script .py)
```

---

## 2. Urutan Eksekusi Pipeline

Jalankan perintah berikut secara berurutan di sel Google Colab.

### Langkah 1: Pembuatan Dataset (Preprocessing & Hard Cleaning)

Script ini akan membaca data mentah, mengekstrak envelope CBFV & SQI, memotong menjadi segmen 1024 sampel, melakukan filter otomatis (NaN/Flat), dan menyimpan hasilnya.

```python
%cd "/content/TCD tracing code/Code PY/"

!python process_dataset.py \
    --healthy_path "/content/TCD_Data/Healthy Subjects" \
    --icu_path "/content/TCD_Data/ICU Patients"
```

*   **Output:** File `.npz` individual di folder `processed_segments/`.
*   **Visualisasi:** Menampilkan contoh plot segmen envelope dan SQI dengan sumbu waktu yang benar.

### Langkah 2: Agregasi Dataset

Menggabungkan semua segmen valid dari file `.npz` individual menjadi satu dataset master.

```python
!python aggregate_segments.py
```

*   **Output:** `tcd_dataset.npz` (Dataset gabungan).

### Langkah 3: Analisis SQI & Pelabelan Kualitas (Core SQA)

Script ini menganalisis distribusi SQI, menentukan threshold kualitas (Tight/Loose), dan memberikan label otomatis (GOOD/BAD/BORDERLINE).

```python
!python analyze_and_label_sqi.py
```

*   **Output:**
    *   `sqi_labels.npz`: File berisi label kualitas untuk setiap segmen.
    *   `sqi_seg_distribution.png`: Histogram distribusi nilai SQI.

### Langkah 4: Visualisasi Kualitatif (Tambahan)

Dua script ini memberikan bukti visual tentang kualitas sinyal.

**A. Gradasi Kualitas (Low -> High)**
Menampilkan contoh sinyal pada target SQI 0.1, 0.3, 0.5, 0.7, dan 0.9.
```python
!python visualize_sqi_progression.py
```
*   **Output:** `sqi_progression.png`.

**B. Sampel Kategori (Good/Borderline/Bad)**
Menampilkan grid sampel acak untuk setiap kategori kualitas.
```python
!python visualize_quality_samples.py
```
*   **Output:** `quality_samples.png`.

### Langkah 5: Split Data (Train/Test)

Memisahkan data menjadi set Latih dan Uji agar evaluasi valid.

```python
!python split_dataset.py
```

*   **Output:** `tcd_train.npz` dan `tcd_test.npz`.

### Langkah 6: (Opsional) Training VAE

Melatih VAE untuk belajar bentuk sinyal normal dan menghitung *reconstruction error*.

```python
!python train_vae.py --dataset_filepath tcd_train.npz
```

*   **Output:** `vae_model.pth`, `recon_errors.npz`.
*   **Visualisasi VAE:** `!python visualize_vae_performance.py` -> `vae_performance_analysis.png`.

### Langkah 7: (Opsional) Training CycleGAN untuk Restorasi

Melatih CycleGAN untuk mengubah segmen BORDERLINE menjadi GOOD.

```python
!python train_cyclegan.py --train_data_path tcd_train.npz --epochs 50
```

*   **Output:** `generator_AB.pth`.
*   **Visualisasi GAN:** `!python visualize_gan_restoration.py` -> `gan_restoration_analysis.png`.

### Langkah 8: Training Klasifikasi (Healthy vs ICU) - FINAL

Melatih model klasifikasi (ResNet18 & Self-ResNet18) dengan berbagai skenario data.

```python
!python train_classifier.py --epochs 20
```

*   **Output:**
    *   Metrik performa lengkap.
    *   `classification_results_all_models.png`: Grafik perbandingan performa antar skenario.

---

## 3. Ringkasan File Output

| File |
| :--- | 
| `tcd_dataset.npz` | Dataset utama berisi segmen CBFV dan SQI valid. |
| `sqi_labels.npz` | Label kualitas (GOOD/BAD/BORDERLINE) untuk segmen. |
| `sqi_progression.png` | Visualisasi gradasi kualitas sinyal (0.1 - 0.9). |
| `quality_samples.png` | Grid sampel sinyal Good, Borderline, Bad. |
| `classification_results_all_models.png` | Grafik hasil akhir perbandingan klasifikasi. |

```