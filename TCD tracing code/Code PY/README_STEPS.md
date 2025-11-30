# Panduan Eksekusi Pipeline TCD SQA & Klasifikasi

Dokumen ini menjelaskan langkah-langkah untuk menjalankan seluruh pipeline proyek, mulai dari pemrosesan data mentah hingga klasifikasi akhir.

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
        ├── __init__.py
        ├── main.py
        ├── preprocessing.py
        ├── spectrogram_utils.py
        ├── tracing.py
        ├── sqa.py
        ├── postprocessing.py
        ├── process_dataset.py
        ├── analyze_and_label_sqi.py
        ├── vae_model.py
        ├── train_vae.py
        ├── cyclegan_model.py
        ├── train_cyclegan.py
        ├── classifier_model.py
        └── train_classifier.py
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

*   **Output:** `tcd_dataset.npz` (Dataset segmen valid dan corrupted).
*   **Visualisasi:** Script juga akan menampilkan contoh plot segmen envelope dan SQI.

### Langkah 2: Analisis SQI & Pelabelan Kualitas (Core SQA)

Script ini menganalisis distribusi SQI dari segmen valid, menentukan threshold kualitas, dan memberikan label otomatis (GOOD/BAD/BORDERLINE).

```python
!python analyze_and_label_sqi.py
```

*   **Output:**
    *   `sqi_labels.npz`: File berisi label kualitas untuk setiap segmen.
    *   `sqi_seg_distribution.png`: Histogram distribusi nilai SQI.

### Langkah 3: (Opsional) Training VAE

Script ini melatih model Variational Autoencoder (VAE) untuk mempelajari pola sinyal normal dan menghitung *reconstruction error* sebagai fitur kualitas tambahan.

```python
!python train_vae.py
```

*   **Output:**
    *   `vae_model.pth`: Model VAE terlatih.
    *   `recon_errors.npz`: Error rekonstruksi untuk setiap segmen.
    *   `vae_training_loss.png` & `vae_reconstruction_sample.png`.

### Langkah 4: (Opsional) Training CycleGAN untuk Restorasi

Script ini melatih CycleGAN untuk mengubah segmen kualitas BORDERLINE menjadi menyerupai segmen GOOD.

```python
!python train_cyclegan.py
```

*   **Output:**
    *   `generator_AB.pth`: Model generator (Borderline -> Good).
    *   `cyclegan_restoration_sample.png`: Contoh hasil restorasi sinyal.

### Langkah 5: Training Klasifikasi (Healthy vs ICU)

Script ini adalah eksperimen utama. Ia melatih model CNN 1D untuk membedakan pasien Healthy dan ICU menggunakan berbagai skenario data (Baseline vs SQA).

```python
!python train_classifier.py
```

*   **Skenario yang Dijalankan:**
    1.  **Baseline (No SQA):** Menggunakan semua segmen valid.
    .   **Proposed (SQA - Good Only):** Hanya menggunakan segmen berlabel GOOD.
    3.  **Proposed + GAN (Jika ada):** Menggunakan segmen GOOD + segmen BORDERLINE yang direstorasi.
*   **Output:**
    *   Metrik performa (Akurasi, AUC, Sensitivitas, Spesifisitas) di terminal.
    *   `classification_results.png`: Grafik perbandingan performa antar skenario.

---

## 3. Catatan Tambahan

*   Pastikan path data di Langkah 1 disesuaikan dengan lokasi sebenarnya di Google Drive/Colab Anda.
*   Langkah 3 (VAE) dan 4 (CycleGAN) bersifat opsional. Pipeline klasifikasi (Langkah 5) akan tetap berjalan (dengan melewatkan skenario GAN) jika langkah tersebut dilewati.

```