Oke, kita rapihin dan **update** plannya sesuai diskusi terakhir:

* Hard rule bukan cuma NaN, tapi juga flat / nggak ada beat.
* **Core SQA** pakai **SQI Wadehn (wajib)**.
* **VAE** hanya sebagai **opsional / indikator reconstruction error tambahan**, bukan penentu utama.
* CycleGAN juga opsional.

Aku tulis ulang end-to-end-nya, sudah *fix* dengan versi yang lebih “aman buat skripsi”.

---

## 1. Data & ruang lingkup

* Dataset: **Transcranial Doppler Ultrasound Database (Philips CX50)** – IEEE DataPort.
* Subjek:

  * 6 **healthy volunteers**
  * 12 **neurocritical care / ICU patients** (TBI, hydrocephalus, hemorrhage, dsb.)
* Arteri:

  * Healthy: hanya **MCA**
  * ICU: punya rekaman **MCA** dan **ICA**

**Keputusan desain:**

* Task utama: **klasifikasi Healthy vs ICU**.
* Untuk menghindari bias jenis arteri, dan mengikuti setup paper *Deep Learning Framework for the Detection of Abnormality*:

  * **Hanya rekaman MCA** yang dipakai, baik untuk healthy maupun ICU.
* Rekaman ICA:

  * diabaikan untuk klasifikasi,
  * boleh dipertimbangkan untuk eksperimen unsupervised tambahan, tapi **bukan core FP**.

---

## 2. Tahap A – Ekstraksi maximal flow velocity waveform (algoritma Wadehn)

**Tujuan:** Mengubah sinyal TCD mentah (IQ/echo) menjadi:

* **envelope maksimal kecepatan aliran darah (CBFV)**
* * **Signal Quality Index (SQI) per beat**

mengikuti metode Wadehn & Heldt.

**Langkah utama:**

1. **Wall filtering & STFT**

   * High-pass filter untuk mengurangi kontribusi dinding pembuluh.
   * STFT → Doppler spectrogram (waktu × frekuensi), magnitude di-log & dinormalisasi → dikonversi ke kecepatan.

2. **Binarisasi adaptif**

   * Hitung threshold awal pakai Otsu.
   * Uji beberapa nilai γ di sekitar Otsu (γ₁…γₖ).
   * Untuk tiap γ:

     * binarisasi spectrogram jadi “sinyal vs noise”,
     * median filter 2D untuk mereduksi speckle.

3. **Envelope tracing**

   * Pada tiap kolom waktu: pilih piksel “atap” dari massa sinyal yang memenuhi sanity check:

     * di bawah piksel masih ada area putih,
     * perubahan kecepatan antar waktu tidak melompat di luar batas fisiologis.

4. **Beat detection & SQA beat-wise (Wadehn)**

   * Deteksi onset beat memakai algoritma [29] (aslinya untuk ABP, diadaptasi dengan skala amplitude).
   * Untuk tiap beat:

     * cek 4 sanity check (amp minimum, max−min, range durasi absolut, dan durasi relatif terhadap median).
   * Bentuk **template beat** (median dari beat yang lolos sanity check).
   * Hitung **normalized MSE** antara setiap beat dan template (75% awal durasi).
   * Beat dengan normalized MSE > threshold → dilabel **artefak**.
   * Definisikan:

     * **artifact-index** = % beat yang dilabel artefak per 1 menit,
     * **SQI_beat** = 1 − normalized MSE (0–100%) → kualitas beat.

5. **Pemilihan envelope terbaik**

   * Untuk tiap γ:

     * jalankan proses SQA, hitung artifact-index.
   * Pilih kandidat envelope dengan artifact-index terkecil sebagai **envelope final** untuk 1 menit tersebut.
   * SQI_beat dari kandidat terpilih → dipakai sebagai SQI final.

**Output Tahap A (per rekaman MCA):**

* `cbfv_envelope(t)` @ ~217 Hz
* daftar beat (onset/offset)
* `SQI_beat` untuk setiap beat

---

## 3. Tahap B – Segmentasi & hard cleaning otomatis

**Tujuan:** Mengotomasi pembuangan segmen “corrupted parah” (signal loss, flat line, dsb.) yang di paper lain dibuang lewat inspeksi visual/manual.

### 3.1. Segmentasi

* Envelope MCA di-window menjadi segmen panjang **1024 sampel** (~4.7 s), seperti di paper deep learning sebelumnya.
* Awalnya tanpa overlap (opsional nanti bisa coba overlap).

### 3.2. Hard rule filter (otomatis)

Untuk setiap segmen 1024 sampel:

1. **Cek NaN / Inf (numerik tidak valid)**

   * Jika ada sampel `NaN` atau `Inf` → segmen **dibuang**.

2. **Cek flat / low amplitude (kehilangan pulsasi)**

   * Hitung:

     * `amp = max(seg) − min(seg)`
     * `std = std(seg)`
   * Jika:

     * `amp < A_min` **atau** `std < std_min`
       (nilai A_min dan std_min ditentukan dari statistik data, misalnya 10 cm/s dan 3 cm/s)
       → segmen dianggap “flat/no-pulse” → **dibuang**.

3. **Cek beat validity (pakai info Wadehn)**

   * Hitung:

     * `n_total_beat` = jumlah beat yang onset-nya jatuh di segmen,
     * `n_valid_beat` = jumlah beat yang lolos sanity check Wadehn (bukan artefak).
   * Jika:

     * `n_total_beat == 0`, atau
     * `n_valid_beat == 0`, atau
     * (opsional) `artifact_index_seg` > threshold tinggi (mis. 0.9)
       → segmen dianggap **corrupted** → **dibuang**.

Segmen yang lolos semua hard rule ini disebut **`segments_clean`**.

**Output Tahap B:**

* `segments_clean.npy` → array `(N_seg, 1024)`
  = segmen envelope yang:

  * numerik valid,
  * punya pulsasi,
  * masih punya minimal satu beat valid.

Segmen yang mirip kategori **“corrupted”** di Fig. 3(d) (TCD GAN) diharapkan tereliminasi di tahap ini.

---

## 4. Tahap C – Modul SQA otomatis (berbasis SQI Wadehn)

*(VAE = opsi tambahan sebagai indikator reconstruction error)*

Ini inti kontribusi “ganti manual quality labeling” dengan mekanisme otomatis.

### 4.1. SQI per segmen (SQI_seg)

Dari output Wadehn:

* Untuk tiap segmen 1024 sampel di `segments_clean`:

  1. Identifikasi beat-beat yang interval waktunya jatuh dalam segmen tersebut.
  2. Ambil `SQI_beat` untuk beat-beat itu.
  3. Hitung **aggregat**, misalnya:

     * `SQI_seg = median(SQI_beat di segmen)`
* Simpan semua ke `SQI_seg.npy`.

Ini mengangkat SQI dari level **beat** → level **segmen 1024 sampel**.

i
Gunakan dua ambang:

* `T_high` (misal 80%)
* `T_low` (misal 40%)

Untuk setiap segmen:

* Jika `SQI_seg ≥ T_high` → label **GOOD / High quality**
* Jika `SQI_seg ≤ T_low` → label **BAD / Low quality**
* Jika `T_low < SQI_seg < T_high` → label **BORDERLINE / Medium**

Simpan:

* `quality_labels_sqi.npy` (isi: 1 untuk GOOD, 0 untuk BAD, −1 untuk BORDERLINE)

👉 Bagian ini **langsung menggantikan manual label**:

* high / low / mediocre / corrupted di paper CycleGAN TCD, dan
* clear / corrupted di paper deep learning framework,
  tapi definisinya **berbasis SQI Wadehn**, bukan visual human.

### 4.3. (Opsional) VAE 1D untuk reconstruction error tambahan

Bagian ini **opsional / ekstensi**, bukan core:

* Train **VAE 1D** secara unsupervised di `segments_clean`:

  * input: segmen 1024 sampel yang sudah normalisasi,
  * loss: reconstruction MSE + β·KL.
* Setelah training:

  * untuk setiap segmen, hitung:

    * `recon_error = MSE(x, x_hat)`.
* Reconstruction error ini bisa dipakai untuk:

  * analisis tambahan (apakah segmen dengan SQI rendah juga punya error tinggi),
  * atau filter ekstra: buang segmen dengan error di atas persentil tertentu.
* VAE tidak mengubah definisi SQI, tapi menyediakan **indikator kualitas tambahan berbasis data-driven**.

### 4.4. SQA-Net (classifier kualitas dari pseudo-label SQI)

**Tujuan:** punya model ringan (CNN 1D) yang bisa memprediksi kualitas segmen (GOOD vs BAD) langsung dari waveform, tanpa selalu menghitung SQI beat.

1. Dataset:

   * Input: `segments_clean`
   * Label: `quality_labels_sqi`
   * Hanya pakai segmen dengan label:

     * 1 (GOOD) dan 0 (BAD).
       Segmen BORDERLINE (−1) **tidak dipakai** untuk training SQA-Net.

2. Model:

   * 1D-CNN sederhana:

     * beberapa Conv1D + ReLU + pooling,
     * AdaptiveAvgPool1d,
     * Fully connected → 2 output (GOOD/BAD).

3. Training:

   * Loss: CrossEntropy,
   * Output: `sqa_net.pth`.

**Fungsi SQA otomatis (versi implementasi):**

* Untuk segmen baru:

  1. Pastikan lolos hard rule (Tahap B).
  2. **Opsi 1 (paling grounded):**

     * Hitung SQI_seg dari SQI_beat → pakai threshold T_high/T_low.
  3. **Opsi 2 (praktis):**

     * Langsung pakai SQA-Net untuk klasifikasi GOOD vs BAD (dilatih dari pseudo-label SQI).

---

## 5. Tahap D – (Opsional) CycleGAN untuk restorasi segmen borderline

Bagian ini benar-benar **opsional** / advanced.

**Motivasi:**

* Segmen BORDERLINE (SQI di tengah-tengah) masih mengandung informasi fisiologis,
* tapi kualitasnya tidak setinggi GOOD → berpotensi merusak training classifier.

**Ide:**

* Domain Y (high-quality): segmen berlabel **GOOD** (SQI_seg ≥ T_high).
* Domain X (borderline): segmen berlabel **BORDERLINE** (T_low < SQI_seg < T_high).

Latih **1D-CycleGAN**:

* G_{X→Y}: mengubah segmen borderline → “lebih mirip” segmen GOOD.
* G_{Y→X}: kebalikannya, untuk cycle consistency.
* D_X, D_Y: discriminator di domain X dan Y.

Loss:

* adversarial loss,
* cycle consistency loss,
* identity loss (GOOD yang masuk G_{X→Y} harus tetap mirip),
* opsional: MSE/korelasi waveform.

**Penggunaan:**

* Di konfigurasi eksperimen tertentu:

  * Segmen BORDERLINE + G_{X→Y}(BORDERLINE) bisa ikut training classifier sebagai versi “dipulihkan”.

**Perbedaan dengan paper GAN:**

* Di paper GAN TCD, domain high/low didefinisikan manual (MATLAB app + label high/low/mediocre/corrupted).
* Di sini:

  * domain X/Y dibentuk otomatis dari **SQI_seg Wadehn**, bukan anotasi manual.

---

## 6. Tahap E – Klasifikasi Healthy vs ICU

Setelah modul SQA siap, baru masuk ke **task utama**.

**Data untuk klasifikasi:**

* Hanya segmen dari **rekaman MCA** (healthy & ICU).
* Setiap segmen punya:

  * label kualitas (GOOD / BORDERLINE / BAD dari SQI),
  * label kelas pasien (Healthy vs ICU).

### 6.1. Pemilihan segmen untuk training

Definisikan beberapa skenario:

1. **Baseline 1 – Tanpa SQA (No SQA)**

   * Hanya pakai hard rule (Tahap B) untuk buang NaN/flat/corrupted.
   * Semua segmen yang lolos hard rule → dipakai untuk training.

2. **Baseline 2 – SQA berbasis SQI sederhana**

   * Segmen yang dipakai hanya yang `SQI_seg ≥ T_baseline`
     (misalnya median SQI_seg keseluruhan, tanpa kategori GOOD/BAD/BORDERLINE detail).

3. **Proposed – SQA otomatis (SQI-based + SQA-Net)**

   * Segmen yang dipakai training adalah:

     * hanya segmen **GOOD** (SQI_seg ≥ T_high),
     * (opsional) + segmen BORDERLINE yang sudah direstorasi dengan CycleGAN.
   * BAD dibuang.

Dengan ini kamu bisa menunjukkan efek SQA otomatis dibanding baseline.

### 6.2. Model klasifikasi

* Kandidat arsitektur:

  * CNN 1D sederhana (mirip SQA-Net, tapi output 2 kelas: Healthy/ICU), atau
  * Self-ResNet / Self-ResAttentioNet seperti di paper deep learning framework.

* Input:

  * Segmen 1024 sampel, dengan normalisasi per segmen.

* Output:

  * 2 kelas: 0 = Healthy, 1 = ICU.

* Splitting:

  * **patient-wise split**: seluruh segmen dari satu pasien hanya muncul di salah satu dari train/val/test untuk menghindari leakage.

### 6.3. Evaluasi

* Metrik:

  * Accuracy,
  * ROC-AUC,
  * Sensitivity (recall ICU),
  * Specificity (recall Healthy),
  * Confusion matrix.

* Bandingkan:

  * Baseline No SQA vs SQI-only vs Proposed SQA,
  * Lihat apakah:

    * SQA otomatis meningkatkan akurasi/AUC,
    * mengurangi kesalahan akibat segmen berkualitas jelek,
    * membuat performa lebih stabil.

---

## 7. Ringkasan perbedaan dengan metode sebelumnya

* **Vs Wadehn (2020)**
  Kamu tidak hanya memakai algoritma Wadehn untuk ekstraksi envelope, beat detection, dan SQI, tetapi:

  * mengangkat **SQI Wadehn sebagai basis modul SQA otomatis di level segmen 1024 sampel**,
  * memakai SQI untuk membentuk kategori kualitas dan filter data sebelum deep learning.

* **Vs Deep Learning Framework (Nisha dkk.)**
  Mereka:

  * masih melakukan manual cleaning (buang signal cuts/NaN/flat line),
  * dan manual labeling Clear vs Corrupted.

  Kamu:

  * mengganti manual cleaning dengan **hard rule otomatis**,
  * mengganti manual Clear/Corrupted dengan **kategori kualitas berbasis SQI Wadehn**,
  * dan melatih **SQA-Net** dari pseudo-label SQI sehingga tidak butuh anotasi kualitas manual dalam skala besar.

* **Vs TCD GAN & ECG CycleGAN**
  Studi tersebut memakai CycleGAN untuk memperbaiki kualitas sinyal, tetapi domain clean/dirty ditentukan lewat **label visual manual** oleh ahli.
  Dalam rencanamu, jika CycleGAN dipakai, domain high/low didefinisikan otomatis dari **SQI_seg** (dan opsional rekonstruksi VAE), sehingga pendekatan restorasi menjadi **quality-aware tanpa labeling manual**.

---

Kalau mau, habis ini kita bisa:

* translate plan ini jadi **outline Bab 3 skripsi** (3.1–3.6), atau
* fokus ke satu bagian dulu (misal: tulis formal bagian “Tahap C – SQA berbasis SQI Wadehn” lengkap dengan persamaan).
