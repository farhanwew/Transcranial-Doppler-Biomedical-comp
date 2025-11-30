Siap, kita rangkum dulu semuanya biar kamu punya bahan report yang rapi 👇
Aku bagi dua bagian sesuai poinmu.

---

## 1. Ringkasan masing-masing paper & pendekatannya

### 1.1. Wadehn & Heldt (2020) – *Adaptive Maximal Blood Flow Velocity Estimation From Transcranial Doppler Echos*

**Tujuan utama:**

* Mengusulkan **algoritma otomatis** untuk:

  1. Mengestimasi **maximal blood flow velocity** (envelope CBFV) dari sinyal TCD mentah (echo/IQ).
  2. Menghitung **signal quality index (SQI)** untuk menilai kualitas sinyal/beat.

**Data & setup:**

* Perangkat: **Philips CX50**, arteri **MCA & ICA**.
* Subjek:

  * 6 **healthy** → 16 rekaman MCA (±2 jam).
  * 12 **neurocritical care patients** → 16 rekaman (MCA/ICA, ±2 jam 40 menit).
* Pasien ICU datang dengan kondisi TBI, hydrocephalus, intraparenchymal / subarachnoid hemorrhage, dsb.

**Pendekatan algoritmik (core steps):**

1. **Dari IQ → spectrogram Doppler**

   * Wall filter (high-pass 100 Hz).
   * STFT → spectrogram (waktu × frekuensi).
   * Log-compression + normalisasi → intensitas [0,1].
   * Konversi frekuensi → kecepatan aliran darah.

2. **Segmentasi gambar (biner) & adaptive threshold**

   * Pakai **Otsu threshold** sebagai titik awal.
   * Lakukan **grid search** di sekitar threshold Otsu (0.9γ, 0.95γ, γ, 1.05γ, 1.1γ).
   * Untuk setiap threshold:

     * Binarize spectrogram (sinyal = putih, noise = hitam).
     * Terapkan **2D median filter** untuk menghilangkan speckle.

3. **Envelope tracing + physiological sanity checks**

   * Untuk tiap kolom waktu:

     * Cari beberapa kandidat pixel putih di bagian kecepatan tertinggi.
     * Terapkan 2 cek fisiologis:

       * Di bawah kandidat masih ada “kolom putih” (bukan pixel nyasar).
       * Transition antar waktu **tidak meloncat > ~30 cm/s**.
     * Kalau tidak ada kandidat valid → pakai nilai sebelumnya, dengan fallback setelah beberapa sampel.

4. **Signal Quality Assessment (SQA) & SQI**

   * Deteksi beat pada envelope.
   * Buat **template beat** (median beat).
   * Hitung **normalized MSE** tiap beat terhadap template (fokus di 75% pertama beat).
   * Beat dengan error tinggi → artefak.
   * Dari sini:

     * **Artifact index** per segmen (persentase beat artefak).
     * **SQI beat-wise** (0–100%) berdasarkan kemiripan dengan template.

5. **Feedback loop**

   * Semua langkah diulang untuk tiap threshold kandidat.
   * Dipilih threshold yang menghasilkan **artifact index terkecil** → envelope final.

**Peran MTCM:**

* **MTCM (Modified Threshold Crossing Method)** hanya dipakai sebagai **pembanding/baseline**.
* Menunjukkan bahwa metode Wadehn:

  * lebih robust ke noise,
  * lebih akurat mengikuti “atap” spectrogram,
  * dan punya SQI untuk menandai bagian jelek.

---

### 1.2. Nisha dkk. (2023) – *A Deep Learning Framework for the Detection of Abnormality in MCA Blood Flow*

**Tujuan utama:**

* Mengembangkan **deep learning classifier** untuk mendeteksi **abnormalitas aliran darah MCA**:

  * terutama **Healthy vs ICU/neurocritical patients**.
* Menggunakan **waveform maximal flow velocity** sebagai input, yang diekstrak dari dataset Philips CX50.

**Data:**

* Sumber: **IEEE DataPort – Transcranial Doppler Ultrasound Database (Philips CX50)**.
* Subjek:

  * 6 healthy, 12 ICU (sama kelompok seperti Wadehn).
* Tapi di studi ini:

  * Fokus hanya **MCA**,
  * Dibagi lagi menjadi banyak **segmen 1024 sampel @ 217 Hz**.

**Pendekatan (pipeline):**

1. **Ekstraksi envelope**

   * Menggunakan **algoritma Wadehn** untuk mendapatkan **maximal flow velocity waveform** dari sinyal TCD.
   * Termasuk sanity checks fisiologis dari Wadehn.

2. **Pembersihan awal (drop sinyal sangat jelek) – manual**

   * **Dibuang**:

     * segmen yang mengandung **signal cuts**,
     * segmen dengan **NaN**,
     * segmen **flat line**,
     * segmen tanpa “informasi fisiologis” (tidak tampak pola pulsasi).
   * Langkah ini dilakukan berdasarkan **inspeksi dan kriteria manual**.

3. **Segmentasi & manual quality labeling**

   * Envelope kemudian di-**window** jadi segmen 1024 sampel (~4.7 s).
   * Setiap segmen dilabeli secara **manual**:

     * **Clear** → minim artefak.
     * **Corrupted** → masih ada pola fisiologis, tapi noisy/distorsi.
   * Segmen yang benar-benar rusak (NaN, flat, dll.) sudah dibuang di langkah sebelumnya.

4. **Model deep learning**

   * Arsitektur: varian **Self-ONN** (Self-ResNet18, Self-ResAttentioNet18).
   * Input: 1D waveform segmen.
   * Task utama:

     * klasifikasi **Healthy vs ICU** (abnormal).
   * Mereka juga analisis performa terhadap segmen Clear vs Corrupted.

**Catatan penting:**

* **Kualitas sinyal ditangani secara manual**:

  * pilih segmen mana yang dipakai,
  * label Clear vs Corrupted.
* SQI Wadehn **tidak dipakai** untuk menggantikan manual labeling kualitas, hanya jadi bagian internal definisi sanity checks.

---

### 1.3. Paper GAN – *Deep learning-based middle cerebral artery blood flow abnormality* (DopplerNet + 1D-CycleGAN)

**Tujuan utama:**

* Mendeteksi **abnormalitas aliran MCA** menggunakan deep learning,
* sambil **mengatasi masalah kualitas sinyal** dengan **1D-CycleGAN** untuk “membersihkan” sinyal.

**Pendekatan (pipeline):**

1. **Ekstraksi envelope**

   * Lagi-lagi menggunakan **algoritma Wadehn** untuk mendapatkan waveform maximal flow velocity MCA.

2. **Manual data cleaning & multi-level quality labeling**

   * Tiga pengamat / operator **menilai secara visual** waveform dan membaginya menjadi:

     * **High-quality**
     * **Moderate-quality**
     * **Low-quality**
     * **Corrupted**
   * Sinyal **corrupted** → **dibuang total**.

3. **Restorasi sinyal dengan 1D-CycleGAN**

   * Domain **X**: sinyal **Low/Moderate quality**.
   * Domain **Y**: sinyal **High quality**.
   * Latih **1D-CycleGAN**:

     * Generator X→Y: membuat sinyal low/moderate jadi mirip high-quality.
     * Generator Y→X: untuk cycle consistency.
     * Dua discriminator untuk membedakan sinyal real/fake.
   * Loss:

     * adversarial, cycle-consistency, identity,
     * plus MSE & korelasi untuk menjaga bentuk wave.
   * Output: “versi bersih” dari sinyal low/moderate.

4. **Klasifikasi dengan DopplerNet**

   * Model CNN 1D (DopplerNet/DopplerNetv2).
   * Input: sinyal high-quality + sinyal restored (hasil CycleGAN).
   * Output: klasifikasi **Healthy vs ICU / abnormal**.

**Catatan penting:**

* Lagi-lagi, **kualitas sinyal ditentukan secara manual** (high/low/moderate/corrupted).
* GAN digunakan untuk **restorasi**, tetapi **berangkat dari label kualitas manual**.

---

### 1.4. Jaishankar dkk. – *Spectral Approach to Noninvasive ICP Estimation* (sekilas)

* Paper ini tidak memakai dataset Philips CX50 secara langsung untuk tugasmu, tapi:

  * dipakai Wadehn sebagai rujukan **protokol klinis** (bagaimana data di-collect, kondisi pasien, dll.).
* Datasetnya lebih fokus ke kombinasi **ABP, ICP, CBFV** dan model ICP, bukan ke SQA otomatis atau klasifikasi Healthy vs ICU.

---

## 2. Implementasi rencana FP-mu & perbaikan terhadap metode sebelumnya

Sekarang kita rangkum **rencana FP tim kamu** berdasarkan diskusi, dan apa yang membedakannya dari 3 paper di atas.

### 2.1. Goal FP (versi singkat)

> **Mengembangkan modul *automatic Signal Quality Assessment (SQA)* untuk waveform CBFV TCD (MCA) dari dataset Philips CX50**, sehingga:
>
> * tidak perlu manual quality labeling besar-besaran,
> * bisa memfilter / menangani segmen sinyal jelek secara otomatis,
> * dan meningkatkan robustness model klasifikasi (Healthy vs ICU / normal vs abnormal).

Kamu **boleh** tetap pakai GAN (misalnya 1D-CycleGAN) untuk restorasi, **tapi bedanya label kualitasnya bukan dari manusia**, melainkan dari kombinasi **SQI + VAE/autoencoder** (pseudo-label otomatis).

---

### 2.2. Rencana implementasi FP – alur teknis

#### 1) Data & ekstraksi envelope

* Dataset: **Philips CX50 TCD** dari IEEE DataPort (Healthy Subjects + ICU Patients).
* Langkah:

  1. Ambil sinyal echo/IQ.
  2. Jalankan **algoritma Wadehn** (code Matlab sudah tersedia):

     * wall filter → STFT → spectrogram → envelope tracing.
  3. Simpan:

     * **Envelope CBFV** @ ~217 Hz,
     * **SQI beat-wise** dari modul SQA Wadehn (jika diambil dari code).

Output tahap ini:
➡️ Satu deret waktu CBFV per rekaman + informasi beat + SQI beat.

---

#### 2) Segmentasi & *hard rule filter* (otomatis)

* Window envelope menjadi **segmen tetap**:

  * misalnya 1024 sampel (≈4.7 s), mengikuti paper Nisha/GAN.
* Untuk setiap segmen:

  * Buang otomatis jika:

    * ada **NaN / Inf**,
    * **flat line** (std sangat kecil, max−min di bawah threshold),
    * energi sinyal sangat kecil atau tidak masuk rentang fisiologis wajar.
* Ini mengotomasi langkah “buang signal cuts / NaN / flat line” yang di paper lain dilakukan manual.

Output:
➡️ **Candidate segments** (yang masih punya bentuk sinyal).

---

#### 3) Fitur kualitas & VAE (unsupervised quality modeling)

Untuk tiap candidate segment:

1. Hitung **fitur berbasis Wadehn**:

   * Gabungkan beat di dalam segmen,
   * hitung **SQI_seg = median SQI beat** di segmen tersebut.

2. Hitung **fitur statistik sederhana**:

   * varian, max−min, skewness, energy, rasio HF/LF (opsional).

3. **Latih 1D autoencoder / VAE**:

   * Input: segmen envelope (1024 sampel),
   * Tanpa label.
   * Setelah training, untuk tiap segmen:

     * hitung **reconstruction error** (MSE),
     * normalisasi ke [0,1].

Tujuan VAE:
➡️ Memberikan **skor “keanehan”** sinyal tanpa membutuhkan domain knowledge klinis.

---

#### 4) Hybrid SQI – gabungan SQI Wadehn + VAE

Definisikan **skor kualitas gabungan** per segmen:

* Misal:
  [
  \text{SQI_final} = \alpha \cdot \text{SQI_seg} + (1-\alpha)\cdot (1 - \text{norm_error_AE})
  ]
  dengan 0 ≤ α ≤ 1 (mis. α = 0.5).

Interpretasi:

* **SQI_seg tinggi** dan **error_AE rendah** → SQI_final tinggi (segmen sangat bagus).
* SQI_seg rendah dan error_AE tinggi → SQI_final rendah (segmen sangat jelek).

---

#### 5) Pseudo-label kualitas & SQA-Net

Gunakan **SQI_final** untuk membuat **pseudo-label otomatis**:

* Jika `SQI_final ≥ T_high` → label **GOOD** (high-quality).
* Jika `SQI_final ≤ T_low` → label **BAD** (very poor quality).
* Jika di antara T_low dan T_high → **BORDERLINE** (opsional: abaikan dari training SQA-Net).

Lalu:

* Latih **SQA-Net (small 1D-CNN / MLP)**:

  * Input: segmen envelope 1024 sampel.
  * Label: GOOD vs BAD (pseudo-label).
* Fungsi SQA-Net:

  * Modul cepat untuk memberikan **prediksi kualitas** segmen di inference,
  * tanpa perlu menghitung VAE lagi (jika kamu ingin efisiensi).

Output tahap ini:
➡️ **Modul SQA otomatis** yang bisa menandai segmen GOOD/BAD berdasarkan pseudo-label, bukan manual label.

---

#### 6) (Opsional) 1D-CycleGAN untuk restorasi segmen borderline

Kalau kamu masih ingin pakai GAN:

* Gunakan pseudo-label:

  * Domain **Y (high quality)** = segmen dengan SQI_final ≥ T_high.
  * Domain **X (low/moderate quality)** = segmen dengan SQI_final di area tengah (BORDERLINE).
  * Segmen **BAD banget** (SQI_final ≤ T_low) → dibuang saja.
* Latih **1D-CycleGAN X↔Y** seperti di paper DopplerNet, BEDANYA:

  * tidak ada manual quality labeling,
  * domain X/Y didefinisikan otomatis oleh modul SQI_final.

Hasil:

* Segmen borderline bisa di-*restore* menjadi bentuk yang lebih mirip high-quality sebelum masuk ke klasifier.

---

#### 7) Klasifikasi abnormalitas (Healthy vs ICU)

Untuk tahap akhir:

* Ikuti pendekatan Nisha / DopplerNet:

  * Model: Self-ResNet / Self-ResAttentioNet / DopplerNet / CNN 1D biasa.
  * Input:

    * segmen GOOD saja, atau
    * GOOD + restored (hasil CycleGAN).
* Target: **Healthy vs ICU** (binary classification).

Eksperimen yang bisa kamu lakukan:

1. **Baseline**:

   * Tanpa SQA, semua segmen dipakai (kecuali NaN/flat).
2. **Dengan rule-only SQA**:

   * Hanya filter hard (NaN, flat, range).
3. **Dengan hybrid SQI + VAE**:

   * Hanya pakai GOOD, atau GOOD+borderline weighted.
4. **Dengan hybrid SQI + VAE + GAN**:

   * GOOD + restored borderline.

Bandingkan akurasi, AUC, sensitivitas/spesifisitas, dll.

---

### 2.3. Apa yang diperbaiki dibanding metode sebelumnya?

#### Dibanding Wadehn (2020):

* Wadehn:

  * fokus ke **estimasi envelope** dan **SQI beat-wise**,
  * SQI dipakai untuk pilih threshold terbaik & menandai beat artefak,
  * **tidak membangun modul SQA segmen** (4–5 detik) untuk pipeline deep learning.

* Kamu:

  * **Mengangkat SQI Wadehn** dari sekadar “alat internal” menjadi fitur utama modul SQA di level segmen.
  * Menggabungkannya dengan **VAE (unsupervised)** untuk menilai kualitas tanpa label klinis.

#### Dibanding Nisha (2023) – DL Abnormality:

* Nisha:

  * Buang sinyal rusak (signal cuts, NaN, flat) **secara manual**.
  * Label **Clear vs Corrupted** → **manual**.
  * Tidak ada modul SQA otomatis; sangat bergantung pada “curated signal”.

* Kamu:

  * **Membuat modul SQA otomatis**:

    * hard rule filter menggantikan manual buang segmen rusak,
    * hybrid SQI+VAE menggantikan manual Clear/Corrupted.
  * Mengurangi kebutuhan anotasi manual yang berat.
  * Bisa menunjukkan bahwa klasifikasi tetap bagus bahkan tanpa kurasi manual besar.

#### Dibanding paper GAN (DopplerNet + 1D-CycleGAN):

* Paper GAN:

  * Quality class (high/moderate/low/corrupted) → **penilaian visual manual oleh ahli**.
  * GAN dipakai **setelah** manual labeling kualitas.

* Kamu:

  * Tetap boleh pakai 1D-CycleGAN,
  * **tapi domain high/low quality ditentukan otomatis**:

    * menggunakan SQI_final (SQI+VAE),
    * bukan lagi berdasarkan mata manusia.
  * Novelty: *“quality-aware GAN restoration tanpa manual quality labeling”*.

---

Kalau kamu mau, next kita bisa:

* susun versi “Bab 2 Tinjauan Pustaka” dari tiga paper itu,
* dan versi paragraf untuk “Bab 3 Metodologi” yang menjelaskan pipeline FP kamu dengan bahasa formal (biar nanti gampang copy ke laporan).
