import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Import modul lokal
from preprocessing import preprocess_time_vector, segment_echo_signal, load_tcd_data
from spectrogram_utils import generate_spectrogram
from tracing import spectrogram_tracing
from postprocessing import postprocess_cbfv

def run_single_signal_extraction(filepath, filename):
    print(f"Memulai ekstraksi sinyal untuk: {filename}")

    # Parameter (sesuai dengan implementasi yang ada)
    fc = 1.75e6   # Frekuensi pembawa
    ss = 1540     # Kecepatan suara
    angle = 0     # Sudut Doppler
    size_sp_filt = [3, 3] # Ukuran kernel filter median untuk spektrogram

    full_path = os.path.join(filepath, filename)

    if not os.path.exists(full_path):
        print(f"Error Fatal: File data tidak ditemukan di: {full_path}")
        print("Pastikan file tersebut ada di direktori yang sama dengan skrip ini.")
        return

    print(f"Memuat data dari: {full_path}")
    raw_echo_data = load_tcd_data(full_path)

    # 1. Pra-pemrosesan data waktu
    t_processed = preprocess_time_vector(raw_echo_data['t'].values)

    # 2. Segmentasi sinyal gema (biasanya menjadi segmen 60 detik)
    tcd_echo_segments = segment_echo_signal(raw_echo_data['I'].values, raw_echo_data['Q'].values, t_processed)
    
    # Ambil segmen pertama untuk demonstrasi
    if not tcd_echo_segments['IQ']:
        print("Tidak ada segmen yang valid setelah segmentasi. Mungkin data terlalu pendek.")
        return

    iq_segment = tcd_echo_segments['IQ'][0]
    fs_echo = tcd_echo_segments['fs'][0]
    t_echo = tcd_echo_segments['t'][0]
    ix_low_freq = tcd_echo_segments['ix_low_freq'][0]

    print(f"Memproses segmen pertama (panjang: {len(iq_segment)} sampel, Fs: {fs_echo:.2f} Hz)")

    # 3. Hasilkan Spektrogram
    f_sp, t_sp, sp_seg, fs_sp = generate_spectrogram(iq_segment, fs_echo, t_echo)

    # 4. Konversi frekuensi Doppler ke kecepatan (cm/s)
    v_spectrogram = 100 * f_sp * ss / (2 * fc * np.cos(np.deg2rad(angle)))

    # 5. Pemrosesan Citra Spektrogram (kompresi logaritmik & filter median)
    sp_log = np.log2(sp_seg + 1e-9)
    sp_log = sp_log / np.max(sp_log)
    sp_filt = signal.medfilt2d(sp_log, kernel_size=size_sp_filt)

    # 6. Penelusuran Envelope Adaptif (Algoritma utama)
    # Sesuaikan indeks pengecualian Otsu untuk segmen saat ini
    ix_exclude_otsu_mapped = []
    if len(ix_low_freq) > 0:
        ratio = fs_sp / fs_echo
        ix_exclude_otsu_mapped = np.round(ix_low_freq * ratio).astype(int)

    cbfv_raw, cbfv_sqi, img_bw, _ = spectrogram_tracing(v_spectrogram, sp_filt, fs_sp, ix_exclude_otsu_mapped)

    # 7. Pasca-pemrosesan (filter Butterworth & Median)
    cbfv_final_smooth = postprocess_cbfv(cbfv_raw, fs_sp)

    print("Ekstraksi sinyal selesai. Plotting hasil...")

    # Visualisasi
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 10))

    # Extent untuk imshow: [kiri, kanan, bawah, atas]
    if t_sp.size > 0 and v_spectrogram.size > 0:
        extent = [t_sp[0], t_sp[-1], v_spectrogram[0], v_spectrogram[-1]]
    else:
        extent = [0, 1, 0, 1] # Fallback if empty

    # 1. Spektrogram dengan Traces
    ax[0].imshow(sp_filt, aspect='auto', origin='lower', extent=extent, cmap='gray', vmin=0.1, vmax=0.95)
    ax[0].plot(t_sp, cbfv_final_smooth, 'r', linewidth=2, label='CBFV Envelope Halus')
    ax[0].set_ylabel('CBFV [cm/s]')
    ax[0].legend(loc='upper right')
    ax[0].set_title(f'Spektrogram TCD & Penelusuran Envelope untuk {filename}')

    # 2. Envelope CBFV Mentah dan Akhir
    ax[1].plot(t_sp, cbfv_raw, 'b', linewidth=1.5, label='CBFV Mentah')
    ax[1].plot(t_sp, cbfv_final_smooth, 'r--', linewidth=2, label='CBFV Akhir (Dihaluskan)')
    ax[1].set_ylabel('CBFV [cm/s]')
    ax[1].legend(loc='upper right')
    ax[1].set_title('Perbandingan Envelope CBFV Mentah dan Dihaluskan')

    # 3. Indeks Kualitas Sinyal (SQI)
    ax[2].plot(t_sp, 100 * cbfv_sqi, 'g', linewidth=1.5)
    ax[2].set_ylabel('SQI [%]')
    ax[2].set_xlabel('Waktu [s]')
    ax[2].set_title('Indeks Kualitas Sinyal (SQI)')

    plt.tight_layout()
    plt.show()

# --- Cara menjalankan contoh ini ---

if __name__ == "__main__":

    # Dapatkan direktori tempat skrip ini berada

    script_dir = os.path.dirname(os.path.abspath(__file__))

    

    # File data berada di direktori yang SAMA dengan skrip

    data_filename = 'Healthy_Subjects_Recording_8.txt'

    data_full_path = os.path.join(script_dir, data_filename)

    

    print(f"Mencari data di: {data_full_path}")

    

    # Jalankan ekstraksi sinyal

    # Kita berikan direktori dan nama file secara terpisah sesuai signature fungsi

    run_single_signal_extraction(script_dir, data_filename)
