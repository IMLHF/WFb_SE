import numpy as np
import scipy
import scipy.signal
import librosa
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
# ffmpeg -i 20180829_191732_mono.wav -ar 16000 -ac 1 20180829_191732_mono16k.wav


def picture_spec(spec, name):
  '''
  spec:[time,fft_dot]
  '''
  spec_t = spec.T
  plt.figure(figsize=(12, 5))
  plt.pcolormesh(spec_t)
  plt.title('STFT Magnitude')
  plt.xlabel('Time')
  plt.ylabel('Frequency')
  plt.savefig(name+".jpg")
  print("write pic "+name)
  plt.close()


def picture_wave(wave_t, name, framerate):
  nframes = np.shape(wave_t)[0]
  _time = np.arange(0, nframes)*(1.0 / framerate)
  plt.plot(_time, wave_t)
  plt.xlabel("Time")
  plt.ylabel("Amplitude")
  plt.title("Single channel wave")
  plt.grid(True)
  plt.savefig(name+".jpg")
  # plt.show()
  print("write pic "+name)
  plt.close()


def magnitude_spectrum_librosa_stft(signal, NFFT, overlap):
  '''
  return mag_spec:[time,fft_dot]
  '''
  signal = np.array(signal, dtype=np.float32)
  tmp = librosa.core.stft(signal,
                          n_fft=NFFT,
                          hop_length=NFFT-overlap,
                          window=scipy.signal.windows.hann)
  tmp = np.absolute(tmp)
  return tmp.T  # tmp.T:[time, fft_dot]


def phase_spectrum_librosa_stft(signal, NFFT, overlap):
  '''
  return theta:[time, angle_vector]
  '''
  signal = np.array(signal, dtype=np.float)
  tmp = librosa.core.stft(signal,
                          n_fft=NFFT,
                          hop_length=NFFT-overlap,
                          window=scipy.signal.windows.hann)
  tmp = np.angle(tmp)
  return tmp.T  # tmp.T:[time, angle_vector]


def librosa_istft(magnitude_complex, NFFT, overlap):
  '''
  magnitude_complex:[time,frequence]
  '''
  tmp = librosa.core.istft(magnitude_complex.T,
                           win_length=NFFT,
                           hop_length=NFFT-overlap,
                           window=scipy.signal.windows.hann)
  return tmp # wave


def griffin_lim(spec, NFFT, overlap, max_iter, mixed_wav):
  '''
  spec:[time,fft_dot]
  '''
  # y = np.random.random(np.shape(librosa_istft(spec,
  #                                             NFFT=NFFT,
  #                                             overlap=overlap,)))
  y = mixed_wav
  for i in range(max_iter-1):
    stft_matrix = librosa.core.stft(y,
                                    n_fft=NFFT,
                                    hop_length=NFFT-overlap,
                                    window=scipy.signal.windows.hann)
    stft_matrix = stft_matrix.T
    # stft_matrix = spec * (stft_matrix / np.maximum(np.abs(stft_matrix),1e-10))
    stft_matrix = spec * np.exp(1j*np.angle(stft_matrix))
    y = librosa.core.istft(stft_matrix.T,
                           win_length=NFFT,
                           hop_length=NFFT-overlap,
                           window=scipy.signal.windows.hann)
  return y


def stft_for_reconstruction(x, fft_size, hopsamp):
    """Compute and return the STFT of the supplied time domain signal x.
    Args:
        x (1-dim Numpy array): A time domain signal.
        fft_size (int): FFT size. Should be a power of 2, otherwise DFT will be used.
        hopsamp (int):
    Returns:
        The STFT. The rows are the time slices and columns are the frequency bins.
    """
    window = np.hanning(fft_size)
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    return np.array([np.fft.rfft(window*x[i:i+fft_size])
                     for i in range(0, len(x)-fft_size, hopsamp)])


def istft_for_reconstruction(X, fft_size, hopsamp):
    """Invert a STFT into a time domain signal.
    Args:
        X (2-dim Numpy array): Input spectrogram. The rows are the time slices and columns are the frequency bins.
        fft_size (int):
        hopsamp (int): The hop size, in samples.
    Returns:
        The inverse STFT.
    """
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    window = np.hanning(fft_size)
    time_slices = X.shape[0]
    len_samples = int(time_slices*hopsamp + fft_size)
    x = np.zeros(len_samples)
    for n,i in enumerate(range(0, len(x)-fft_size, hopsamp)):
        x[i:i+fft_size] += window*np.real(np.fft.irfft(X[n]))
    return x


def griffin_lim_v2(magnitude_spectrogram, fft_size, overlap, iterations, mixed_wave=None):
    """Reconstruct an audio signal from a magnitude spectrogram.
    Given a magnitude spectrogram as input, reconstruct
    the audio signal and return it using the Griffin-Lim algorithm from the paper:
    "Signal estimation from modified short-time fourier transform" by Griffin and Lim,
    in IEEE transactions on Acoustics, Speech, and Signal Processing. Vol ASSP-32, No. 2, April 1984.
    Args:
        magnitude_spectrogram (2-dim Numpy array): The magnitude spectrogram. The rows correspond to the time slices
            and the columns correspond to frequency bins.
        fft_size (int): The FFT size, which should be a power of 2.
        hopsamp (int): The hope size in samples.
        iterations (int): Number of iterations for the Griffin-Lim algorithm. Typically a few hundred
            is sufficient.
    Returns:
        The reconstructed time domain signal as a 1-dim Numpy array.
    """
    tmp_mag = librosa.core.stft(mixed_wave,
                                n_fft=fft_size,
                                hop_length=fft_size-overlap,
                                window=scipy.signal.windows.hann).T
    mixed_wave = istft_for_reconstruction(tmp_mag, fft_size, fft_size-overlap)

    hopsamp = fft_size - overlap
    time_slices = magnitude_spectrogram.shape[0]
    len_samples = int(time_slices*hopsamp + fft_size)
    # Initialize the reconstructed signal to noise.
    x_reconstruct = mixed_wave
    if mixed_wave is None:
      x_reconstruct = np.random.randn(len_samples)
    n = iterations # number of iterations of Griffin-Lim algorithm.
    while n > 0:
        n -= 1
        reconstruction_spectrogram = stft_for_reconstruction(x_reconstruct, fft_size, hopsamp)
        reconstruction_angle = np.angle(reconstruction_spectrogram)
        # Discard magnitude part of the reconstruction and use the supplied magnitude spectrogram instead.
        proposal_spectrogram = magnitude_spectrogram*np.exp(1.0j*reconstruction_angle)
        prev_x = x_reconstruct
        x_reconstruct = istft_for_reconstruction(proposal_spectrogram, fft_size, hopsamp)
        diff = np.sqrt(sum((x_reconstruct - prev_x)**2)/x_reconstruct.size)
        print('Reconstruction iteration: {}/{} RMSE: {} '.format(iterations - n, iterations, diff))
    return x_reconstruct
