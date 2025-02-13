import numpy as np

def fft_interpolate(fs, amp_array, time_array, k=0):
    """
    元の離散信号をFFTしてフーリエ係数を求め，直交基底の線形結合として表現することで補間する．
    また，その線形結合の微分も容易に求まる．
    ----------
    ## Parameters
        fs : float
            サンプリングレート Hz
        amp_array : numpy array
            もとの信号
        time_array : numpy array
            補間したい時間配列
        k : int
            微分係数
    ----------
    ## returuns
        amp_array_interp : numpy array
            補間した振幅
    """
    N = len(amp_array)
    freq_array = np.fft.rfftfreq(N,d=1/fs)
    fft_amp_array = 2.0 / N * np.fft.rfft(amp_array) #N//2までFFT
    fft_amp_array[0] = fft_amp_array[0]/2.0

    fft_amp_array = 1j**k * fft_amp_array #微分する

    an = np.real(fft_amp_array)
    bn = - np.imag(fft_amp_array)
    
    N_2 = len(freq_array) #N//2
    amp_array_interp = np.zeros_like(time_array,dtype=np.float64)
    # k回微分すると(2πft)**kがそれぞれの基底にかかる
    for i in range(N_2):
        amp_array_interp += (2*np.pi*freq_array[i])**k * (an[i] * np.cos(2*np.pi*freq_array[i]*time_array) \
                    + bn[i] * np.sin(2*np.pi*freq_array[i]*time_array))

    return amp_array_interp
N = 81
fs =20#サンプリングレート[Hz]
t = np.arange(start=0.0, stop=N/fs, step=1.0/fs) #時間軸
test_signal = np.cos(2*np.pi*21.0*t) \
            + np.cos(2*np.pi*52.0*t) + np.cos(2*np.pi*70.0*t)

import matplotlib.pyplot as plt

# FFTによる補間
interp_signal = fft_interpolate(fs, test_signal, t, k=0)

# 元信号とFFT補間結果をプロットする
plt.figure(figsize=(10, 6))
plt.plot(t, test_signal, label="Original Signal", linestyle="--")
plt.plot(t, interp_signal, label="FFT Interpolated Signal", alpha=0.7)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("FFT Interpolation Result")
plt.legend()
plt.grid(True)
plt.show()