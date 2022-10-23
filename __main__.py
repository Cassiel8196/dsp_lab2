from scipy.io import wavfile
# import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import wave


# from scipy.fftpack import dct


def record(file):
    #  录音并保存为名为file.wav，会录制一段5秒的声音在windows上存储为wav格式的音频
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 3
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("start recording...")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("end")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return 0


def enframe(data, wlen, inc):
    # x为data wlen窗长 inc为帖移（一般取窗长的0~0.5
    nx = len(data)  # 获取整个音频的长度
    nlen = wlen
    if inc is None:
        inc = nlen
    nf = (nx - nlen + inc) // inc  # 获取在进行窗口移动和帧长所有可以分成的长度个数
    frameout = np.zeros((nf, nlen))  # 设置数组大小和长度 nf为可分开的长度，nlen时表示每段的长度
    indf = np.multiply(inc, np.array([i for i in range(nf)]))
    for i in range(nf):
        frameout[i, :] = data[indf[i]:indf[i] + nlen]
    if isinstance(wlen, list):
        frameout = np.multiply(frameout, np.array(wlen))
    return frameout


# 想法：自己写一个spec_x=myfft(x)，还没写[5]DCT和h_mel，别的写完了
def mfcc_cal(spec_x, num_filter, n):
    # s是Mel滤波器能量,n是阶数，num_filter是Mel滤波器数量,求某一帧s的DCT得到的mcff系数
    res = 0
    for _i in range(num_filter):
        res = res + np.log(spec_x[_i]) * np.cos(np.pi * n * (2 * num_filter - 1) / (2 * num_filter))
    return np.sqrt(2 / num_filter) * res


def energy_cal(spec_x, num_filter, _h_mel):  # spec_x是一帧的信号的频谱
    spec_energy = np.power(np.real(spec_x), 2) + np.power(np.imag(spec_x), 2)
    spec_mel = np.zeros(num_filter)
    for m in range(num_filter):
        spec_mel[m] = sum(np.multiply(spec_energy, _h_mel[m]))  # 相乘后求和
    return spec_mel


# 递归FFT，利用分治思想的dft
def myfft(x):  # 需要在主函数里面提前把x补到2^L点长
    _n = len(x)
    _m = int(len(x) / 2)
    s = np.zeros(_n, dtype=complex)
    if _n == 2:
        s[0] = x[0] + x[1]
        s[1] = x[0] - x[1]
    else:
        _x1 = x[0::2]
        _x2 = x[1::2]
        _s1 = myfft(_x1)
        _s2 = myfft(_x2)
        for r in range(_m):
            s[r] = _s1[r] + np.exp(-2j * np.pi * r / _n) * _s2[r]
            s[r + _m] = _s1[r] - np.exp(-2j * np.pi * r / _n) * _s2[r]
    return s


def mel(s_x, nfilt, sample_rate):
    NFFT = 512
    # sample_rate  采样率
    # s_x是求过fft之后的
    mag_frames = np.absolute(s_x)  # 求幅值
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # 功率谱

    # 滤波器组 Filter Banks
    """
    三角滤波器，nfilt是滤波器个数，应用于功率谱以提取频带。 
    赫兹（f）和梅尔（m）之间的转换：
                m = 2595log10(1+f/700)
                f = 700*(10^(m/2595)-1)
    """
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Hz -> Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # 等间隔分刻度
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Mel -> Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)  # bin储存的是刻度对应的傅里叶变换点数
    # fbank 存储的是每个滤波器的值
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    W2 = int(NFFT / 2 + 1)  # fs/2内对应的FFT点数
    df = sample_rate / NFFT
    freq = []  # 采样频率值
    for n in range(0, W2):
        freqs = int(n * df)
        freq.append(freqs)
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    """
        plt.plot(freq, fbank[m - 1, :], 'r')  # 频域滤波器波形
    plt.show()
    # 将filter_bank中的0值改为最小负数，防止运算出现问题，再对每个滤波器的能量取log即得到log梅尔频谱
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    """
    """
    # 画热力图
    plt.title("filter_banks")
    plt.imshow(np.flipud(filter_banks.T), cmap=plt.cm.jet, aspect=0.1,
               extent=[0, filter_banks.shape[1], 0, filter_banks.shape[0]])
    plt.xlabel("Frames", fontsize=14)
    plt.ylabel("Dimension", fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.savefig('filter_banks.png')
    plt.show()
    """
    return fbank


def main():
    testfile = 'test.wav'
    record(testfile)
    fs, data = wavfile.read(testfile)
    print(fs)
    print(data)
    inc = 100
    wlen = 200
    x = enframe(data, wlen, inc)  # 分完帧了
    # t1 = np.linspace(0, 5 * np.pi, 200)  # 时间坐标
    # x1 = np.sin(2 * np.pi * t1)  # 正弦函数
    # 输入x,然后进行分帧，分成x[i]
    num_frame = 200  # 每一段的帧数
    # s_x = np.zeros(num_frame)
    num_melfilter = 40  # 滤波器的个数
    # for i in range(len(x)):
    #    s_x[i] = fft_recurrence(x[i])  # 求fft变换
    s_x = myfft(x)  # Magnitude of the FFT
    h_mel = mel(s_x, num_melfilter, 8000)
    s1_x = np.zeros((num_frame, num_melfilter))
    mfcc_x = np.zeros(num_frame)

    for i in range(num_frame):
        s1_x[i] = energy_cal(s_x[i], num_melfilter, h_mel)
        mfcc_x[i] = mfcc_cal(s1_x[i], num_melfilter, 12)

    print(mfcc_x)  # 输出


if __name__ == '__main__':
    main()
