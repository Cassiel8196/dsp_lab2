from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import wave
import math

"""
from cmath import sin, cos, pi
# 可以参考，反正这个fft也能用
class FFT_pack():
    def __init__(self, _list=[], N=0):  # _list 是传入的待计算的离散序列，N是序列采样点数，对于本方法，点数必须是2^n才可以得到正确结果
        self.list = _list  # 初始化数据
        self.N = N
        self.total_m = 0  # 序列的总层数
        self._reverse_list = []  # 位倒序列表
        self.output =  []  # 计算结果存储列表
        self._W = []  # 系数因子列表
        for _ in range(len(self.list)):
            self._reverse_list.append(self.list[self._reverse_pos(_)])
        self.output = self._reverse_list.copy()
        for _ in range(self.N):
            self._W.append((cos(2 * pi / N) - sin(2 * pi / N) * 1j) ** _)  # 提前计算W值，降低算法复杂度

    def _reverse_pos(self, num) -> int:  # 得到位倒序后的索引
        out = 0
        bits = 0
        _i = self.N
        data = num
        while (_i != 0):
            _i = _i // 2
            bits += 1
        for i in range(bits - 1):
            out = out << 1
            out |= (data >> i) & 1
        self.total_m = bits - 1
        return out

    def FFT(self, _list, N, abs=True) -> list:  # 计算给定序列的傅里叶变换结果，返回一个列表，结果是没有经过归一化处理的
        #参数abs=True表示输出结果是否取得绝对值
        self.__init__(_list, N)
        for m in range(self.total_m):
            _split = self.N // 2 ** (m + 1)
            num_each = self.N // _split
            for _ in range(_split):
                for __ in range(num_each // 2):
                    temp = self.output[_ * num_each + __]
                    temp2 = self.output[_ * num_each + __ + num_each // 2] * self._W[__ * 2 ** (self.total_m - m - 1)]
                    self.output[_ * num_each + __] = (temp + temp2)
                    self.output[_ * num_each + __ + num_each // 2] = (temp - temp2)
        if abs == True:
            for _ in range(len(self.output)):
                self.output[_] = self.output[_].__abs__()
        return self.output

"""
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



def mfcc_cal(spec_x, num_filter, n):
    # s是Mel滤波器能量,n是阶数，num_filter是Mel滤波器数量,求某一帧s的DCT得到的mcff系数,
    res = 0
    for _i in range(num_filter):
        res = res + np.log(spec_x[_i]) * np.cos(np.pi * n * (2 * _i - 1) / (2 * num_filter))
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


def mel(nfilt, sample_rate):   # ipt是一帧，对每一帧mel，帧长加到了256
    NFFT = 256
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
        plt.plot(freq, fbank[m - 1, :], 'r')  # 频域滤波器波形
    plt.show()
    """
    # 将filter_bank中的0值改为最小负数，防止运算出现问题，再对每个滤波器的能量取log即得到log梅尔频谱
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    """

    return fbank

def judge(x):
    L = len(x)  # 信号长度
    N = np.power(2, np.ceil(np.log2(L)))  # 下一个最近二次幂
    for i in range (0,math.ceil(math.log2(N))):
            new_list = np.append(x, 0)
    return new_list

def main():
    testfile = 'test.wav'
    record(testfile)
    fs, data = wavfile.read(testfile)
    print(fs)
    print(len(data))
    print(data)
    inc = 100
    wlen = 200
    # 输入x,然后进行分帧，分成x[i]
    x = enframe(data, wlen, inc)  # 分完帧了
    num_frame = int((len(data)-wlen+inc)/inc)  # 每一段的帧数
    s_x = np.zeros((num_frame,wlen))  # s_x的每一行是一帧
    s_x = s_x.tolist()
    x = x.tolist()
    num_melfilter = 40  # 滤波器的个数
    #每一帧长200，补为256，s_x 是分帧FFT之后的
    for i in range(num_frame):
        for j in range(199,255):
            x[i].append(0)
        s_x[i] = myfft((x[i]))
    array = np.asarray(s_x)
    eg_frames = np.power(np.real(s_x), 2) + np.power(np.imag(s_x), 2)  #能量谱
    h_mel = mel(num_melfilter, 8000)
    S_m = np.zeros((num_frame, 40))
    #  print(h_mel.shape,eg_frames.shape,array.shape) 40x129  470x256  470x256
    for i in range(num_frame):
        S_m[i] = np.dot(eg_frames[i][0:129], h_mel.T)
    mfcc_x = np.zeros(num_frame)
    for i in range(num_frame):
        mfcc_x[i] = mfcc_cal(S_m[i], num_melfilter, 12)
    print(mfcc_x)  # 输出


if __name__ == '__main__':
    main()
