import numpy as np
from scipy import signal as sgl
from matplotlib import pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq, fftshift, ifft
from datetime import datetime

f = open("log_files/20220505_14-28-32")
data = f.read()
data = data.split("\\n")

x = []
y = []
z = []
j2 = []
time_buf = []
time = []

for i, el in enumerate(data):
    if el == "'":
        break
    k = el.split(" ")
    for j, ks in enumerate(k):
        if "'b'" in ks:
            k[j] = ks.replace("'b'", "")
    # a = k[1]
    start_number = 9500
    finish_number = 75000
    if i == start_number:
        j2_start = float(k[6])
        time_buf.append(k[11].split(":"))
        start_time = int(time_buf[i - start_number][0]) * 60 * 60 + int(
            time_buf[i - start_number][1]) * 60 + int(
            time_buf[i - start_number][2])
    if finish_number > i > start_number:
        j2.append(float(k[6]) - j2_start)
        x.append(k[2])
        y.append(k[3])
        z.append(k[4])
        time_buf.append(k[11].split(":"))
        time_actual = int(time_buf[i - start_number][0]) * 60 * 60 + int(
            time_buf[i - start_number][1]) * 60 + int(
            time_buf[i - start_number][2]) - start_time
        time.append(i - start_number)

x = np.array(x)
x = x.astype(np.float)
j2 = np.array(j2)
j2 = sgl.medfilt(j2, 7)

# smooth_cof = 0.5
# j2_avg = []
# j2_avg.append(j2[0])
#
# while i < len(j2):
#     window_average = round((smooth_cof * j2[i]) + (1 - smooth_cof) * j2_avg[-1], 5)
#     j2_avg.append(window_average)

number_series = pd.Series(j2)

moving_averages = round(number_series.ewm(alpha=0.01, adjust=True).mean(), 5)
j2_avg = moving_averages.tolist()
j2_avg = np.array(j2_avg)

# j2 = j2.astype(np.float)
time = np.array(time)
time = time.astype(np.float)

freq = 100
sample_rate = 850

ts = 1.0 / sample_rate
t = np.arange(0, 1, ts)

y = np.sin(2 * np.pi * freq * t)

freq = 350

y += np.sin(2 * np.pi * freq * t)

yf = fft(y)
N = len(yf)
n = np.arange(N)
T = 1 / sample_rate
freq = n / T
xf = fftfreq(N, T)[:N//2]

plt.figure("j2 torques log2", figsize=(12, 6))
plt.subplot(121)

plt.stem(xf, 2.0/N * np.abs(yf[0:N//2]), 'b',
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
# plt.xlim(0, 2.5)

plt.subplot(122)
# plt.plot(time, j2, j2_avg)
plt.plot(t, y, 'r')
plt.xlabel('Recieved packet')
plt.ylabel('Torque Nm')
plt.tight_layout()
plt.show()
