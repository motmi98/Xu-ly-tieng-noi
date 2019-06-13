import wave
import librosa
from librosa import display
import numpy as np
import sounddevice
from scipy import signal
import matplotlib.pyplot as plt
import gc


chorus1 = 40
chorus2 = 100
song, sr = librosa.load('test.wav', sr=None)
freqs = [40, 80, 120, 180, 300]
#sounddevice.play(song)
#sounddevice.wait()
n = song.shape[0]
duration = n/sr
print(duration)
f, t, Zxx = signal.stft(song, fs=sr, nperseg=4096)
ZxxT = np.abs(Zxx.transpose())
max_indexs = []
hashes = []
distance = int(len(f)/5)
for i in range(len(ZxxT)-1):
    j = signal.find_peaks(ZxxT[i][:len(f)], distance=distance, prominence=0.5)
    max_indexs.append(j[0])
#    if i<2500:
#       print(i, t[i], j[0])
for i in range(20):
    print(t[765+i], max_indexs[765+i], t[2020+i], max_indexs[2020+i])
"""
for i in ZxxT:
    j = np.where(i == max(i))
    max_index.append(j[0][0])

for i in range(len(max_index)-5):
    distance1 = np.power(np.power(max_index[i] - max_index[i+1], 2) + 1, 0.5)
    distance2 = np.power(np.power(max_index[i] - max_index[i+2], 2) + 4, 0.5)
    distance3 = np.power(np.power(max_index[i] - max_index[i+3], 2) + 9, 0.5)
    distance4 = np.power(np.power(max_index[i] - max_index[i+4], 2) + 16, 0.5)
    distance5 = np.power(np.power(max_index[i] - max_index[i+5], 2) + 25, 0.5)
    distance = np.power(distance1, 2) + np.power(distance2, 2) + np.power(distance3, 2) + np.power(distance4, 2) + np.power(distance5, 2)
    #print(hash1, i, t[i])
    hashes.append(distance)

print(hashes[623], hashes[2004])
pos = []
for i in range(0, len(ZxxT)-5):
    pos.clear()
    for j in range((i+1), len(ZxxT)-5):
        if np.abs(hashes[i]-hashes[j]) < np.power(hashes[i], 0.5)*0.1:
            if len(pos) > 5:
                break
            else:
                pos.append([i, j])
                #if j < 2100:
                    #print(i, j, round(t[i], 3), round(t[j], 3), hashes[i], hashes[j])
gc.collect()
"""
#plt.figure(figsize=(15, 5))
#data = signal.spectrogram(song, fs=sr)
#for i in data:
#    maximum = -np.Inf
#    for j in i:
#        maximum = np.maximum(maximum, j)
#    max_list.append(np.round(maximum))
#print(max_list)
#peaks, _ = signal.find_peaks(song, prominence=0.8)
#print(peaks)

