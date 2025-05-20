# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 10:51:30 2025

@author: vibroacustica

https://docs.scipy.org/doc/scipy-1.15.2/tutorial/signal.html#tutorial-stft

https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.signal.ShortTimeFFT.html#scipy.signal.ShortTimeFFT


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
from matplotlib import ticker
from matplotlib import colormaps
from DataReader import DWDataReader

blockSize = 10000
overlap = 0.75 # 0.75 significa 75%
fmax = 1600 # Frequencia maxima do grafico. A frequencia de amostragem é definida pelo dt.

dreader = DWDataReader()
dreader.open_data_file("TestAbsorcao_Medicao")
data = dreader.get_measurements_as_dataframe()

arr = np.asarray(data['AI A-1']).flatten()

nt = int(len(arr))

time =  np.zeros(shape=(nt,))
p001 =  np.zeros(shape=(nt,))
p002 =  np.zeros(shape=(nt,))

time = np.asarray(data['Time']).flatten()
p001 = np.asarray(data['AI A-1']).flatten()
p002 = np.asarray(data['AI A-2']).flatten()

dt = time[1]

fsampling = 1/dt

w = np.hanning(blockSize)
blockShift = int(blockSize*(1-overlap))

SFT = ShortTimeFFT(w, hop=blockShift, fs=fsampling, scale_to='magnitude')

Sx001 = SFT.stft(p001)  # perform the STFT
Sx001 = 2*Sx001 # Verificar se isso é necessário.

Sx002 = SFT.stft(p002)  # perform the STFT
Sx002 = 2*Sx002 # Verificar se isso é necessário.

Sx001_mag   = np.abs(Sx001)
Sx001_phase = np.angle(Sx001)
Sx001_real  = Sx001.real
Sx001_imag  = Sx001.imag

Sx002_mag   = np.abs(Sx002)
Sx002_phase = np.angle(Sx002)
Sx002_real  = Sx002.real
Sx002_imag  = Sx002.imag


nBlock = np.size(Sx001_mag,1)

blockTime = blockSize*dt
df = 1/blockTime
nf = int(np.floor(blockSize*0.5))+1

freq = np.linspace(0,fsampling*0.5,nf)
t_lo, t_hi = SFT.extent(nt)[:2]
t1 = SFT.lower_border_end[0] * SFT.T
t2 = SFT.upper_border_begin(nt)[0] * SFT.T

timeBlock = np.linspace(t_lo,t_hi,nBlock)

Sx001_mag_average   = np.zeros(shape=(nf,))
Sx001_phase_average = np.zeros(shape=(nf,))
Sx001_real_average  = np.zeros(shape=(nf,))
Sx001_imag_average  = np.zeros(shape=(nf,))

Sx002_mag_average   = np.zeros(shape=(nf,))
Sx002_phase_average = np.zeros(shape=(nf,))
Sx002_real_average  = np.zeros(shape=(nf,))
Sx002_imag_average  = np.zeros(shape=(nf,))


for i in range(0,nf):
    Sx001_mag_average[i]=np.average(Sx001_mag[i,:])
    Sx001_phase_average[i]=np.average(Sx001_phase[i,:])
    Sx001_real_average[i]=np.average(Sx001_real[i,:])
    Sx001_imag_average[i]=np.average(Sx001_imag[i,:])
    Sx002_mag_average[i]=np.average(Sx002_mag[i,:])
    Sx002_phase_average[i]=np.average(Sx002_phase[i,:])
    Sx002_real_average[i]=np.average(Sx002_real[i,:])
    Sx002_imag_average[i]=np.average(Sx002_imag[i,:])


H12_real  = np.zeros(shape=(nf,))
H12_imag  = np.zeros(shape=(nf,))
H12_mag  = np.zeros(shape=(nf,))
H12_phase  = np.zeros(shape=(nf,))
for i in range(0,nf):
    x2 = Sx001_real_average[i]
    x1 = Sx002_real_average[i]
    y2 = Sx001_imag_average[i]
    y1 = Sx002_imag_average[i]
    H12_real[i] = (x1*x2+y1*y2)/(x2*x2+y2*y2)
    H12_imag[i] = (y1*x2-x1*y2)/(x2*x2+y2*y2)
    H12_mag[i] = np.sqrt(H12_real[i]*H12_real[i] + H12_imag[i]*H12_imag[i])
    H12_phase[i] = np.arctan2(H12_imag[i],H12_real[i])

plt.rcParams["figure.figsize"] = [20,4]

# plt.plot(time,p001,'-',label='p001')
# plt.grid()
# plt.minorticks_on()
# plt.grid(which='major',linestyle='-',color='lightgray',linewidth='1.0')
# plt.grid(which='minor',linestyle=':',color='lightgray',linewidth='1.0')
# plt.show()

figure, ([ax1,ax2]) = plt.subplots(2,constrained_layout=True) 
ax1.plot(time,p001,'-',label='Magnitude')
ax2.plot(time,p002,'-',label='Magnitude')
plt.show()


plt.rcParams["figure.figsize"] = [15,10]

figure, ([ax1,ax2,ax3,ax4]) = plt.subplots(4,constrained_layout=True) 
figure.suptitle("Custom Title")
ax1.semilogy(freq,Sx001_mag_average,'-',label='p001')
ax1.semilogy(freq,Sx002_mag_average,'-',label='p002')
ax1.set_ylabel("Magnitude")
ax1.grid()
ax1.minorticks_on()
ax1.grid(which='major',linestyle='-',color='lightgray',linewidth='1.0')
ax1.grid(which='minor',linestyle=':',color='lightgray',linewidth='1.0')
#ax1.set_xlim([0, fmax]) 
#ax1.set_ylim([0.01, 100]) 
#ax1.set_xticks(np.arange(0, 360, step=30))
#ax1.set_xticks(np.arange(0, 360, step=10),minor=True)
ax1.legend(ncols=1)
ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:6.4f}"))
ax2.plot(freq,Sx001_phase_average*180/np.pi,'-',label='Phase')
ax2.set_xlim([0, fmax]) 
ax3.plot(freq,Sx001_real_average,'-',label='Real')
ax3.set_xlim([0, fmax]) 
ax4.plot(freq,Sx001_imag_average,'-',label='Imaginary')
ax4.set_xlim([0, fmax]) 
plt.show()


figure, ([ax1,ax2,ax3,ax4]) = plt.subplots(4,constrained_layout=True) 
ax1.semilogy(freq,H12_mag,'-',label='H12')
ax2.plot(freq,H12_phase*180/np.pi,'-',label='H12')
ax3.plot(freq,H12_real,'-',label='H12')
ax4.plot(freq,H12_imag,'-',label='H12')
plt.show()


Sx001_mag_log = np.log(Sx001_mag)
colormax = np.max(Sx001_mag_log)
colormin = np.min(Sx001_mag_log)
"""
https://matplotlib.org/stable/users/explain/colors/colormaps.html
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pcolormesh.html
"""
# largura = int(297*0.10)
# altura = int(210*0.03)
# plt.rcParams["figure.figsize"] = [largura,altura]
# #plt.pcolormesh(timeBlock, freq, Sx_mag_log, shading='gouraud',cmap='hot_r')
# #plt.pcolormesh(timeBlock, freq, Sx_mag_log, shading='gouraud',cmap='plasma')
# #plt.pcolormesh(timeBlock, freq, Sx_mag_log, shading='gouraud',cmap='YlOrRd')
# plt.pcolormesh(timeBlock, freq, Sx_mag_log, shading='gouraud',cmap='plasma',vmin=0.6*colormin,vmax=0.6*colormax)
# plt.xlim(0,time[nt-1])
# plt.xlabel("Time [s]")
# plt.ylabel("Frequency [Hz]")
# plt.show()

def calculate_absorption_coeficent(freq, H12_real, H12_imag, l, s, c=343):
        """
        Calcula o coeficiente de absorção acústica a partir de H (complexo).

        Parâmetros:
        - freq: array de frequências (Hz)
        - H: array de valores complexos H = G12 / G11
        - l: distância da amostra até o centro do microfone mais próximo (em metros)
        - s: distância entre os centros dos microfones (em metros)
        - c: velocidade do som (m/s), padrão: 343 m/s

        Retorna:
        - alpha: array com o coeficiente de absorção
        """
        freq = np.asarray(freq)
        #H = np.asarray(H)
        
        Hr = H12_real
        Hi = H12_imag
        Hr2 = Hr**2
        Hi2 = Hi**2

        k = 2 * np.pi * freq / c
        Hr2 = Hr**2
        Hi2 = Hi**2

        D = 1 + Hr2 + Hi2 - 2 * (Hr * np.cos(k * s) + Hi * np.sin(k * s))

        Rr = (2 * Hr * np.cos(k * (2 * l + s)) - np.cos(2 * k * l) - (Hr2 + Hi2) * np.cos(2 * k * (l + s))) / D
        Ri = (2 * Hr * np.sin(k * (2 * l + s)) - np.sin(2 * k * l) - (Hr2 + Hi2) * np.sin(2 * k * (l + s))) / D

        alpha = 1 - Rr**2 - Ri**2
        
        # Ensure values are within physical limits (sometimes numerical issues can cause values >1)
        alpha = np.clip(alpha, -1, 1)
        
        return alpha



alpha = calculate_absorption_coeficent(freq, H12_real, H12_imag, 0.1, 0.05)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(freq, alpha, marker='o', color='blue', label='Coef. de absorção')
ax.set_xscale('linear')
ax.set_ylim(bottom=-1, top=1)
ax.set_xlim(left=0,right=1600)
ax.set_xlabel("Frequência (Hz)")
ax.set_ylabel("Coeficiente de Absorção")
ax.set_title("Coeficiente de Absorção vs Frequência")
ax.grid(True)
ax.legend()
fig.tight_layout()
plt.show()
