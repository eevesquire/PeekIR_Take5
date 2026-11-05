from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SCAN = Path(r"C:\orcad\PeekIR_Final\outputs\scan_12")
SAMPLES, FS, SLOPE, C, NFFT, TI = 512, 9121e3, 63.343e12, 299792458.0, 1024, 4.5225e-10

profiles = []
for f in sorted(SCAN.glob("row_*_Raw_0.bin")):
    d = np.fromfile(f, dtype=np.int16)
    d = d[: len(d) // 8 * 8]
    rx1 = d[0::8] + 1j * d[4::8]               # receiver 1, same as the reconstruction
    n = len(rx1) // SAMPLES
    if n == 0:
        continue
    chirps = rx1[: n * SAMPLES].reshape(n, SAMPLES)
    profiles.append(np.abs(np.fft.fft(chirps, NFFT, axis=1)).mean(axis=0))

p = np.mean(profiles, axis=0)
depth_mm = (np.arange(NFFT) * C * FS / (2 * SLOPE * NFFT) - C * TI / 2) * 1000
keep = (depth_mm > 0) & (depth_mm < 1500)

plt.plot(depth_mm[keep], 20 * np.log10(p[keep]))
plt.xlabel("Depth Z (mm, same scale as reconstruction)")
plt.ylabel("Mean magnitude (dB)")
plt.title("Range profile")
plt.grid(True)
plt.show()