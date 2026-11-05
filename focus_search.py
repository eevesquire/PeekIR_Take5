from pathlib import Path
import numpy as np
from scipy.fft import fft
import sar_reconstruct as sr
import sar_coordinator as sc

SCAN = Path(r"C:\orcad\PeekIR_Final\outputs\scan_18")   # your 120x120 scan folder
W, H, Y_STEP = 150, 150, 1           # width, height, row step you scanned with (mm)
SPEED, PERIOD, SAMPLES = 25, 40, 512  # pipeline defaults
Z_MIN, Z_MAX, Z_STEP = 400, 900, 5    # depths to test (mm)

frames = int((W / SPEED) * 1000 / PERIOD) + 10
rows = sc.num_rows_from_height_step(H, Y_STEP)
dx = SPEED * PERIOD / 1000
n_move = int(round(W / dx))

raw = sr.stack(SAMPLES, frames, rows, 1, str(SCAN), lambda y: f"row_{y}_Raw_0.bin",
               snake=True, n_move=n_move)
raw_fft = fft(raw, n=1024, axis=0)

C, FS, K, TI = 299792458.0, 9121e3, 63.343e12, 4.5225e-10
results = []
for z in np.arange(Z_MIN, Z_MAX + Z_STEP, Z_STEP):
    k = int(round(K / FS * (2 * z * 1e-3 / C + TI) * 1024))
    mf = sr.create_matched_filter(1024, dx, 1024, Y_STEP, z)
    img, _, _ = sr.reconstruct_sar_image(raw_fft[k], mf, dx, Y_STEP, W, H)
    mag = np.abs(img)
    sharpness = (mag**4).sum() / (mag**2).sum() ** 2   # higher = more focused
    results.append((z, mag.max(), sharpness))
    print(f"z = {z:5.0f} mm   peak = {mag.max():.3e}   sharpness = {sharpness:.3e}")

best_peak = max(results, key=lambda r: r[1])[0]
best_sharp = max(results, key=lambda r: r[2])[0]
print(f"\nBest by peak brightness: {best_peak:.0f} mm")
print(f"Best by sharpness:       {best_sharp:.0f} mm")