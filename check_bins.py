from pathlib import Path

SCAN = Path(r"C:\orcad\PeekIR_Final\outputs\scan_12")   # your 50x50 scan folder
ROWS, FRAMES = 51, 60                                   # 50x50 mm at default settings
EXPECTED = FRAMES * 512 * 4 * 2 * 2                     # 491,520 bytes

for r in range(1, ROWS + 1):
    f = SCAN / f"row_{r}_Raw_0.bin"
    if not f.exists():
        print(f"row {r:3}: MISSING")
    elif f.stat().st_size != EXPECTED:
        size = f.stat().st_size
        print(f"row {r:3}: {size:>9} bytes  ({size / EXPECTED:.0%} of expected)")
print("done - rows not listed are complete")