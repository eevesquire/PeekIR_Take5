from pathlib import Path
import sar_complete as sc

SCAN_FOLDER = Path(r"C:\orcad\PeekIR_Final\outputs\front\metal\front_metal_005")   # your scan's folder

args = sc.build_parser().parse_args([
    "-W", "150", "-H", "160",
    "--y-step", "1",                 # must match the scan
    "--zstart", "415", "--zend", "510", "--zstep", "2",
    "--viz-html-root", r"C:\orcad\PeekIR_Final\visualizations",
])

sc.run_reconstruction(args, SCAN_FOLDER)