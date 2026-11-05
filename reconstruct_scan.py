from pathlib import Path
import sar_complete as sc

SCAN_FOLDER = Path(r"C:\orcad\PeekIR_Final\outputs\scan_18")   # your scan's folder

args = sc.build_parser().parse_args([
    "-W", "150", "-H", "150",
    "--y-step", "1",                 # must match the scan
    "--zstart", "670", "--zend", "720", "--zstep", "2",
    "--viz-html-root", r"C:\orcad\PeekIR_Final\visualizations",
])

sc.run_reconstruction(args, SCAN_FOLDER)