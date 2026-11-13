from pathlib import Path
import sar_complete as sc

SCAN_FOLDER = Path(r"C:\orcad\PeekIR_Final\outputs\test\check\test_check_002")   # your scan's folder

args = sc.build_parser().parse_args([
    "-W", "300", "-H", "280",
    "--y-step", "2",                 # must match the scan
    "--zstart", "300", "--zend", "450", "--zstep", "1",
    "--viz-html-root", r"C:\orcad\PeekIR_Final\visualizations",
])

sc.run_reconstruction(args, SCAN_FOLDER)