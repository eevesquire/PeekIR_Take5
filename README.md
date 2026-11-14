# PeekIR Dashboard System

`peekir_dashboard.py` is the central control surface for the full PeekIR app. From one UI, operators can:

- run mmWave SAR scans and 3D reconstruction, with scan position, depth range and training labels set in the form
- load and view previously reconstructed interactive scans, with detected metal outlined on them
- see a **threat assessment** for every scan: IR + mmWave metal detection, size-based threat level, and gun / knife shape recognition
- measure and calibrate IR metal detection against a no-metal reference
- save labelled training data (IR, mmWave, or both) for future machine-learning work
- monitor RGB and IR camera streams from the Raspberry Pi
- run face matching against a wanted-person database
- run an experimental AI metal / no-metal check on a captured IR frame (**Capture IR**)
- pair and monitor a Bluetooth metal detector (metal state + battery)

---

## Core Components

| Component | Runs on | Purpose |
|---|---|---|
| `peekir_dashboard.py` | Windows/laptop | Main FastAPI dashboard + web UI + orchestration |
| `sar_complete.py` | Windows/laptop | One-shot SAR pipeline: capture (`sar_coordinator.py`) + reconstruct (`sar_reconstruct.py`) |
| `sar_batch.py` | Windows/laptop | Command-line batch scanning (many scans in a row, labels, IR snapshots, training export) |
| `sar_coordinator.py` | Windows (with mmWave Studio) | TCP orchestrator between radar client and Pi gantry client |
| `sar_scan_tcp.lua` | mmWave Studio host | Radar/DCA1000 client controlled by coordinator commands |
| `move.py` | Raspberry Pi | Gantry control: `move.py tcp` (scan client), `move.py origin` (homing), emergency-stop monitor |
| `pi_camera_server.py` | Raspberry Pi | RGB/IR camera stream, frame API and raw IR (temperature) frames used by the dashboard |
| `metal_detector.py` | Windows/laptop | BLE integration for metal-detection notifications and battery |
| `sar_reconstruct.py` | Windows/laptop | SAR reconstruction and visualization artifact generation |
| `ir_metal_size.py` | Windows/laptop | IR metal detection: size of metal on the mannequin → threat level |
| `mmwave_metal_size.py` | Windows/laptop | mmWave metal detection from the reconstructed volume → threat level; combined IR + mmWave verdict; viewer outlines |
| `shape_classify.py` | Windows/laptop | Shape of each detected piece → gun / knife / suspicious object (shared by IR and mmWave) |
| `training_export.py` | Windows/laptop | Saves labelled, text-free training data to `training_data/` |
| `ir_preprocess.py` | Windows/laptop | Helper for importing the previous team's colourised IR images (`training_export.py import-legacy-ir`) |
| `ir_ai_metal.py` + `ir_knife_model.pth` | Windows/laptop | AI metal / no-metal check (small CNN) used by the **Capture IR** button (experimental) |
| `train_ir_model.py` | any PC with the dataset | Trains the AI model (`ir_knife_model.pth`) from `knife_dataset/{knife,no_knife}/*.png` |
| `requirements-dashboard.txt` / `-ai.txt` / `-face.txt` | Windows/laptop | Python packages: core / AI check / face matching (all optional except core) |

`ir_metal_size.py`, `mmwave_metal_size.py`, `shape_classify.py`, `training_export.py`, `ir_ai_metal.py` and the model file must sit in the same folder as `peekir_dashboard.py`. **First-time installation: see [SETUP.md](SETUP.md).** If `shape_classify.py` is missing, gun/knife labels switch off and the threat level comes from size only.

---

## Full App Workflow

### 1) mmWave Data Collection (from dashboard)

In **mmWave data collection**, the user enters:

- `width_mm`, `height_mm`, `scan_name`
- **X start**, **Y start**, **Y step** (mm): gantry start position and row spacing (X/Y start must be ≥ 0)
- **Z start**, **Z end** (mm): reconstruction depth range
- **View**, **Label**, **Metal?**, **Weapon type**, **Object**, **True cm²** and **Save for training**: labels for the training data (all optional)

Empty fields use the `sar_complete.py` defaults. The last values used are remembered by the browser.

When they press **Start scan**, `peekir_dashboard.py` starts:

```bash
python sar_complete.py -W <width_mm> -H <height_mm> --scan-name <scan_name> --viz-html-root visualizations \
    -o outputs [-x <x_start>] [-y <y_start>] [--y-step <mm>] [--zstart <mm>] [--zend <mm>] \
    --ir-sync-file outputs/.ir_pending/<scan_name>.go --home-after-scan
```

`sar_complete.py` then:

1. Homes the gantry (`move.py origin` over SSH)
2. Tells the dashboard the gantry is out of the way (`ready_for_ir`) and waits while the dashboard takes the **"before" IR snapshot** and measures it (max 60 s)
3. Launches scan capture through `sar_coordinator.py` and waits for completion
4. Homes the gantry again, so the **"after" IR snapshot** isn't blocked either
5. Launches `sar_reconstruct.py` on the captured data (saves `sar_volume.npz` and the Plotly viewer)
6. Emits progress messages the dashboard consumes for live status/progress bar updates

After the process exits, the dashboard:

1. takes the "after" IR snapshot;
2. writes `outputs/<scan_name>/scan_meta.json` (scan settings + labels);
3. measures the mmWave volume and pushes the combined threat result to the **Threat assessment** panel;
4. exports the scan to `training_data/` if **Save for training** is on.

### 2) SAR Control Chain

`sar_coordinator.py` is the real-time orchestrator. It accepts two persistent TCP clients:

- `RADAR` (`sar_scan_tcp.lua`)
- `GANTRY` (`move.py tcp` on Pi)

Per row, coordinator logic performs:

1. `ARM <row_file>` to radar/DCA1000
2. Start gantry movement (`MOVE ...`)
3. Wait for `MOTOR_STARTED` (+ optional stabilization)
4. Trigger `CAPTURE`
5. Wait for both `CAPTURE_DONE` and `MOVE_COMPLETE`
6. Continue in snake/raster pattern

This deterministic TCP timing is what keeps radar captures aligned with aperture position.

**Failures and the emergency stop.** If anything goes wrong mid-scan, the scan **ends with no further movement**:
- `run_scan` raises `ScanAborted`. Anything on the gantry side (an `ERROR` reply, a disconnect or a timeout) raises `GantryStopped`.
- `sar_complete.py` then skips the post-scan homing and the reconstruction, and exits with code 1. The dashboard shows "GANTRY STOPPED (emergency stop?)" and marks the scan failed: no "after" IR, no training export.
- `sar_batch.py` ends the whole batch on a gantry stop, with no retries and no re-homing.

On the Pi, `move.py` watches the e-stop button (GPIO 24):
- When it is pressed, the motors stop, `ERROR EMERGENCY STOP` is sent to the coordinator, and the client exits.
- While it is held, `move.py` refuses to start any movement, including the next scan's homing.
- A limit switch hit during a scan move is reported as an error.
- The motor PWM is switched off on every way the program can end (including a dropped SSH session).

Release the e-stop before the next scan. The next scan homes first, as usual.

### 3) SAR Reconstruction and 3D Imaging Logic

`sar_reconstruct.py` converts row `.bin` files into spatial imagery by:

1. **Range FFT** on FMCW ADC data (extract depth-frequency information)
2. **Depth slicing** over Z (`zstart` to `zend`, or single `zindex`)
3. For each Z slice, **aperture-domain reconstruction** (`mf`, `fista`, or `bpa`)
4. Build a volumetric stack across Z
5. Export:
   - XY max-intensity projection
   - XZ and YZ projections
   - interactive Plotly Z-slice HTML viewer
   - `sar_volume.npz` (with `--save-npz`, on by default in `sar_complete.py`): volume, max-intensity and depth maps, and the mm axes. The threat detection uses this file.
   - optional GIF / optional 3D scatter

Why this enables 3D SAR with this setup:

- The XY gantry synthesizes a larger aperture by moving the radar spatially.
- FMCW chirps provide range/depth sensitivity (Z dimension).
- Coherent combination of echoes across aperture positions + depth focusing reconstructs volumetric structure, which is then visualized as sliceable 3D data.

---

## Threat Detection (IR + mmWave)

Threat detection is **classical, rule-based computer vision**, not a trained model. Every rule and cutoff is written and tuned by hand, so it works without training data. The scans are also saved as labelled training data so a future machine-learning model can be trained and compared against this baseline.

### IR (`ir_metal_size.py`)

Bare metal acts like a mirror in thermal IR. With a heating pad on the mannequin, metal reads **colder** than the warm body.

1. **Set reference** with NO metal on the mannequin (raw temperature frames from the Pi).
2. Each measurement is compared to the reference:
   - drift and uneven warming are removed;
   - very hot spots (heating pad, lights) are ignored;
   - changes larger than the sensor noise are kept;
   - small mannequin movements are tolerated.
3. The changed regions are cleaned up and outlined, then measured in cm² using the calibration.
4. **Calibrate** once per camera setup, with a metal piece of known size facing the camera.

### mmWave (`mmwave_metal_size.py`)

No calibration is needed, because the reconstruction grid is already in mm.

1. The max-intensity image from `sar_volume.npz` is smoothed by 2 mm.
2. **Background** = the scan's median brightness.
3. **Metal** = spots at least **4×** brighter than the background. Spots weaker than half the strongest one are dropped as echoes.
4. Each piece's edge goes at half of its own brightness, but never below 35% of the strongest return.
5. The area is computed from the grid cell size (1 × 2 mm with the usual settings).

### Threat levels

| Level | Largest piece |
|---|---|
| NONE | no metal |
| LOW | < 25 cm² |
| MEDIUM | 25–150 cm² |
| HIGH | ≥ 150 cm² |

The cutoffs are shared by IR and mmWave and can be changed with `python ir_metal_size.py levels --low 25 --high 150`.

### Shapes: gun / knife / suspicious object (`shape_classify.py`)

Each piece's outline is measured at any rotation, in cm:

- **Gun** = L shape: the piece lies along two adjacent sides of its box and leaves the rest empty (demo prop: 8 × 5 in L panel, 1 in arms).
- **Knife** = long, thin, solid piece. It is told apart from short rectangles by *length − width* (≥ 7.5 cm), which sensor blur doesn't change much (demo prop: two 3 × 1 in rectangles taped end to end).
- **Everything else** = "suspicious object" (the exact shape is still saved).
- A gun or knife raises the threat level to **HIGH** whatever its size.
- IR needs calibration for knife detection (it works in cm). Keep pieces at least ~1 cm apart: closer pieces merge into one outline in mmWave.

### Combined verdict

- **CONFIRMED**: both sensors see metal; level = the higher of the two.
- **POSSIBLE (IR only / mmWave only)**: only one sensor sees metal.
- **CLEAR**: neither sees metal.

The weapon headline reads, e.g., "GUN detected (IR + mmWave)" or "possible KNIFE (IR only)". mmWave only sees its W × H scan window, while IR sees the whole camera view, so "IR only" can simply mean the metal is outside the scanned area.

---

## Dashboard Features

### mmWave Visualization + Threat Assessment

- The dashboard lists available HTML viewers from `visualizations/`.
- Users can load completed scans into the embedded interactive viewer.
- **Outlines on viewer**: the detected metal is drawn on the slice viewer, with each piece labelled (e.g. "KNIFE · 42 cm²") and the threat level under the title. Uncheck to see the plain viewer.
- **Threat assessment panel**:
  - the combined verdict, level and weapon headline;
  - an IR card and an mmWave card, each with the objects found and their sizes, the threat level from size alone, and whether a gun/knife raised it, plus the annotated image. Hover over a card's text for technical notes.
  - a dropdown with every scan in `outputs/` that has IR or mmWave results, newest first.
- The panel updates automatically when a dashboard scan finishes, when a new scan folder appears (checked every 8 s, so `sar_batch.py` scans show up too), and when a viewer is loaded (matched to its scan folder by file name).
- mmWave results are measured on demand, and again whenever `sar_volume.npz` or a settings file is newer than the saved result.

### RGB Camera + Face Recognition

- Dashboard displays live RGB feed from `pi_camera_server.py` (`/api/camera/stream`).
- On **Capture RGB**, it pulls one frame and runs local matching with:
  - ArcFace embedding (`DeepFace`)
  - cosine-distance comparison against precomputed wanted-person embeddings (`facial_rec/demo_embedding.json`)
- If matched under threshold, UI surfaces wanted person match immediately with annotated capture.

### IR Camera + Metal Size / Threat Level

- Dashboard displays live IR feed from `pi_camera_server.py` (`/api/ir/stream`).
- **Set reference**: capture with NO metal on the mannequin.
- **Measure metal**: compares a fresh capture to the reference and shows the threat level, the pieces (gun / knife / suspicious object, size) and an annotated image.
- **Calibrate…**: put a metal piece of known size on the mannequin, facing the camera, and enter its size (e.g. `10x15` cm).
- **Show difference**: per-pixel change vs the reference, with tuning advice, for when metal is missed.
- **Training data (IR only)**: saves the current reference + a fresh capture as a labelled IR sample (view, metal, weapon type, object, true area).
- Retake the reference just before each scan or measurement. Results get noisy when it is more than ~10 minutes old.

### Capture IR: AI metal check (experimental)

- **Capture IR** takes one IR frame from the Pi and runs it through the team's small CNN (`ir_ai_metal.py` + `ir_knife_model.pth`). It shows "AI check (experimental): METAL / no metal · metal score N%" and the frame.
- The preprocessing matches `train_ir_model.py`: the colourised frame is converted to grayscale, resized to 160×120 and scaled to 0–1. The model's "knife" label means metal.
- It is a second opinion only: it gives no location or size, and it **does not change the threat level or verdict**.
- The model loads with `torch.load(..., weights_only=True)`, so nothing in the file is executed. A retrained model file is picked up on the next capture, without a restart.
- Requires PyTorch (CPU build is enough): see `requirements-ai.txt`.
- Known limits:
  - The input is an auto-scaled colour image, not temperatures.
  - The current model was tested with a random per-image split, so its reported accuracy is optimistic.
  - Check it on fresh captures with and without metal before relying on it.
- `PEEKIR_IR_DETECTOR=yolo` switches the button back to the previous team's YOLO gun model. That needs `ultralytics` and their `weapon_detection/.../best.pt`, and is not expected to work on the heated-mannequin setup.

### Bluetooth Metal Detector Integration

- Dashboard pairs/monitors BLE device through `metal_detector.py`.
- Incoming notifications are streamed over WebSocket to the UI.
- UI displays:
  - connection state (connecting/connected/error/stopped)
  - metal state (`HIGH` / `LOW`)
  - battery percentage
- Visual alerting is triggered when metal is detected.

---

## Installing

Full step-by-step instructions (SSH key, Pi, firewall, mmWave Studio, first IR setup, troubleshooting) are in **[SETUP.md](SETUP.md)**. In short:

- **Python:** 64-bit **3.12** recommended (3.11–3.13 work; TensorFlow, needed only for face matching, has no packages for 3.14, and neither TensorFlow nor PyTorch exists for 32-bit Python).
- **Packages:**

  | File | Contents | Needed for |
  |---|---|---|
  | `requirements-dashboard.txt` | fastapi, uvicorn, websockets, httpx, bleak, numpy, pillow, scipy, matplotlib, plotly, opencv-python | everything (core) |
  | `requirements-ai.txt` | torch, pillow, numpy | Capture IR AI check |
  | `requirements-face.txt` | deepface (pulls in TensorFlow) | face matching only |

- Install into the project environment with `python -m pip install -r <file>`. If the environment was made with **uv** (it has no pip), use `uv pip install -r <file>` instead.
- PyTorch CPU build: `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- Optional features switch themselves off when their packages are missing:
  - no `deepface` → no face matching;
  - no `torch` → Capture IR shows an install hint;
  - no `shape_classify.py` → no gun/knife labels.

## Running the Dashboard

```bash
python peekir_dashboard.py
```

Run it from the project environment. In VS Code, select `.venv\Scripts\python.exe` with *Python: Select Interpreter*, so the terminal and the ▶ button both use it.

Alternative:

```bash
uvicorn peekir_dashboard:app --host 127.0.0.1 --port 8080
```

The dashboard is typically run on the Windows/laptop machine adjacent to mmWave Studio.

### IR setup per camera position (details in SETUP.md)

1. Put the mannequin in place with the heating pad on, and start the dashboard.
2. If needed, turn the body mask off: `python ir_metal_size.py levels --body-mask off`. The body mask finds the mannequin from the warmest area, so the heating pad and lights can make it hide the metal.
3. **Set reference** (no metal), then **Calibrate…** with a piece of known size.
4. Before the demo, do one mmWave scan with **no metal**, to confirm the bare mannequin reads CLEAR.

The detection settings are stored in `ir_metal_state/` and are committed to the repo: `settings.json` (IR), `mmwave_settings.json`, and `shape_settings.json` once a shape setting has been changed. The reference frames (`ir_metal_state/*.npz`) are not committed: they are specific to one setup and moment.

---

## Important Runtime Notes

- `peekir_dashboard.py` can auto-start `pi_camera_server.py` over SSH (configurable via env vars).
- SAR outputs are written under `outputs/<scan_name>/`. Dashboard scans also get:
  - `ir_start.*`, `ir_end.*` (snapshots + results) and `ir_reference.npz`;
  - `mmwave_result.json` / `.png`;
  - `scan_meta.json`.
- Interactive SAR HTML files for dashboard loading are written to `visualizations/`.
- Scan names must be unique across both `outputs/` and `visualizations/`. The threat panel identifies scans by folder name, so folder names should also be unique across subfolders of `outputs/`.
- The mmWave result depends on the metal being inside the scan window: if a piece touches the scan border, move X/Y start or enlarge the scan.
- Large data (`outputs/`, `training_data/`, `visualizations/`, `knife_dataset/`, `*.bin`, `*.npz`) is excluded by `.gitignore`. Share it separately, e.g. as a zip on a cloud drive. The detection settings (`ir_metal_state/*.json`) and the AI model (`ir_knife_model.pth`, ~150 KB) are committed.

---

## Key Environment Variables

**Pi and SSH**
- `PI_BASE_URL` (default `http://10.245.61.148:9000`)
- `PI_SSH_AUTO_START` (`1`/`0`)
- `PI_SSH_HOST`, `PI_SSH_USER`, `PI_SSH_PROJECT_DIR`, `PI_SSH_VENV`

**Dashboard**
- `PEEKIR_HOST`, `PEEKIR_PORT` (default `127.0.0.1:8080`)
- `PEEKIR_OPEN_BROWSER`, `PEEKIR_BROWSER_DELAY`
- `PEEKIR_DATA_ROOT`

**Face matching**
- `PEEKIR_POI_DB`
- `PEEKIR_FACE_COSINE_THRESHOLD`, `PEEKIR_FACE_WARMUP`

**Capture IR**
- `PEEKIR_IR_DETECTOR`: `ai` (default, team CNN) or `yolo` (previous team's gun model)
- `PEEKIR_IR_AI_MODEL`: path to the AI model file (default: `ir_metal_model.pth` or `ir_knife_model.pth` next to the dashboard)
- `PEEKIR_IR_GUN_WEIGHTS`, `PEEKIR_IR_GUN_CONF`: YOLO weights / confidence (only with `yolo`)

**Threat detection**
- `PEEKIR_METAL_STATE`: folder with detection settings and the IR reference (default `./ir_metal_state`)
- `PEEKIR_SCAN_IR` (`1`/`0`): IR snapshots before/after dashboard scans
- `PEEKIR_SCAN_IR_SECONDS`: IR capture length per snapshot (default 2)
- `PEEKIR_IR_AFTER_HOMING` (`1`/`0`): take the "before" snapshot after the gantry is homed (default on)
- `PEEKIR_HOME_AFTER_SCAN` (`1`/`0`): home the gantry right after the capture (default on)
- `PEEKIR_THREAT_ROOTS`: extra folders to search for scans (separated by `;` on Windows)

**Training data**
- `PEEKIR_TRAINING_DIR`: training data folder (default `./training_data`)

---

## Command-Line Tools

### Direct SAR CLI (optional, without dashboard)

You can still run SAR capture/reconstruction directly:

```bash
python sar_complete.py -W 150 -H 150 --scan-name scan_test
```

or manually split orchestration and reconstruction:

```bash
python sar_coordinator.py
python sar_reconstruct.py --folder outputs/scan_0 --plotly
```

### Batch scanning (`sar_batch.py`)

```bash
python sar_batch.py --count 1 --view front --label metal -x 50 -y 0 -W 300 -H 280 --y-step 2 \
    --zstart 300 --zend 450 --reconstruct plotly --object Lpanel --weapon-type gun
```

Takes IR snapshots after homing, measures each scan and exports training data. Useful options:
- `--pause`: change labels between scans
- `--no-ir`: skip the IR snapshots
- `--no-training`: skip the training export

### IR metal detection (`ir_metal_size.py`)

```bash
python ir_metal_size.py reference --pi http://<pi>:9000          # no metal on the mannequin
python ir_metal_size.py calibrate --pi http://<pi>:9000 --width-cm 10 --height-cm 10
python ir_metal_size.py measure   --pi http://<pi>:9000
python ir_metal_size.py status                                   # current settings
python ir_metal_size.py levels --low 25 --high 150 --body-mask off
python ir_metal_size.py report outputs/ --remeasure --csv ir_report.csv
```

### mmWave metal detection (`mmwave_metal_size.py`)

```bash
python mmwave_metal_size.py measure-all outputs/                 # scans that have sar_volume.npz
python mmwave_metal_size.py reconstruct-all outputs/ --viz       # old scans: rebuild volume + viewer from row files
python mmwave_metal_size.py bake-outlines outputs/               # viewer copies with outlines -> visualizations/outlined/
python mmwave_metal_size.py levels --min-contrast 4 --depth-range 285 385
python mmwave_metal_size.py report outputs/ --csv threat_report.csv
```

For old scans without `scan_meta.json`, give the scan settings, e.g.:

```bash
python mmwave_metal_size.py reconstruct-all outputs/ --width-mm 300 --height-mm 280 --y-step 2 \
    --frames 310 --rows 141 --speed 25 --periodicity 40 --snake --zstart 300 --zend 450
```

### Shape settings (`shape_classify.py`)

```bash
python shape_classify.py settings                                # show
python shape_classify.py settings --knife-min-excess-cm 8 --weapon-level MEDIUM
python shape_classify.py settings --off                          # size-based threat only
python shape_classify.py settings --reset
```

### AI metal check (`ir_ai_metal.py`, `train_ir_model.py`)

```bash
python ir_ai_metal.py some_ir_image.png                 # METAL / NO METAL + metal score
python ir_ai_metal.py --eval knife_dataset              # accuracy, misses, false alarms on a labelled folder
python train_ir_model.py                                # retrain from knife_dataset/ (needs torch + scikit-learn)
```

`train_ir_model.py` writes `ir_knife_model.pth` next to itself. Run it in the repo folder, or copy the file there, and the dashboard uses the new model on the next capture. Keep `knife_dataset/` out of git (share it as a zip).

### Training data (`training_export.py`)

```bash
python training_export.py summary
python training_export.py export-all outputs/                    # scans not yet exported
python training_export.py capture-ir --pi http://<pi>:9000 --view front --metal yes --weapon-type knife --count 5
python training_export.py import-legacy-ir --src knife_dataset   # previous team's images -> legacy_ir/
```

`training_data/` has `paired/` (IR + mmWave), `ir_only/`, `mmwave_only/` and `legacy_ir/` sections. Each one has `front/side × metal/nometal` class folders, plus a `README.md` and an `index.csv`.

The images are clean (no text), and the raw data (temperature frames, `sar_volume.npz`) is included. `label.json` holds:
- the ground-truth labels (view, metal, `weapon_type`, object, true area);
- separately, the detector's own guesses (treat these as weak labels).

Split train/test by `scene_id`.