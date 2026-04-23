# build_poi_db.py
import os
import json
from deepface import DeepFace

POI_DIR = "facial_rec/poi"
OUTPUT_JSON = "demo_embedding.json"

db = []

for fname in os.listdir(POI_DIR):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(POI_DIR, fname)

    # represent() = detect + align + embedding
    reps = DeepFace.represent(
        img_path=path,
        model_name="ArcFace",
        detector_backend="retinaface"
    )

    # represent can return a list; take the first face
    # each rep is a dict with "embedding"
    if isinstance(reps, list):
        rep = reps[0]
    else:
        rep = reps

    db.append({
        "name": os.path.splitext(fname)[0],
        "image_path": path,
        "embedding": rep["embedding"]
    })

# save to disk
with open(OUTPUT_JSON, "w") as f:
    json.dump(db, f)

print(f"Saved {len(db)} POI embeddings to {OUTPUT_JSON}")
