import json
import numpy as np
from deepface import DeepFace

POI_DB_JSON = "demo_embedding.json"
SUSPECT = "facial_rec/testfaces/sean.jpg"

# 1. load poi db
with open(POI_DB_JSON, "r") as f:
    poi_db = json.load(f)

# 2. embed suspect ONCE
sus_reps = DeepFace.represent(
    img_path=SUSPECT,
    model_name="ArcFace",
    detector_backend="retinaface"
)
sus_rep = sus_reps[0] if isinstance(sus_reps, list) else sus_reps
sus_emb = np.array(sus_rep["embedding"], dtype="float32")

# ArcFace cosine threshold (same as you saw earlier)
COSINE_THRESHOLD = 0.68

def cosine_distance(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return 1 - np.dot(a, b)

best_name = None
best_path = None
best_dist = 999.0

for poi in poi_db:
    poi_emb = np.array(poi["embedding"], dtype="float32")
    dist = cosine_distance(sus_emb, poi_emb)
    if dist < best_dist:
        best_dist = dist
        best_name = poi["name"]
        best_path = poi["image_path"]

print(f"Closest was: {best_name} ({best_path})")
print(f"Distance (cosine): {best_dist:.4f} (threshold: {COSINE_THRESHOLD:.4f})")

if best_dist <= COSINE_THRESHOLD:
    print("Match: YES")
else:
    print("Match: NO")
