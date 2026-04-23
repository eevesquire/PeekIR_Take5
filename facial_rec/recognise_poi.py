import os
from deepface import DeepFace

POI_DIR = "/poi"
SUSPECT = "faces/vidhya.jpg"

poi_paths = [
    os.path.join(POI_DIR, f)
    for f in os.listdir(POI_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

best = None  # we'll store (poi_path, result_dict)

for poi_path in poi_paths:
    try:
        result = DeepFace.verify(
            img1_path=SUSPECT,
            img2_path=poi_path,
            model_name="ArcFace",   # solid model
            detector_backend="retinaface"  # good detector; can omit
        )
        # result has keys like: verified, distance, threshold, model, detector_backend
        if best is None or result["distance"] < best[1]["distance"]:
            best = (poi_path, result)
    except Exception as e:
        print(f"Skipping {poi_path}: {e}")

if best is None:
    print("No faces processed at all.")
else:
    poi_path, result = best
    distance = result["distance"]
    threshold = result["threshold"]
    verified = result["verified"]


    if verified:
        print(f"Best match: {os.path.basename(poi_path)}")
        print(f"Distance: {distance:.4f} (threshold: {threshold:.4f})")
        print(f"Match: YES")
    else:
        print("No match found in folder.")
        print(f"Closest was: {os.path.basename(poi_path)}")
        print(f"Distance: {distance:.4f} (threshold: {threshold:.4f})")
