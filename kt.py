import json
import numpy as np

TRAIN_FILE = r"B:\Code\V3.0\data\ember\train_features_0.jsonl"

def main():
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        line = f.readline().strip()

    obj = json.loads(line)

    print("=== TOP LEVEL KEYS ===")
    print(list(obj.keys()))
    print()

    for k, v in obj.items():
        if isinstance(v, list):
            print(f"{k}: list, len={len(v)}")
        elif isinstance(v, dict):
            print(f"{k}: dict, keys={list(v.keys())[:20]}")
        else:
            print(f"{k}: {type(v).__name__} -> {v}")

    print("\n=== DETAIL CHECK ===")
    if "features" in obj:
        feats = obj["features"]
        print("features type:", type(feats).__name__)
        if isinstance(feats, list):
            print("features len:", len(feats))
        elif isinstance(feats, dict):
            print("features dict keys:", list(feats.keys())[:50])

    # thử tìm vector số chiều lớn nhất trong object
    print("\n=== NUMERIC VECTOR CANDIDATES ===")
    for k, v in obj.items():
        if isinstance(v, list) and len(v) > 10:
            ok = True
            for x in v[:20]:
                if not isinstance(x, (int, float)):
                    ok = False
                    break
            if ok:
                print(f"{k} looks like numeric vector, len={len(v)}")

if __name__ == "__main__":
    main()