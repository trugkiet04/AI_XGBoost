import os
import json
import time
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.feature_extraction import FeatureHasher

DATA_DIR = r"B:\Code\V3.0\data\ember"
RANDOM_STATE = 42

TRAIN_FILES = [
    os.path.join(DATA_DIR, "train_features_0.jsonl"),
    os.path.join(DATA_DIR, "train_features_1.jsonl"),
    os.path.join(DATA_DIR, "train_features_2.jsonl"),
    os.path.join(DATA_DIR, "train_features_3.jsonl"),
    os.path.join(DATA_DIR, "train_features_4.jsonl"),
    os.path.join(DATA_DIR, "train_features_5.jsonl"),
]

TEST_FILE = os.path.join(DATA_DIR, "test_features.jsonl")

MODEL_OUT = os.path.join(DATA_DIR, "ember2017_xgb_full_gpu_manual.json")
REPORT_OUT = os.path.join(DATA_DIR, "ember2017_xgb_full_gpu_report.txt")


def safe_list(x, n=0):
    if isinstance(x, list):
        return x
    return [0] * n


def vectorize_ember_raw(obj):
    feats = []

    # 1) histogram: 256
    histogram = safe_list(obj.get("histogram", []), 256)
    if len(histogram) < 256:
        histogram = histogram + [0] * (256 - len(histogram))
    feats.extend(histogram[:256])

    # 2) byteentropy: 256
    byteentropy = safe_list(obj.get("byteentropy", []), 256)
    if len(byteentropy) < 256:
        byteentropy = byteentropy + [0] * (256 - len(byteentropy))
    feats.extend(byteentropy[:256])

    # 3) strings
    strings = obj.get("strings", {})
    printabledist = safe_list(strings.get("printabledist", []), 96)
    if len(printabledist) < 96:
        printabledist = printabledist + [0] * (96 - len(printabledist))

    feats.extend([
        float(strings.get("numstrings", 0)),
        float(strings.get("avlength", 0)),
        float(strings.get("entropy", 0)),
        float(strings.get("paths", 0)),
        float(strings.get("urls", 0)),
        float(strings.get("registry", 0)),
        float(strings.get("MZ", 0)),
    ])
    feats.extend(printabledist[:96])

    # 4) general
    general = obj.get("general", {})
    feats.extend([
        float(general.get("size", 0)),
        float(general.get("vsize", 0)),
        float(general.get("has_debug", 0)),
        float(general.get("exports", 0)),
        float(general.get("imports", 0)),
        float(general.get("has_relocations", 0)),
        float(general.get("has_resources", 0)),
        float(general.get("has_signature", 0)),
        float(general.get("has_tls", 0)),
        float(general.get("symbols", 0)),
    ])

    # 5) header
    header = obj.get("header", {})
    coff = header.get("coff", {})
    optional = header.get("optional", {})

    header_tokens = []
    for k, v in coff.items():
        header_tokens.append(f"coff_{k}={v}")
    for k, v in optional.items():
        header_tokens.append(f"opt_{k}={v}")

    header_hashed = (
        FeatureHasher(n_features=128, input_type="string")
        .transform([header_tokens])
        .toarray()[0]
    )
    feats.extend(header_hashed)

    # 6) section
    section = obj.get("section", {})
    sections = section.get("sections", [])
    entry = section.get("entry", "")

    feats.extend([
        float(len(sections)),
        float(sum(1 for s in sections if s.get("size", 0) == 0)),
        float(sum(1 for s in sections if s.get("name", "") == "")),
    ])

    section_size_pairs = [(s.get("name", ""), float(s.get("size", 0))) for s in sections]
    section_entropy_pairs = [(s.get("name", ""), float(s.get("entropy", 0))) for s in sections]
    section_vsize_pairs = [(s.get("name", ""), float(s.get("vsize", 0))) for s in sections]

    section_size_hashed = (
        FeatureHasher(n_features=50, input_type="pair")
        .transform([section_size_pairs])
        .toarray()[0]
    )
    section_entropy_hashed = (
        FeatureHasher(n_features=50, input_type="pair")
        .transform([section_entropy_pairs])
        .toarray()[0]
    )
    section_vsize_hashed = (
        FeatureHasher(n_features=50, input_type="pair")
        .transform([section_vsize_pairs])
        .toarray()[0]
    )
    entry_hashed = (
        FeatureHasher(n_features=50, input_type="string")
        .transform([[entry]])
        .toarray()[0]
    )

    entry_props = []
    for s in sections:
        if s.get("name", "") == entry:
            entry_props.extend(s.get("props", []))

    props_hashed = (
        FeatureHasher(n_features=50, input_type="string")
        .transform([entry_props])
        .toarray()[0]
    )

    feats.extend(section_size_hashed)
    feats.extend(section_entropy_hashed)
    feats.extend(section_vsize_hashed)
    feats.extend(entry_hashed)
    feats.extend(props_hashed)

    # 7) imports
    imports = obj.get("imports", {})
    import_tokens = []
    if isinstance(imports, dict):
        for dll, funcs in imports.items():
            import_tokens.append(f"dll:{dll}")
            if isinstance(funcs, list):
                for fn in funcs:
                    import_tokens.append(f"imp:{dll}:{fn}")

    imports_hashed = (
        FeatureHasher(n_features=256, input_type="string")
        .transform([import_tokens])
        .toarray()[0]
    )
    feats.extend(imports_hashed)

    # 8) exports
    exports = obj.get("exports", [])
    if not isinstance(exports, list):
        exports = []

    exports_hashed = (
        FeatureHasher(n_features=128, input_type="string")
        .transform([exports])
        .toarray()[0]
    )
    feats.extend(exports_hashed)

    return np.asarray(feats, dtype=np.float32)


def load_ember_jsonl(file_list):
    X = []
    y = []

    for file_path in file_list:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Khong tim thay file: {file_path}")

        print(f"[+] Loading {file_path}")
        kept_before = len(y)

        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                obj = json.loads(line)

                label = obj.get("label", -1)
                if label not in [0, 1]:
                    continue

                try:
                    vec = vectorize_ember_raw(obj)
                except Exception as e:
                    print(f"[!] Skip line {i} in {file_path}: {e}")
                    continue

                X.append(vec)
                y.append(label)

                if i % 50000 == 0:
                    print(f"    processed {i} lines... kept_total={len(y)}")

        kept_after = len(y)
        print(f"    kept from this file: {kept_after - kept_before}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    return X, y


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true, y_pred, digits=4, zero_division=0
        ),
    }


def metrics_to_text(name, metrics):
    cm = metrics["confusion_matrix"]
    lines = [
        f"===== {name} =====",
        f"Accuracy : {metrics['accuracy']}",
        f"Precision: {metrics['precision']}",
        f"Recall   : {metrics['recall']}",
        f"F1-score : {metrics['f1']}",
        f"ROC-AUC  : {metrics['roc_auc']}",
        "",
        "Confusion Matrix:",
        str(cm),
        "",
        "Classification Report:",
        metrics["classification_report"],
        "",
    ]
    return "\n".join(lines)


def main():
    start_time = time.time()

    print("[1] Load full train files (0 -> 5)...")
    X_full, y_full = load_ember_jsonl(TRAIN_FILES)
    print("Full train shape:", X_full.shape, y_full.shape)

    if len(X_full) == 0:
        raise ValueError("Khong load duoc mau nao tu full train set.")

    print("[2] Split train / validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_full,
        y_full,
        test_size=0.2,
        stratify=y_full,
        random_state=RANDOM_STATE
    )

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val  :", X_val.shape, "y_val  :", y_val.shape)

    print("[3] Load official test...")
    X_test, y_test = load_ember_jsonl([TEST_FILE])
    print("X_test :", X_test.shape, "y_test :", y_test.shape)

    print("[4] Create DMatrix...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": 8,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "tree_method": "hist",
        "device": "cuda",
        "seed": RANDOM_STATE,
    }

    evals = [(dtrain, "train"), (dval, "val")]

    print("[5] Train full model on GPU...")
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=300,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=10
    )

    print("[6] Evaluate validation...")
    val_prob = model.predict(dval)
    val_pred = (val_prob >= 0.5).astype(np.int32)
    val_metrics = compute_metrics(y_val, val_pred, val_prob)

    print(metrics_to_text("VALIDATION", val_metrics))

    print("[7] Evaluate official test...")
    test_prob = model.predict(dtest)
    test_pred = (test_prob >= 0.5).astype(np.int32)
    test_metrics = compute_metrics(y_test, test_pred, test_prob)

    print(metrics_to_text("TEST", test_metrics))

    print("[8] Save model...")
    model.save_model(MODEL_OUT)
    print(f"Saved model to: {MODEL_OUT}")

    elapsed = time.time() - start_time

    print("[9] Write report to txt...")
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write("EMBER 2017 FULL TRAIN REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Data directory : {DATA_DIR}\n")
        f.write(f"Model output   : {MODEL_OUT}\n")
        f.write(f"Random state   : {RANDOM_STATE}\n")
        f.write(f"Train files    : {len(TRAIN_FILES)} files (0 -> 5)\n")
        f.write(f"Train shape    : {X_full.shape}\n")
        f.write(f"Train split    : X_train={X_train.shape}, X_val={X_val.shape}\n")
        f.write(f"Test shape     : {X_test.shape}\n")
        f.write(f"Best iteration : {model.best_iteration}\n")
        f.write(f"Best score     : {model.best_score}\n")
        f.write(f"Elapsed sec    : {elapsed:.2f}\n\n")

        f.write("PARAMS\n")
        f.write("-" * 60 + "\n")
        for k, v in params.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

        f.write(metrics_to_text("VALIDATION", val_metrics))
        f.write("\n")
        f.write(metrics_to_text("TEST", test_metrics))
        f.write("\n")

    print(f"Saved report to: {REPORT_OUT}")


if __name__ == "__main__":
    main()