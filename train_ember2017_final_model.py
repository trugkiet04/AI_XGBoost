import os
import json
import time
import numpy as np
import xgboost as xgb

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

MODEL_OUT = os.path.join(DATA_DIR, "ember2017_xgb_final_model.json")
REPORT_OUT = os.path.join(DATA_DIR, "ember2017_xgb_final_model_report.txt")

FINAL_THRESHOLD = 0.52


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

    total_seen = 0
    total_kept = 0
    total_skipped = 0

    for file_path in file_list:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Khong tim thay file: {file_path}")

        print(f"[+] Loading {file_path}")
        file_seen = 0
        file_kept = 0
        file_skipped = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                file_seen += 1
                total_seen += 1

                try:
                    obj = json.loads(line)
                except Exception as e:
                    file_skipped += 1
                    total_skipped += 1
                    print(f"[JSON ERROR] {file_path} line {i}: {e}")
                    continue

                label = obj.get("label", -1)
                if label not in [0, 1]:
                    continue

                try:
                    vec = vectorize_ember_raw(obj)
                except Exception as e:
                    file_skipped += 1
                    total_skipped += 1
                    print(f"[VECTORIZE ERROR] {file_path} line {i}: {e}")
                    continue

                X.append(vec)
                y.append(label)
                file_kept += 1
                total_kept += 1

                if i % 50000 == 0:
                    print(
                        f"    processed={i}, kept_file={file_kept}, "
                        f"skipped_file={file_skipped}, kept_total={total_kept}"
                    )

        print(f"    done: seen={file_seen}, kept={file_kept}, skipped={file_skipped}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    print(
        f"[LOAD SUMMARY] total_seen={total_seen}, total_kept={total_kept}, "
        f"total_skipped={total_skipped}, X.shape={X.shape}, y.shape={y.shape}"
    )

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

    print("[1] Load full official train (0 -> 5)...")
    X_train_full, y_train_full = load_ember_jsonl(TRAIN_FILES)
    print("X_train_full:", X_train_full.shape, "y_train_full:", y_train_full.shape)

    if len(X_train_full) == 0:
        raise ValueError("Khong load duoc sample nao tu full train.")

    print("[2] Load official test...")
    X_test, y_test = load_ember_jsonl([TEST_FILE])
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    if len(X_test) == 0:
        raise ValueError("Khong load duoc sample nao tu official test.")

    print("[3] Create DMatrix...")
    dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Best params from Trial 10
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "tree_method": "hist",
        "device": "cuda",
        "seed": RANDOM_STATE,
        "max_depth": 8,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "max_bin": 256,
    }

    print("[4] Train final model on FULL TRAIN...")
    model = xgb.train(
        params=params,
        dtrain=dtrain_full,
        num_boost_round=797,   # best_iteration from Trial 10
        evals=[(dtrain_full, "train_full")],
        verbose_eval=50
    )

    print("[5] Evaluate official test with fixed threshold...")
    test_prob = model.predict(dtest)
    test_pred = (test_prob >= FINAL_THRESHOLD).astype(np.int32)
    test_metrics = compute_metrics(y_test, test_pred, test_prob)

    print(metrics_to_text("TEST (final model)", test_metrics))

    print("[6] Save final model...")
    model.save_model(MODEL_OUT)
    print(f"Saved model to: {MODEL_OUT}")

    elapsed = time.time() - start_time

    print("[7] Write final report...")
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write("EMBER 2017 FINAL MODEL REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Data directory : {DATA_DIR}\n")
        f.write(f"Model output   : {MODEL_OUT}\n")
        f.write(f"Report output  : {REPORT_OUT}\n")
        f.write(f"Random state   : {RANDOM_STATE}\n")
        f.write(f"Train shape    : {X_train_full.shape}\n")
        f.write(f"Test shape     : {X_test.shape}\n")
        f.write(f"Fixed threshold: {FINAL_THRESHOLD}\n")
        f.write(f"Boost rounds   : 797\n")
        f.write(f"Elapsed sec    : {elapsed:.2f}\n\n")

        f.write("FINAL PARAMS\n")
        f.write("-" * 80 + "\n")
        for k, v in params.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

        f.write(metrics_to_text("TEST (final model)", test_metrics))
        f.write("\n")

    print(f"Saved report to: {REPORT_OUT}")


if __name__ == "__main__":
    main()