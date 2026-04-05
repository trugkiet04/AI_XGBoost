import os
import json
import time
import itertools
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

MODEL_OUT = os.path.join(DATA_DIR, "ember2017_xgb_full_gpu_tuned_16trial_fixed.json")
REPORT_OUT = os.path.join(DATA_DIR, "ember2017_xgb_full_gpu_tuned_16trial_fixed_report.txt")


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

        print(
            f"    done: seen={file_seen}, kept={file_kept}, skipped={file_skipped}"
        )

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


def compute_confusion_stats(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return tn, fp, fn, tp, tpr, fpr, tnr, fnr


def search_best_threshold(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)

    rows = []

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(np.int32)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        tn, fp, fn, tp, tpr, fpr, tnr, fnr = compute_confusion_stats(y_true, y_pred)

        row = {
            "threshold": float(thr),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "tpr": float(tpr),
            "fpr": float(fpr),
            "tnr": float(tnr),
            "fnr": float(fnr),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
        rows.append(row)

    best = max(rows, key=lambda r: (r["f1"], r["recall"], r["precision"], r["accuracy"]))
    top10 = sorted(
        rows,
        key=lambda r: (r["f1"], r["recall"], r["precision"], r["accuracy"]),
        reverse=True
    )[:10]

    return best, top10, rows


def threshold_rows_to_text(rows):
    lines = []
    header = (
        f"{'threshold':>9} {'accuracy':>9} {'precision':>10} {'recall':>8} "
        f"{'f1':>8} {'tpr':>8} {'fpr':>8} {'tnr':>8} {'fnr':>8} "
        f"{'tn':>8} {'fp':>6} {'fn':>6} {'tp':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in rows:
        lines.append(
            f"{r['threshold']:>9.2f} "
            f"{r['accuracy']:>9.6f} "
            f"{r['precision']:>10.6f} "
            f"{r['recall']:>8.6f} "
            f"{r['f1']:>8.6f} "
            f"{r['tpr']:>8.6f} "
            f"{r['fpr']:>8.6f} "
            f"{r['tnr']:>8.6f} "
            f"{r['fnr']:>8.6f} "
            f"{r['tn']:>8d} "
            f"{r['fp']:>6d} "
            f"{r['fn']:>6d} "
            f"{r['tp']:>8d}"
        )

    return "\n".join(lines)


def build_param_candidates():
    grid = {
        "max_depth": [6, 8],
        "eta": [0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "min_child_weight": [1, 3],
        "gamma": [0.0, 0.1],
        "reg_alpha": [0.0, 0.1],
        "reg_lambda": [1.0],
        "max_bin": [256],
    }

    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    candidates = []
    for combo in itertools.product(*values):
        p = dict(zip(keys, combo))
        candidates.append(p)

    return candidates


def candidate_score(best_thr_row, val_metrics, model):
    best_iter = int(getattr(model, "best_iteration", 0))
    return (
        val_metrics["f1"],
        val_metrics["roc_auc"],
        val_metrics["recall"],
        val_metrics["precision"],
        val_metrics["accuracy"],
        -best_thr_row["fpr"],
        best_iter,
    )


def main():
    start_time = time.time()

    print("[1] Load full train files (0 -> 5)...")
    X_full, y_full = load_ember_jsonl(TRAIN_FILES)
    print("Full train shape:", X_full.shape, y_full.shape)

    if len(X_full) == 0:
        raise ValueError("Khong load duoc sample nao. Hay kiem tra log [VECTORIZE ERROR].")

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

    if len(X_test) == 0:
        raise ValueError("Khong load duoc sample nao tu official test.")

    print("[4] Create DMatrix...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    candidates = build_param_candidates()
    print(f"[5] Total tuning candidates: {len(candidates)}")

    best_model = None
    best_params = None
    best_thr_row = None
    best_top10_thresholds = None
    best_val_metrics = None
    best_trial_index = None
    best_sort_key = None
    trial_summaries = []

    evals = [(dtrain, "train"), (dval, "val")]

    for idx, cand in enumerate(candidates, 1):
        print("\n" + "=" * 90)
        print(f"[TRIAL {idx}/{len(candidates)}] params = {cand}")

        params = {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "auc"],
            "tree_method": "hist",
            "device": "cuda",
            "seed": RANDOM_STATE,
            **cand,
        }

        try:
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=800,
                evals=evals,
                early_stopping_rounds=50,
                verbose_eval=50
            )
        except Exception as e:
            print(f"[!] Trial {idx} failed: {e}")
            trial_summaries.append({
                "trial": idx,
                "status": "failed",
                "params": params,
                "error": str(e),
            })
            continue

        val_prob = model.predict(dval)

        thr_row, top10_thresholds, _ = search_best_threshold(
            y_true=y_val,
            y_prob=val_prob,
            thresholds=np.round(np.arange(0.01, 1.00, 0.01), 2)
        )

        chosen_threshold = thr_row["threshold"]
        val_pred = (val_prob >= chosen_threshold).astype(np.int32)
        val_metrics = compute_metrics(y_val, val_pred, val_prob)

        sort_key = candidate_score(thr_row, val_metrics, model)

        summary = {
            "trial": idx,
            "status": "ok",
            "params": params,
            "best_iteration": int(getattr(model, "best_iteration", -1)),
            "best_score": float(getattr(model, "best_score", -1.0)),
            "chosen_threshold": float(chosen_threshold),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "val_auc": float(val_metrics["roc_auc"]),
            "val_fpr": float(thr_row["fpr"]),
        }
        trial_summaries.append(summary)

        print(f"Trial {idx} result:")
        print(f"  best_iteration : {summary['best_iteration']}")
        print(f"  best_score     : {summary['best_score']}")
        print(f"  threshold      : {summary['chosen_threshold']}")
        print(f"  val_accuracy   : {summary['val_accuracy']}")
        print(f"  val_precision  : {summary['val_precision']}")
        print(f"  val_recall     : {summary['val_recall']}")
        print(f"  val_f1         : {summary['val_f1']}")
        print(f"  val_auc        : {summary['val_auc']}")
        print(f"  val_fpr        : {summary['val_fpr']}")

        if best_sort_key is None or sort_key > best_sort_key:
            best_sort_key = sort_key
            best_model = model
            best_params = params
            best_thr_row = thr_row
            best_top10_thresholds = top10_thresholds
            best_val_metrics = val_metrics
            best_trial_index = idx

    if best_model is None:
        raise RuntimeError("Tat ca cac trial deu fail. Khong co model nao duoc chon.")

    print("\n" + "=" * 90)
    print(f"[6] Best trial: {best_trial_index}")
    print("Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    best_threshold = best_thr_row["threshold"]

    print(f"\nBest threshold: {best_threshold}")
    print("Best threshold row:")
    for k, v in best_thr_row.items():
        print(f"  {k}: {v}")

    print("\nTop 10 thresholds by F1:")
    print(threshold_rows_to_text(best_top10_thresholds))

    print("\n" + metrics_to_text("VALIDATION (best tuned model)", best_val_metrics))

    print("[7] Evaluate official test with best tuned model...")
    test_prob = best_model.predict(dtest)
    test_pred = (test_prob >= best_threshold).astype(np.int32)
    test_metrics = compute_metrics(y_test, test_pred, test_prob)

    print(metrics_to_text("TEST (best tuned model)", test_metrics))

    print("[8] Save best model...")
    best_model.save_model(MODEL_OUT)
    print(f"Saved model to: {MODEL_OUT}")

    elapsed = time.time() - start_time

    print("[9] Write report to txt...")
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write("EMBER 2017 FULL TRAIN TUNED 16-TRIAL FIXED REPORT\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Data directory : {DATA_DIR}\n")
        f.write(f"Model output   : {MODEL_OUT}\n")
        f.write(f"Random state   : {RANDOM_STATE}\n")
        f.write(f"Train files    : {len(TRAIN_FILES)} files (0 -> 5)\n")
        f.write(f"Train shape    : {X_full.shape}\n")
        f.write(f"Train split    : X_train={X_train.shape}, X_val={X_val.shape}\n")
        f.write(f"Test shape     : {X_test.shape}\n")
        f.write(f"Total trials   : {len(candidates)}\n")
        f.write(f"Best trial     : {best_trial_index}\n")
        f.write(f"Best iteration : {best_model.best_iteration}\n")
        f.write(f"Best score     : {best_model.best_score}\n")
        f.write(f"Best threshold : {best_threshold}\n")
        f.write(f"Elapsed sec    : {elapsed:.2f}\n\n")

        f.write("BEST PARAMS\n")
        f.write("-" * 90 + "\n")
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

        f.write("BEST THRESHOLD ROW (SELECTED BY F1 ON VALIDATION)\n")
        f.write("-" * 90 + "\n")
        for k, v in best_thr_row.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

        f.write("TOP 10 THRESHOLDS BY F1\n")
        f.write("-" * 90 + "\n")
        f.write(threshold_rows_to_text(best_top10_thresholds))
        f.write("\n\n")

        f.write(metrics_to_text("VALIDATION (best tuned model)", best_val_metrics))
        f.write("\n")
        f.write(metrics_to_text("TEST (best tuned model)", test_metrics))
        f.write("\n")

        f.write("ALL TRIAL SUMMARIES\n")
        f.write("-" * 90 + "\n")
        for row in trial_summaries:
            if row["status"] == "failed":
                f.write(f"Trial {row['trial']} FAILED\n")
                f.write(f"Params: {row['params']}\n")
                f.write(f"Error : {row['error']}\n\n")
            else:
                f.write(
                    f"Trial {row['trial']} | "
                    f"best_iteration={row['best_iteration']} | "
                    f"best_score={row['best_score']} | "
                    f"threshold={row['chosen_threshold']} | "
                    f"val_acc={row['val_accuracy']} | "
                    f"val_prec={row['val_precision']} | "
                    f"val_rec={row['val_recall']} | "
                    f"val_f1={row['val_f1']} | "
                    f"val_auc={row['val_auc']} | "
                    f"val_fpr={row['val_fpr']}\n"
                )
                f.write(f"Params: {row['params']}\n\n")

    print(f"Saved report to: {REPORT_OUT}")


if __name__ == "__main__":
    main()