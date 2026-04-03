import os
import traceback
from datetime import datetime

import numpy as np
import xgboost as xgb
from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "ember2017_xgb_full_gpu_manual.json")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# Hỗ trợ họ file PE của Windows
ALLOWED_EXTENSIONS = {
    "exe", "dll", "sys", "ocx", "scr", "cpl", "com", "efi", "drv", "mui"
}

# Cho phép cả file PE không có đuôi (nếu muốn)
ALLOW_EXTENSIONLESS_PE = True

EXPECTED_DIM = 1390

# Threshold tốt nhất tìm được từ validation theo F1
BEST_THRESHOLD = 0.51

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
model = xgb.Booster()
model.load_model(MODEL_PATH)

# =========================
# HTML
# =========================
HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EMBER2017 Localhost Inference</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background: #f7f7f7;
        }
        .box {
            background: white;
            padding: 24px;
            border-radius: 12px;
            max-width: 820px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }
        h1 { margin-top: 0; }
        input[type=file] { margin: 12px 0; }
        button {
            padding: 10px 18px;
            border: 0;
            border-radius: 8px;
            cursor: pointer;
        }
        .ok {
            margin-top: 20px;
            padding: 16px;
            border-radius: 10px;
            background: #eefaf0;
            border: 1px solid #b7e3bf;
        }
        .err {
            margin-top: 20px;
            padding: 16px;
            border-radius: 10px;
            background: #fff0f0;
            border: 1px solid #efb2b2;
            white-space: pre-wrap;
        }
        .meta {
            color: #555;
            font-size: 14px;
        }
        .subbox {
            margin-top: 14px;
            padding: 12px;
            border-radius: 8px;
            background: #f8f8f8;
            border: 1px solid #e2e2e2;
        }
        code {
            background: #f1f1f1;
            padding: 2px 6px;
            border-radius: 6px;
        }
        ul {
            margin-top: 8px;
            margin-bottom: 8px;
        }
    </style>
</head>
<body>
    <div class="box">
        <h1>EMBER2017 XGBoost Localhost</h1>
        <p class="meta">
            Model: <code>{{ model_path }}</code><br>
            Expected feature dim: <code>{{ expected_dim }}</code><br>
            Best threshold (F1 on validation): <code>{{ best_threshold }}</code><br>
            Supported PE extensions:
            <code>.exe, .dll, .sys, .ocx, .scr, .cpl, .com, .efi, .drv, .mui</code>
        </p>

        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file" accept=".exe,.dll,.sys,.ocx,.scr,.cpl,.com,.efi,.drv,.mui" required>
            <br>
            <button type="submit">Upload và Predict</button>
        </form>

        {% if result %}
            <div class="ok">
                <h3>Kết quả</h3>
                <p><b>File:</b> {{ result.filename }}</p>
                <p><b>Prediction:</b> {{ result.prediction }}</p>
                <p><b>Malware probability:</b> {{ result.probability }}</p>
                <p><b>Threshold:</b> {{ result.threshold }}</p>
                <p><b>Static Risk Score:</b> {{ result.risk_score }}/100</p>
                <p><b>Risk Level:</b> {{ result.risk_level }}</p>
                <p><b>Feature shape:</b> {{ result.feature_shape }}</p>
                <p><b>Detected type:</b> {{ result.detected_type }}</p>

                <div class="subbox">
                    <h4 style="margin-top: 0;">Risk Score Breakdown</h4>
                    <p><b>AI Score:</b> {{ result.risk_breakdown.ai_score }}/60</p>
                    <p><b>Static Score:</b> {{ result.risk_breakdown.static_score }}/40</p>
                    <ul>
                        <li><b>API / Imports Score:</b> {{ result.risk_breakdown.api_score }}/12</li>
                        <li><b>Strings Score:</b> {{ result.risk_breakdown.string_score }}/10</li>
                        <li><b>Entropy Score:</b> {{ result.risk_breakdown.entropy_score }}/8</li>
                        <li><b>Anomaly Score:</b> {{ result.risk_breakdown.anomaly_score }}/5</li>
                        <li><b>Metadata Score:</b> {{ result.risk_breakdown.meta_score }}/5</li>
                    </ul>
                </div>

                <div class="subbox">
                    <h4 style="margin-top: 0;">Detected Static Signals</h4>
                    <ul>
                        {% for item in result.detected_signals %}
                            <li>{{ item }}</li>
                        {% endfor %}
                    </ul>
                </div>
            </div>
        {% endif %}

        {% if error %}
            <div class="err">
                <h3>Lỗi</h3>
                <div>{{ error }}</div>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

# =========================
# HELPERS
# =========================
def get_file_extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()


def is_pe_file_by_header(file_path: str) -> bool:
    try:
        with open(file_path, "rb") as f:
            sig = f.read(2)
        return sig == b"MZ"
    except Exception:
        return False


def allowed_file(filename: str, temp_saved_path: str = None) -> bool:
    ext = get_file_extension(filename)

    if ext in ALLOWED_EXTENSIONS:
        return True

    if ext == "" and ALLOW_EXTENSIONLESS_PE and temp_saved_path is not None:
        return is_pe_file_by_header(temp_saved_path)

    return False


def detect_pe_type(filename: str, file_path: str) -> str:
    ext = get_file_extension(filename)

    mapping = {
        "exe": "PE Executable",
        "dll": "PE Dynamic Link Library",
        "sys": "PE System Driver",
        "ocx": "PE ActiveX Control",
        "scr": "PE Screensaver Executable",
        "cpl": "PE Control Panel Item",
        "com": "PE Command/Executable",
        "efi": "PE EFI Binary",
        "drv": "PE Driver",
        "mui": "PE Multilingual UI Resource",
    }

    if ext in mapping:
        return mapping[ext]

    if is_pe_file_by_header(file_path):
        return "PE file (extensionless or uncommon extension)"

    return "Unknown"


def try_extract_with_custom_pipeline(file_path: str) -> np.ndarray:
    try:
        from inference_preprocess import extract_features_for_model
    except Exception:
        return None

    feats = extract_features_for_model(file_path)
    if feats is None:
        raise ValueError("extract_features_for_model() trả về None")

    feats = np.asarray(feats, dtype=np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)

    return feats


def try_extract_with_ember(file_path: str) -> np.ndarray:
    try:
        import ember
    except ImportError as e:
        raise ImportError(
            "Không import được ember. Hãy cài đúng môi trường hoặc dùng pipeline custom."
        ) from e

    with open(file_path, "rb") as f:
        bytez = f.read()

    extractor = ember.PEFeatureExtractor(feature_version=1)
    feats = extractor.feature_vector(bytez)

    feats = np.asarray(feats, dtype=np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)

    return feats


def extract_features(file_path: str) -> np.ndarray:
    feats = try_extract_with_custom_pipeline(file_path)

    if feats is None:
        feats = try_extract_with_ember(file_path)

    if feats.shape[1] != EXPECTED_DIM:
        raise ValueError(
            f"Feature dimension mismatch: model cần {EXPECTED_DIM} features "
            f"nhưng extractor tạo ra {feats.shape[1]} features.\n"
            f"Bạn cần sửa preprocess để giống hệt lúc train."
        )

    return feats


def extract_raw_features(file_path: str) -> dict:
    """
    Dùng raw PE features để tính static risk score.
    """
    try:
        from ember1390_encoder import extract_raw_features_from_exe
    except Exception as e:
        raise ImportError(
            "Không import được extract_raw_features_from_exe từ ember1390_encoder.py"
        ) from e

    return extract_raw_features_from_exe(file_path)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def score_api_risk(raw_obj: dict):
    imports_obj = raw_obj.get("imports", {}) or {}
    general = raw_obj.get("general", {}) or {}

    tokens = []
    dlls = set()

    if isinstance(imports_obj, dict):
        for dll, funcs in imports_obj.items():
            dll_lower = str(dll).lower()
            dlls.add(dll_lower)
            tokens.append(dll_lower)
            if isinstance(funcs, list):
                for fn in funcs:
                    tokens.append(str(fn).lower())

    all_text = " ".join(tokens)
    signals = []
    score = 0.0

    process_injection_keywords = [
        "createremotethread", "writeprocessmemory", "virtualallocex",
        "openprocess", "queueuserapc", "ntmapviewofsection",
        "setwindowshookex", "resumeprocess"
    ]
    code_exec_keywords = [
        "winexec", "shellexecute", "createprocess", "cmd.exe",
        "powershell", "rundll32"
    ]
    keylog_keywords = [
        "getasynckeystate", "getkeystate", "setwindowshookex", "getkeyboardstate"
    ]
    network_keywords = [
        "wsastartup", "socket", "connect", "internetopen", "internetconnect",
        "httpopenrequest", "urldownloadtofile", "winhttpopen", "wininet"
    ]
    registry_keywords = [
        "regopenkey", "regsetvalue", "regcreatekey", "regqueryvalue"
    ]
    anti_debug_keywords = [
        "isdebuggerpresent", "checkremotedebuggerpresent", "ntqueryinformationprocess",
        "outputdebugstring"
    ]

    def hit_any(keywords):
        return any(k in all_text for k in keywords)

    if hit_any(process_injection_keywords):
        score += 12
        signals.append("Phát hiện dấu hiệu Process Injection / remote memory manipulation")

    if hit_any(code_exec_keywords):
        score += 10
        signals.append("Phát hiện dấu hiệu thực thi lệnh / tạo tiến trình")

    if hit_any(keylog_keywords):
        score += 9
        signals.append("Phát hiện dấu hiệu theo dõi bàn phím / keylogging")

    if hit_any(network_keywords):
        score += 7
        signals.append("Phát hiện thư viện hoặc API mạng đáng chú ý")

    if hit_any(registry_keywords):
        score += 5
        signals.append("Phát hiện thao tác registry")

    if hit_any(anti_debug_keywords):
        score += 7
        signals.append("Phát hiện dấu hiệu anti-debug / evasion")

    imports_count = int(general.get("imports", 0) or 0)
    if imports_count > 300:
        score += 2
        signals.append("Số lượng imports lớn bất thường")

    score = clamp(score, 0, 12)
    return round(score, 2), signals


def score_strings_risk(raw_obj: dict):
    s = raw_obj.get("strings", {}) or {}
    urls = float(s.get("urls", 0) or 0)
    registry = float(s.get("registry", 0) or 0)
    paths = float(s.get("paths", 0) or 0)
    mz_count = float(s.get("MZ", 0) or 0)
    entropy = float(s.get("entropy", 0) or 0)
    numstrings = float(s.get("numstrings", 0) or 0)

    score = 0.0
    signals = []

    if urls > 0:
        score += 2
        signals.append(f"Phát hiện {int(urls)} URL trong strings")

    if registry > 0:
        score += 2
        signals.append(f"Phát hiện {int(registry)} chuỗi registry")

    if paths > 10:
        score += 2
        signals.append("Số lượng path trong strings tương đối cao")

    if mz_count > 0:
        score += 2
        signals.append("Phát hiện chuỗi 'MZ' bên trong strings (có thể chứa PE/payload nhúng)")

    if entropy > 6.5:
        score += 1.5
        signals.append("Entropy của vùng strings cao")

    if numstrings > 2000:
        score += 1.5
        signals.append("Số lượng strings lớn")

    score = clamp(score, 0, 10)
    return round(score, 2), signals


def score_entropy_risk(raw_obj: dict):
    section = raw_obj.get("section", {}) or {}
    sections = section.get("sections", []) or []

    high_entropy_sections = []
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        ent = float(sec.get("entropy", 0) or 0)
        if ent > 7.0:
            high_entropy_sections.append((sec.get("name", ""), ent))

    signals = []
    if len(high_entropy_sections) >= 3:
        score = 8
    elif len(high_entropy_sections) == 2:
        score = 6
    elif len(high_entropy_sections) == 1:
        score = 3
    else:
        score = 0

    if high_entropy_sections:
        names = ", ".join([f"{name or '<unnamed>'}({ent:.2f})" for name, ent in high_entropy_sections[:5]])
        signals.append(f"Section entropy cao: {names}")

    return float(score), signals


def score_anomaly_risk(raw_obj: dict):
    section = raw_obj.get("section", {}) or {}
    sections = section.get("sections", []) or []
    general = raw_obj.get("general", {}) or {}

    score = 0.0
    signals = []

    if len(sections) == 0:
        score += 2
        signals.append("Không đọc được danh sách sections")

    zero_size_sections = sum(1 for s in sections if float(s.get("size", 0) or 0) == 0)
    if zero_size_sections > 0:
        score += min(2, zero_size_sections * 0.5)
        signals.append(f"Có {zero_size_sections} section size = 0")

    unnamed_sections = sum(1 for s in sections if str(s.get("name", "")).strip() == "")
    if unnamed_sections > 0:
        score += min(1.5, unnamed_sections * 0.5)
        signals.append(f"Có {unnamed_sections} section không có tên")

    if len(sections) > 8:
        score += 1.0
        signals.append("Số lượng sections nhiều hơn bình thường")

    imports_count = float(general.get("imports", 0) or 0)
    has_signature = float(general.get("has_signature", 0) or 0)

    avg_entropy = 0.0
    valid_entropy = [float(s.get("entropy", 0) or 0) for s in sections if isinstance(s, dict)]
    if valid_entropy:
        avg_entropy = sum(valid_entropy) / len(valid_entropy)

    if imports_count < 10 and avg_entropy > 6.5:
        score += 1.5
        signals.append("Imports rất ít nhưng entropy trung bình cao")

    if has_signature == 0 and avg_entropy > 6.8:
        score += 1.0
        signals.append("Không có chữ ký số và entropy cao")

    score = clamp(score, 0, 5)
    return round(score, 2), signals


def score_metadata_risk(raw_obj: dict):
    general = raw_obj.get("general", {}) or {}
    header = raw_obj.get("header", {}) or {}

    score = 0.0
    signals = []

    has_signature = float(general.get("has_signature", 0) or 0)
    has_debug = float(general.get("has_debug", 0) or 0)
    has_resources = float(general.get("has_resources", 0) or 0)
    symbols = float(general.get("symbols", 0) or 0)

    if has_signature == 0:
        score += 2
        signals.append("Không có chữ ký số")

    if has_resources == 0:
        score += 1
        signals.append("Không có PE resources")

    if symbols == 0:
        score += 1
        signals.append("Không có symbols")

    coff = header.get("coff", {}) or {}
    timestamp = float(coff.get("timestamp", 0) or 0)
    if timestamp == 0:
        score += 1
        signals.append("Timestamp trong COFF header bằng 0 hoặc thiếu")

    if has_debug == 0:
        score += 0.5
        signals.append("Không có debug info")

    score = clamp(score, 0, 5)
    return round(score, 2), signals


def compute_static_risk_score(prob: float, raw_obj: dict):
    """
    Risk Score = AI Score (max 60) + Static Score (max 40)
    """
    ai_score = clamp(prob * 60.0, 0, 60)

    api_score, api_signals = score_api_risk(raw_obj)
    string_score, string_signals = score_strings_risk(raw_obj)
    entropy_score, entropy_signals = score_entropy_risk(raw_obj)
    anomaly_score, anomaly_signals = score_anomaly_risk(raw_obj)
    meta_score, meta_signals = score_metadata_risk(raw_obj)

    static_score = clamp(
        api_score + string_score + entropy_score + anomaly_score + meta_score,
        0,
        40
    )

    total_score = clamp(ai_score + static_score, 0, 100)

    if total_score >= 80:
        risk_level = "CRITICAL"
    elif total_score >= 60:
        risk_level = "HIGH"
    elif total_score >= 40:
        risk_level = "MEDIUM"
    elif total_score >= 20:
        risk_level = "LOW"
    else:
        risk_level = "CLEAN"

    detected_signals = (
        api_signals
        + string_signals
        + entropy_signals
        + anomaly_signals
        + meta_signals
    )

    if not detected_signals:
        detected_signals = ["Không phát hiện tín hiệu tĩnh nổi bật"]

    return {
        "risk_score": round(total_score, 2),
        "risk_level": risk_level,
        "risk_breakdown": {
            "ai_score": round(ai_score, 2),
            "static_score": round(static_score, 2),
            "api_score": round(api_score, 2),
            "string_score": round(string_score, 2),
            "entropy_score": round(entropy_score, 2),
            "anomaly_score": round(anomaly_score, 2),
            "meta_score": round(meta_score, 2),
        },
        "detected_signals": detected_signals,
    }


def predict_file(file_path: str, original_filename: str, threshold: float = BEST_THRESHOLD):
    if not is_pe_file_by_header(file_path):
        raise ValueError(
            "File upload không có PE header hợp lệ (không bắt đầu bằng 'MZ'). "
            "Model hiện tại chỉ dùng cho họ file PE của Windows."
        )

    feats = extract_features(file_path)
    raw_obj = extract_raw_features(file_path)

    dtest = xgb.DMatrix(feats)
    prob = float(model.predict(dtest)[0])
    pred = 1 if prob >= threshold else 0

    label = "MALWARE" if pred == 1 else "BENIGN"
    risk_info = compute_static_risk_score(prob, raw_obj)

    return {
        "probability": round(prob, 6),
        "prediction": label,
        "threshold": threshold,
        "feature_shape": str(tuple(feats.shape)),
        "detected_type": detect_pe_type(original_filename, file_path),
        "risk_score": risk_info["risk_score"],
        "risk_level": risk_info["risk_level"],
        "risk_breakdown": risk_info["risk_breakdown"],
        "detected_signals": risk_info["detected_signals"],
    }


# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        save_path = None

        try:
            if "file" not in request.files:
                raise ValueError("Không tìm thấy file trong request.")

            file = request.files["file"]

            if file.filename == "":
                raise ValueError("Bạn chưa chọn file.")

            safe_name = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{safe_name}")
            file.save(save_path)

            if not allowed_file(file.filename, save_path):
                raise ValueError(
                    "Chỉ cho phép upload họ file PE của Windows: "
                    ".exe, .dll, .sys, .ocx, .scr, .cpl, .com, .efi, .drv, .mui "
                    "hoặc file PE không có đuôi nếu có header MZ."
                )

            pred_result = predict_file(save_path, safe_name, threshold=BEST_THRESHOLD)
            pred_result["filename"] = safe_name
            result = pred_result

        except Exception as e:
            error = f"{str(e)}\n\n{traceback.format_exc()}"

    return render_template_string(
        HTML_PAGE,
        result=result,
        error=error,
        model_path=MODEL_PATH,
        expected_dim=EXPECTED_DIM,
        best_threshold=BEST_THRESHOLD,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)