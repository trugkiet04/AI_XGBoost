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
            max-width: 760px;
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
        code {
            background: #f1f1f1;
            padding: 2px 6px;
            border-radius: 6px;
        }
    </style>
</head>
<body>
    <div class="box">
        <h1>EMBER2017 XGBoost Localhost</h1>
        <p class="meta">
            Model: <code>{{ model_path }}</code><br>
            Expected feature dim: <code>{{ expected_dim }}</code><br>
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
                <p><b>Feature shape:</b> {{ result.feature_shape }}</p>
                <p><b>Detected type:</b> {{ result.detected_type }}</p>
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
    """
    Kiểm tra nhanh file có bắt đầu bằng 'MZ' hay không.
    Đây là dấu hiệu phổ biến của file PE.
    """
    try:
        with open(file_path, "rb") as f:
            sig = f.read(2)
        return sig == b"MZ"
    except Exception:
        return False


def allowed_file(filename: str, temp_saved_path: str = None) -> bool:
    """
    Cho phép:
    - file có đuôi thuộc họ PE
    - hoặc file không có đuôi nhưng header là MZ (nếu bật ALLOW_EXTENSIONLESS_PE)
    """
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
    """
    Ưu tiên pipeline tự định nghĩa để khớp với lúc train.
    """
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
    """
    Fallback: dùng trực tiếp ember nếu pipeline custom không có.
    """
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


def predict_file(file_path: str, original_filename: str, threshold: float = 0.5):
    if not is_pe_file_by_header(file_path):
        raise ValueError(
            "File upload không có PE header hợp lệ (không bắt đầu bằng 'MZ'). "
            "Model hiện tại chỉ dùng cho họ file PE của Windows."
        )

    feats = extract_features(file_path)
    dtest = xgb.DMatrix(feats)
    prob = float(model.predict(dtest)[0])
    pred = 1 if prob >= threshold else 0

    label = "MALWARE" if pred == 1 else "BENIGN"

    return {
        "probability": round(prob, 6),
        "prediction": label,
        "threshold": threshold,
        "feature_shape": str(tuple(feats.shape)),
        "detected_type": detect_pe_type(original_filename, file_path),
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

            pred_result = predict_file(save_path, safe_name, threshold=0.5)
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
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)