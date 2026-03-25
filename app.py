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
MODEL_PATH = r"B:\Code\V3.0\data\ember\ember2017_xgb_full_gpu_manual.json"
UPLOAD_DIR = r"B:\Code\V3.0\uploads"
ALLOWED_EXTENSIONS = {"exe"}
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
            Expected feature dim: <code>{{ expected_dim }}</code>
        </p>

        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file" accept=".exe" required>
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
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def try_extract_with_custom_pipeline(file_path: str) -> np.ndarray:
    """
    Ưu tiên dùng pipeline preprocess do bạn tự định nghĩa,
    để khớp 100% với lúc train model 1390 features.
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
    Fallback: thử dùng thư viện ember để trích feature trực tiếp từ PE file.
    Lưu ý: chỉ đúng nếu cách train trước đó cũng dùng đúng pipeline tương thích.
    """
    try:
        import ember
    except ImportError as e:
        raise ImportError(
            "Chưa cài thư viện ember. Hãy cài bằng: pip install ember"
        ) from e

    with open(file_path, "rb") as f:
        bytez = f.read()

    # EMBER 2017 thường dùng feature_version=1
    extractor = ember.PEFeatureExtractor(feature_version=1)
    feats = extractor.feature_vector(bytez)

    feats = np.asarray(feats, dtype=np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)

    return feats


def extract_features(file_path: str) -> np.ndarray:
    """
    Ưu tiên custom pipeline.
    Nếu không có, fallback sang ember.
    """
    feats = try_extract_with_custom_pipeline(file_path)

    if feats is None:
        feats = try_extract_with_ember(file_path)

    if feats.shape[1] != EXPECTED_DIM:
        raise ValueError(
            f"Feature dimension mismatch: model cần {EXPECTED_DIM} features "
            f"nhưng extractor tạo ra {feats.shape[1]} features.\\n"
            f"Bạn cần sửa file inference_preprocess.py để preprocess giống hệt lúc train."
        )

    return feats


def predict_file(file_path: str, threshold: float = 0.5):
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
    }


# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        try:
            if "file" not in request.files:
                raise ValueError("Không tìm thấy file trong request.")

            file = request.files["file"]

            if file.filename == "":
                raise ValueError("Bạn chưa chọn file.")

            if not allowed_file(file.filename):
                raise ValueError("Chỉ cho phép upload file .exe")

            safe_name = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{safe_name}")
            file.save(save_path)

            pred_result = predict_file(save_path, threshold=0.5)
            pred_result["filename"] = safe_name
            result = pred_result

        except Exception as e:
            error = f"{str(e)}\\n\\n{traceback.format_exc()}"

    return render_template_string(
        HTML_PAGE,
        result=result,
        error=error,
        model_path=MODEL_PATH,
        expected_dim=EXPECTED_DIM,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)