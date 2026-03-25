import json
import hashlib
from typing import Any, Dict, List

import numpy as np

# =========================
# NUMPY COMPAT PATCH
# ember cũ dùng np.int / np.bool / np.float...
# =========================
if not hasattr(np, "int"):
    np.int = int

if not hasattr(np, "bool"):
    np.bool = bool

if not hasattr(np, "float"):
    np.float = float

if not hasattr(np, "object"):
    np.object = object

try:
    import lief
except ImportError:
    lief = None

try:
    import ember
except ImportError as e:
    ember = None
    EMBER_IMPORT_ERROR = e
else:
    EMBER_IMPORT_ERROR = None


def _patch_lief_compat():
    """
    Patch các exception tên cũ mà ember cũ đang kỳ vọng,
    để chạy được với bản lief mới hơn.
    """
    if lief is None:
        return

    fallback_exc = Exception
    legacy_names = [
        "bad_format",
        "bad_file",
        "pe_error",
        "parser_error",
        "read_out_of_bound",
    ]

    for name in legacy_names:
        if not hasattr(lief, name):
            setattr(lief, name, fallback_exc)


_patch_lief_compat()

# =========================
# CONFIG DIMENSIONS
# Tổng phải đúng 1390
# =========================
DIM_HISTOGRAM = 256
DIM_BYTEENTROPY = 256
DIM_STRINGS = 104
DIM_GENERAL = 10
DIM_HEADER = 62
DIM_SECTION = 200
DIM_IMPORTS = 374
DIM_EXPORTS = 128

TOTAL_DIM = (
    DIM_HISTOGRAM
    + DIM_BYTEENTROPY
    + DIM_STRINGS
    + DIM_GENERAL
    + DIM_HEADER
    + DIM_SECTION
    + DIM_IMPORTS
    + DIM_EXPORTS
)

assert TOTAL_DIM == 1390, f"Total dim hiện tại là {TOTAL_DIM}, không phải 1390"


# =========================
# HELPERS
# =========================
def _to_float(x, default=0.0):
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _stable_hash(text: str, dim: int) -> int:
    h = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
    return int(h, 16) % dim


def _hash_bag(items: List[str], dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for item in items:
        idx = _stable_hash(str(item), dim)
        v[idx] += 1.0
    return v


# =========================
# RAW FEATURE EXTRACTION
# =========================
def extract_raw_features_from_exe(exe_path: str) -> Dict[str, Any]:
    if ember is None:
        raise ImportError(f"Không import được thư viện ember: {EMBER_IMPORT_ERROR}")

    with open(exe_path, "rb") as f:
        bytez = f.read()

    try:
        extractor = ember.PEFeatureExtractor(feature_version=1)
        raw_obj = extractor.raw_features(bytez)
    except Exception as e:
        raise RuntimeError(
            f"Không trích được raw features từ file PE. "
            f"Có thể file không hợp lệ hoặc ember/lief/numpy chưa tương thích hoàn toàn. "
            f"Chi tiết: {e}"
        ) from e

    if isinstance(raw_obj, str):
        raw_obj = json.loads(raw_obj)

    return raw_obj


# =========================
# ENCODERS
# =========================
def encode_histogram(raw_obj: Dict[str, Any]) -> np.ndarray:
    arr = raw_obj.get("histogram", [])
    v = np.zeros(DIM_HISTOGRAM, dtype=np.float32)
    for i in range(min(len(arr), DIM_HISTOGRAM)):
        v[i] = _to_float(arr[i])
    return v


def encode_byteentropy(raw_obj: Dict[str, Any]) -> np.ndarray:
    arr = raw_obj.get("byteentropy", [])
    v = np.zeros(DIM_BYTEENTROPY, dtype=np.float32)
    for i in range(min(len(arr), DIM_BYTEENTROPY)):
        v[i] = _to_float(arr[i])
    return v


def encode_strings(raw_obj: Dict[str, Any]) -> np.ndarray:
    s = raw_obj.get("strings", {}) or {}
    v = np.zeros(DIM_STRINGS, dtype=np.float32)

    v[0] = _to_float(s.get("numstrings", 0))
    v[1] = _to_float(s.get("avlength", 0))

    pd = s.get("printabledist", [])
    for i in range(min(len(pd), 96)):
        v[2 + i] = _to_float(pd[i])

    base = 98
    v[base + 0] = _to_float(s.get("printables", 0))
    v[base + 1] = _to_float(s.get("entropy", 0))
    v[base + 2] = _to_float(s.get("paths", 0))
    v[base + 3] = _to_float(s.get("urls", 0))
    v[base + 4] = _to_float(s.get("registry", 0))
    v[base + 5] = _to_float(s.get("MZ", 0))

    return v


def encode_general(raw_obj: Dict[str, Any]) -> np.ndarray:
    g = raw_obj.get("general", {}) or {}
    keys = [
        "size",
        "vsize",
        "has_debug",
        "exports",
        "imports",
        "has_relocations",
        "has_resources",
        "has_signature",
        "has_tls",
        "symbols",
    ]
    v = np.zeros(DIM_GENERAL, dtype=np.float32)
    for i, k in enumerate(keys):
        v[i] = _to_float(g.get(k, 0))
    return v


def encode_header(raw_obj: Dict[str, Any]) -> np.ndarray:
    h = raw_obj.get("header", {}) or {}
    coff = h.get("coff", {}) or {}
    optional = h.get("optional", {}) or {}

    values = []

    coff_num_keys = [
        "timestamp",
        "machine",
        "characteristics",
        "major_image_version",
        "minor_image_version",
        "major_linker_version",
        "minor_linker_version",
        "major_operating_system_version",
        "minor_operating_system_version",
        "major_subsystem_version",
        "minor_subsystem_version",
        "size_of_code",
        "size_of_headers",
        "size_of_heap_commit",
    ]

    for k in coff_num_keys:
        values.append(_to_float(coff.get(k, 0)))

    opt_num_keys = [
        "sizeof_code",
        "sizeof_initialized_data",
        "sizeof_uninitialized_data",
        "address_of_entry_point",
        "base_of_code",
        "imagebase",
        "section_alignment",
        "file_alignment",
        "major_os_version",
        "minor_os_version",
        "major_image_version",
        "minor_image_version",
        "major_subsystem_version",
        "minor_subsystem_version",
        "sizeof_image",
        "sizeof_headers",
        "checksum",
        "sizeof_stack_reserve",
        "sizeof_stack_commit",
        "sizeof_heap_reserve",
        "sizeof_heap_commit",
        "loader_flags",
    ]

    for k in opt_num_keys:
        values.append(_to_float(optional.get(k, 0)))

    values = values[:40]
    while len(values) < 40:
        values.append(0.0)

    cats = [
        f"coff_machine={coff.get('machine', '')}",
        f"coff_characteristics={coff.get('characteristics', '')}",
        f"opt_subsystem={optional.get('subsystem', '')}",
        f"opt_magic={optional.get('magic', '')}",
        f"opt_dll_characteristics={optional.get('dll_characteristics', '')}",
        f"opt_imagebase={optional.get('imagebase', '')}",
    ]

    hv = _hash_bag(cats, 22)
    v = np.array(values, dtype=np.float32)
    return np.concatenate([v, hv], axis=0)


def encode_section(raw_obj: Dict[str, Any]) -> np.ndarray:
    sec = raw_obj.get("section", {}) or {}
    entry = sec.get("entry", "") or ""
    sections = sec.get("sections", []) or []

    num = []
    num.append(float(len(str(entry))))
    num.append(float(len(sections)))

    sizes = []
    entropies = []
    vsizes = []

    section_tokens = [f"entry={entry}"]

    for s in sections:
        if not isinstance(s, dict):
            continue

        name = str(s.get("name", ""))
        props = s.get("props", []) or []
        size = _to_float(s.get("size", 0))
        vsize = _to_float(s.get("vsize", 0))
        entropy = _to_float(s.get("entropy", 0))

        sizes.append(size)
        vsizes.append(vsize)
        entropies.append(entropy)

        section_tokens.append(f"name={name}")
        for p in props:
            section_tokens.append(f"prop={p}")
        section_tokens.append(f"name_prop={name}|{','.join(map(str, props))}")

    def add_stats(arr):
        if len(arr) == 0:
            return [0.0] * 6
        arr = np.asarray(arr, dtype=np.float32)
        return [
            float(arr.sum()),
            float(arr.mean()),
            float(arr.std()),
            float(arr.min()),
            float(arr.max()),
            float(np.median(arr)),
        ]

    num.extend(add_stats(sizes))
    num.extend(add_stats(vsizes))
    num.extend(add_stats(entropies))

    while len(num) < 20:
        num.append(0.0)

    hv = _hash_bag(section_tokens, DIM_SECTION - 20)
    return np.concatenate([np.array(num[:20], dtype=np.float32), hv], axis=0)


def encode_imports(raw_obj: Dict[str, Any]) -> np.ndarray:
    imports_obj = raw_obj.get("imports", {}) or {}
    tokens = []

    for dll, funcs in imports_obj.items():
        dll = str(dll).lower()
        tokens.append(f"dll={dll}")

        if isinstance(funcs, list):
            for fn in funcs:
                fn = str(fn)
                tokens.append(f"imp={dll}:{fn}")
                tokens.append(f"func={fn}")

    return _hash_bag(tokens, DIM_IMPORTS)


def encode_exports(raw_obj: Dict[str, Any]) -> np.ndarray:
    exports_obj = raw_obj.get("exports", []) or []
    tokens = [f"exp={str(x)}" for x in exports_obj]
    return _hash_bag(tokens, DIM_EXPORTS)


def raw_to_feature1390(raw_obj: Dict[str, Any]) -> np.ndarray:
    parts = [
        encode_histogram(raw_obj),
        encode_byteentropy(raw_obj),
        encode_strings(raw_obj),
        encode_general(raw_obj),
        encode_header(raw_obj),
        encode_section(raw_obj),
        encode_imports(raw_obj),
        encode_exports(raw_obj),
    ]
    x = np.concatenate(parts, axis=0).astype(np.float32)
    if x.shape[0] != 1390:
        raise ValueError(f"Vector tạo ra có dim={x.shape[0]}, không phải 1390")
    return x


def extract_feature1390_from_exe(exe_path: str) -> np.ndarray:
    raw_obj = extract_raw_features_from_exe(exe_path)
    x = raw_to_feature1390(raw_obj)
    return x.reshape(1, -1)


if __name__ == "__main__":
    x = extract_feature1390_from_exe("sample.exe")
    print("Feature shape:", x.shape)
    print("First 30 values:", x[0][:30])