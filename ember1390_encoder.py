import json
import math
import hashlib
from typing import Any, Dict, List

import numpy as np

try:
    import ember
except ImportError:
    ember = None


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


def _to_int(x, default=0):
    try:
        if x is None:
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def _stable_hash(text: str, dim: int) -> int:
    h = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
    return int(h, 16) % dim


def _hash_bag(items: List[str], dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for item in items:
        idx = _stable_hash(str(item), dim)
        v[idx] += 1.0
    return v


def _hash_weighted_bag(counter_items: List[tuple], dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for key, weight in counter_items:
        idx = _stable_hash(str(key), dim)
        v[idx] += float(weight)
    return v


# =========================
# RAW FEATURE EXTRACTION
# =========================
def extract_raw_features_from_exe(exe_path: str) -> Dict[str, Any]:
    """
    Trích raw feature EMBER-style từ file .exe
    """
    if ember is None:
        raise ImportError("Chưa cài thư viện ember. Hãy pip install ember")

    with open(exe_path, "rb") as f:
        bytez = f.read()

    extractor = ember.PEFeatureExtractor(feature_version=1)
    raw_obj = extractor.raw_features(bytez)

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
    """
    strings gồm:
    - numstrings
    - avlength
    - printabledist (thường 96 chiều)
    - printables
    - entropy
    - paths
    - urls
    - registry
    - MZ
    Tổng: 2 + 96 + 6 = 104
    """
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
    """
    Encode header thành 62 chiều:
    - coff numeric fields
    - optional numeric fields
    - hashed categorical fields (machine/subsystem/magic/... nếu có)
    """
    h = raw_obj.get("header", {}) or {}
    coff = h.get("coff", {}) or {}
    optional = h.get("optional", {}) or {}

    values = []

    # COFF numeric-ish
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

    # OPTIONAL numeric-ish
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

    # hashed categorical
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
    return np.concatenate([v, hv], axis=0)  # 40 + 22 = 62


def encode_section(raw_obj: Dict[str, Any]) -> np.ndarray:
    """
    Encode section thành 200 chiều:
    - entry info numeric
    - aggregate từ sections
    - hashed section names/properties
    """
    sec = raw_obj.get("section", {}) or {}
    entry = sec.get("entry", "") or ""
    sections = sec.get("sections", []) or []

    num = []

    # entry hash as one numeric bucket count later
    num.append(float(len(str(entry))))

    # aggregate thống kê section
    num.append(float(len(sections)))  # số section

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

    # 1 + 1 + 6 + 6 + 6 = 20
    while len(num) < 20:
        num.append(0.0)

    hv = _hash_bag(section_tokens, DIM_SECTION - 20)
    return np.concatenate([np.array(num[:20], dtype=np.float32), hv], axis=0)


def encode_imports(raw_obj: Dict[str, Any]) -> np.ndarray:
    """
    imports là dict:
    {
      "KERNEL32.dll": ["CreateFileW", ...],
      ...
    }
    -> hash bag 374 chiều
    """
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
    exe_path = r"B:\Code\V3.0\Zalo.exe"
    x = extract_feature1390_from_exe(exe_path)
    print("Feature shape:", x.shape)
    print("First 30 values:", x[0][:30])