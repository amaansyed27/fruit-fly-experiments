from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy import sparse

from .neurons import NeuronIndex

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
OPTIC_URL = "https://raw.githubusercontent.com/flyconnectome/2025malecns/main/supplemental_data/optic-column-type-assignments-v1.0.xlsx"
FILES = {
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "optic": "optic-column-type-assignments-v1.0.xlsx",
}
EXPECTED_NEURONS = 166_700
EXPECTED_EDGES = 25_582_938
EXPECTED_CONTACTS = 124_177_617

INHIBITORY_NT = {"gaba", "glutamate", "histamine"}


@dataclass
class ConnectomeGraph:
    neurons: NeuronIndex
    weights: sparse.csr_matrix
    edge_count: int
    synaptic_contacts: int
    transmitter_sign: np.ndarray
    metadata: dict

    def save(self, processed_dir: Path) -> None:
        processed_dir.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(processed_dir / "connectome_signed_normalized.npz", self.weights, compressed=False)
        self.neurons.annotations.to_pickle(processed_dir / "annotations.pkl")
        np.save(processed_dir / "body_ids.npy", self.neurons.body_ids)
        np.save(processed_dir / "transmitter_sign.npy", self.transmitter_sign)
        (processed_dir / "manifest.json").write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, processed_dir: Path) -> "ConnectomeGraph":
        weights = sparse.load_npz(processed_dir / "connectome_signed_normalized.npz").tocsr()
        annotations = pd.read_pickle(processed_dir / "annotations.pkl")
        body_ids = np.load(processed_dir / "body_ids.npy")
        signs = np.load(processed_dir / "transmitter_sign.npy")
        metadata = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
        return cls(
            neurons=NeuronIndex(body_ids=body_ids, annotations=annotations),
            weights=weights,
            edge_count=int(metadata["retained_edges"]),
            synaptic_contacts=int(metadata["retained_synaptic_contacts"]),
            transmitter_sign=signs,
            metadata=metadata,
        )


def default_data_root() -> Path:
    return Path(os.environ.get("FRUIT_FLY_DATA", "data"))


def _sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    partial = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "fruit-fly-experiments/0.1"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as out:
        while chunk := response.read(8 * 1024 * 1024):
            out.write(chunk)
    partial.replace(path)


def fetch_official_data(data_root: Path | None = None) -> dict[str, Path]:
    root = data_root or default_data_root()
    raw = root / "raw"
    paths: dict[str, Path] = {}
    for key, filename in FILES.items():
        path = raw / filename
        url = OPTIC_URL if key == "optic" else f"{BASE_URL}/{filename}"
        download_file(url, path)
        paths[key] = path
    return paths


def _clean_superclass_mask(annotations: pd.DataFrame) -> np.ndarray:
    if "superclass" not in annotations:
        raise ValueError("MaleCNS annotations are missing 'superclass'")
    superclass = annotations["superclass"].fillna("").astype(str).str.strip()
    mask = superclass.ne("").to_numpy()
    if "status" in annotations:
        mask &= annotations["status"].fillna("").astype(str).str.casefold().ne("glia").to_numpy()
    return mask


def select_neurons(annotations: pd.DataFrame) -> pd.DataFrame:
    if "bodyId" not in annotations:
        raise ValueError("MaleCNS annotations are missing 'bodyId'")
    selected = annotations.loc[_clean_superclass_mask(annotations)].copy()
    selected = selected.drop_duplicates("bodyId").sort_values("bodyId").reset_index(drop=True)
    selected["bodyId"] = selected["bodyId"].astype(np.int64)
    for col in ("type", "superclass", "somaSide", "rootSide"):
        if col not in selected:
            selected[col] = None
    return selected


def _transmitter_by_body(nt: pd.DataFrame) -> pd.Series:
    body_col = "body" if "body" in nt else "bodyId" if "bodyId" in nt else None
    if body_col is None:
        raise ValueError("neurotransmitter table has no body/bodyId column")
    label_cols = [c for c in ("consensus_nt", "predicted_nt", "celltype_predicted_nt") if c in nt]
    if not label_cols:
        raise ValueError("neurotransmitter table has no recognized neurotransmitter column")
    confidence_col = next((c for c in ("predicted_nt_confidence", "celltype_predicted_nt_confidence") if c in nt), None)
    cols = [body_col, *label_cols] + ([confidence_col] if confidence_col else [])
    x = nt[cols].dropna(subset=[body_col]).copy()
    x["_nt_label"] = x[label_cols[0]]
    for col in label_cols[1:]:
        x["_nt_label"] = x["_nt_label"].fillna(x[col])
    x = x.dropna(subset=["_nt_label"])
    if confidence_col:
        x[confidence_col] = pd.to_numeric(x[confidence_col], errors="coerce").fillna(-1)
        x = x.sort_values(confidence_col).drop_duplicates(body_col, keep="last")
    else:
        x = x.drop_duplicates(body_col, keep="first")
    return x.set_index(body_col)["_nt_label"].astype(str)


def transmitter_signs(body_ids: np.ndarray, nt: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    by_body = _transmitter_by_body(nt)
    labels = by_body.reindex(body_ids).fillna("unknown").astype(str).str.casefold()
    signs = np.ones(len(body_ids), dtype=np.float32)
    signs[labels.isin(INHIBITORY_NT).to_numpy()] = -1.0
    counts = labels.value_counts().to_dict()
    return signs, {str(k): int(v) for k, v in counts.items()}


def _iter_feather_batches(path: Path, columns: list[str]) -> Iterator[dict[str, np.ndarray]]:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to build the MaleCNS graph") from exc
    source = pa.memory_map(str(path), "r")
    reader = ipc.open_file(source)
    schema_names = set(reader.schema.names)
    missing = set(columns) - schema_names
    if missing:
        raise ValueError(f"weights file missing columns: {sorted(missing)}")
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        yield {name: batch.column(reader.schema.get_field_index(name)).to_numpy(zero_copy_only=False) for name in columns}


def build_graph_from_frames(
    annotations: pd.DataFrame,
    nt: pd.DataFrame,
    edge_batches: Iterator[dict[str, np.ndarray]],
    *,
    strict_expected_counts: bool = False,
) -> ConnectomeGraph:
    selected = select_neurons(annotations)
    body_ids = selected["bodyId"].to_numpy(np.int64)
    signs, nt_counts = transmitter_signs(body_ids, nt)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    contacts = 0
    edge_count = 0

    for batch in edge_batches:
        pre = np.asarray(batch["body_pre"], dtype=np.int64)
        post = np.asarray(batch["body_post"], dtype=np.int64)
        count = np.asarray(batch["weight"], dtype=np.int64)
        pre_idx = np.searchsorted(body_ids, pre)
        post_idx = np.searchsorted(body_ids, post)
        pre_ok = pre_idx < len(body_ids)
        post_ok = post_idx < len(body_ids)
        pre_match = np.zeros_like(pre_ok)
        post_match = np.zeros_like(post_ok)
        pre_match[pre_ok] = body_ids[pre_idx[pre_ok]] == pre[pre_ok]
        post_match[post_ok] = body_ids[post_idx[post_ok]] == post[post_ok]
        keep = pre_match & post_match
        if not np.any(keep):
            continue
        p = pre_idx[keep].astype(np.int32)
        q = post_idx[keep].astype(np.int32)
        c = count[keep].astype(np.float32)
        signed = c * signs[p]
        rows.append(q)
        cols.append(p)
        vals.append(signed)
        edge_count += int(c.size)
        contacts += int(count[keep].sum(dtype=np.int64))

    if not rows:
        raise ValueError("no retained MaleCNS edges were found")
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    val = np.concatenate(vals)
    n = len(body_ids)
    matrix = sparse.coo_matrix((val, (row, col)), shape=(n, n), dtype=np.float32).tocsr()
    incoming_abs = np.asarray(np.abs(matrix).sum(axis=1)).ravel().astype(np.float32)
    inv = np.zeros_like(incoming_abs)
    nz = incoming_abs > 0
    inv[nz] = 1.0 / incoming_abs[nz]
    matrix = sparse.diags(inv, dtype=np.float32).dot(matrix).tocsr()

    if strict_expected_counts:
        if n != EXPECTED_NEURONS:
            raise ValueError(f"expected {EXPECTED_NEURONS:,} retained neurons, got {n:,}")
        if edge_count != EXPECTED_EDGES:
            raise ValueError(f"expected {EXPECTED_EDGES:,} retained edges, got {edge_count:,}")
        if contacts != EXPECTED_CONTACTS:
            raise ValueError(f"expected {EXPECTED_CONTACTS:,} retained contacts, got {contacts:,}")

    metadata = {
        "dataset": "MaleCNS v1.0",
        "retention_rule": "non-empty superclass; exclude status=Glia; keep all released edges with both endpoints retained",
        "retained_neurons": n,
        "retained_edges": edge_count,
        "retained_synaptic_contacts": contacts,
        "matrix_orientation": "row=postsynaptic, column=presynaptic",
        "weight_transform": "presynaptic transmitter sign; normalize absolute incoming weight per postsynaptic neuron",
        "inhibitory_proxy": sorted(INHIBITORY_NT),
        "neurotransmitter_counts": nt_counts,
    }
    return ConnectomeGraph(NeuronIndex(body_ids, selected), matrix, edge_count, contacts, signs, metadata)


def build_official_connectome(data_root: Path | None = None, strict: bool = True) -> ConnectomeGraph:
    root = data_root or default_data_root()
    raw = root / "raw"
    annotations_path = raw / FILES["annotations"]
    nt_path = raw / FILES["neurotransmitters"]
    weights_path = raw / FILES["weights"]
    for path in (annotations_path, nt_path, weights_path):
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run scripts/fetch_data.py first")
    annotations = pd.read_feather(annotations_path)
    nt = pd.read_feather(nt_path)
    graph = build_graph_from_frames(
        annotations,
        nt,
        _iter_feather_batches(weights_path, ["body_pre", "body_post", "weight"]),
        strict_expected_counts=strict,
    )
    graph.metadata["source_files"] = {
        FILES["annotations"]: _sha256(annotations_path),
        FILES["neurotransmitters"]: _sha256(nt_path),
        FILES["weights"]: _sha256(weights_path),
    }
    graph.save(root / "processed")
    return graph
