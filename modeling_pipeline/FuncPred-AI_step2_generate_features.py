#!/usr/bin/env python3
"""
FuncPred-AI_step2_generate_features.py

Generate the raw ESM artifacts needed by selected battery-stage model bundles and
write a shared pooled-feature cache for downstream prediction.

Usage:
    # sequence-only ESMC path:
    python ML/prediction/FuncPred-AI_step2_generate_features.py \
        --fasta BLAST_DB_all_BAHD/unchar_BAHD.fa \
        --battery_modes esmc_seq_embed \
        --feature_cache ML/prediction/unchar_BAHD_features.pt \
        --person NS

    # Optional mixed-coverage path:
    #   ESMC is generated for all FASTA sequences, while structure-dependent
    #   ESM3 features are generated only for sequences present in the structure dir.
    
    python ML/prediction/FuncPred-AI_step2_generate_features.py \
        --fasta BLAST_DB_all_BAHD/unchar_BAHD.fa \
        --funcpred_structures_dir structure/unchar_structures \
        --pickled_models_dir ML/results/v12/20260307_185909/pickled_models \
        --feature_cache ML/prediction/unchar_BAHD_features.pt \
        --timing_tsv ML/prediction/unchar_BAHD_feature_timing.tsv \
        --structure_parse_workers 8 \
        --inference_workers 2 \
        --device cpu \
        --person NS
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ML.prediction.common import (
    FUNC_LOGIT_VARIANT_SPECS,
    append_prediction_log,
    discover_model_bundle_paths,
    iter_fasta_directories,
    load_model_bundle,
    pool_feature_tensor,
    resolve_mode_alias,
    select_structure_file,
)

CACHE_VERSION = 1
_WORKER_ESM3_MODEL = None
_WORKER_ESMC_MODEL = None
_WORKER_DEVICE = "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate raw prediction features for selected battery modes and "
            "write a shared pooled-feature cache."
        )
    )
    parser.add_argument(
        "--funcpred_structures_dir",
        "--structures_dir",
        dest="structures_dir",
        default=None,
        help=(
            "Optional FuncPred-specific structure directory produced by step 1. "
            "Required only for structure-aware modes; `--structures_dir` is retained "
            "as a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--fasta",
        default=None,
        help="FASTA used for sequence-based generation. Required for the default ESMC-first path.",
    )
    parser.add_argument(
        "--pickled_models",
        nargs="*",
        default=[],
        help="Pickled battery-model bundle file(s) and/or directories.",
    )
    parser.add_argument(
        "--pickled_models_dir",
        default=None,
        help="Convenience directory containing pickled battery-model bundles.",
    )
    parser.add_argument(
        "--battery_modes",
        default="esmc_seq_embed",
        help=(
            "Comma-separated battery modes to prepare when model bundles are not provided. "
            "Choices: concat,func_logit,esm3_struc_embed,esmc_seq_embed."
        ),
    )
    parser.add_argument(
        "--function_logit_variants",
        default="esm3_func_logit_struc_step10",
        help=(
            "Comma-separated function-logit variants to prepare when required by the "
            "selected models or modes."
        ),
    )
    parser.add_argument(
        "--feature_cache",
        default=None,
        help="Output pooled-feature cache path. Default: ML/prediction/unchar_BAHD_features.pt",
    )
    parser.add_argument(
        "--prediction_log",
        default=None,
        help="Optional JSON log. Default: alongside the feature cache.",
    )
    parser.add_argument(
        "--timing_tsv",
        default=None,
        help="Optional per-feature timing TSV. Default: alongside the feature cache.",
    )
    parser.add_argument(
        "--raw_artifacts_dir",
        default=None,
        help=(
            "Optional directory for per-sequence raw tensors. Defaults to "
            "<funcpred_structures_dir> when provided, otherwise ML/prediction/unchar_raw_features."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for inference.",
    )
    parser.add_argument(
        "--store_raw_artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store per-sequence raw logits and embedding files alongside the pooled cache.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse raw artifacts already present on disk when possible.",
    )
    parser.add_argument(
        "--structure_parse_workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="CPU-side workers for parallel structure-file parsing before inference.",
    )
    parser.add_argument(
        "--inference_workers",
        type=int,
        default=1,
        help="CPU-only process workers for sequence/structure inference. Forced to 1 on non-CPU devices.",
    )
    parser.add_argument(
        "--worker_smoke_test",
        action="store_true",
        help="Run only a small prefix of sequences and emit worker-memory diagnostics.",
    )
    parser.add_argument(
        "--worker_smoke_test_count",
        type=int,
        default=8,
        help="Number of sequences to process in --worker_smoke_test mode.",
    )
    parser.add_argument("-p", "--person", required=True, help="Initials or name for the run log.")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    device = str(requested).strip().lower()
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available; falling back to cpu.")
        return "cpu"
    if device == "mps":
        mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
        if not mps_ok:
            print("[WARNING] MPS requested but not available; falling back to cpu.")
            return "cpu"
    return device


def load_fasta_sequences(fasta_path: Optional[str]) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    if not fasta_path:
        return sequences
    current_id: Optional[str] = None
    chunks: List[str] = []
    with open(fasta_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(chunks)
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if current_id is not None:
            sequences[current_id] = "".join(chunks)
    return sequences


def _hf_login_if_needed() -> None:
    import huggingface_hub
    from dotenv import load_dotenv

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if token is None:
        raise RuntimeError("HF_TOKEN not found; make sure .env exists")
    huggingface_hub.login(token=token, add_to_git_credential=False)


def initialize_esm3(device: str) -> ESM3InferenceClient:
    from esm.models.esm3 import ESM3

    try:
        model = ESM3.from_pretrained("esm3-open").to(device)
        return model
    except FileNotFoundError:
        _hf_login_if_needed()
        model = ESM3.from_pretrained("esm3-open").to(device)
        return model


def initialize_esmc(device: str) -> ESMCInferenceClient:
    from esm.models.esmc import ESMC

    try:
        model = ESMC.from_pretrained("esmc_600m").to(device)
        return model
    except FileNotFoundError:
        _hf_login_if_needed()
        model = ESMC.from_pretrained("esmc_600m").to(device)
        return model


def _summarize_tensor(name: str, fasta_id: str, tensor: torch.Tensor) -> Dict[str, object]:
    cpu_tensor = tensor.detach().cpu()
    flat = cpu_tensor.reshape(-1).numpy()
    summary = {
        "track": name,
        "shape": list(cpu_tensor.shape),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "nan_count": int(np.isnan(flat).sum()),
    }
    print(
        f"[METRICS] {fasta_id} ({name}): shape={tuple(summary['shape'])} "
        f"mean={summary['mean']:.4f} std={summary['std']:.4f} "
        f"min={summary['min']:.4f} max={summary['max']:.4f} NaNs={summary['nan_count']}"
    )
    return summary


def _parse_requested_modes(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    modes: List[str] = []
    for part in str(raw).split(","):
        cleaned = part.strip()
        if cleaned:
            modes.append(resolve_mode_alias(cleaned))
    return list(dict.fromkeys(modes))


def _parse_requested_variants(raw: str) -> List[str]:
    variants: List[str] = []
    for part in str(raw).split(","):
        cleaned = part.strip()
        if cleaned:
            variants.append(cleaned)
    return list(dict.fromkeys(variants))


def _append_timing_row(
    rows: List[Dict[str, object]],
    *,
    fasta_id: str,
    variant: str,
    feature_family: str,
    status: str,
    sequence_length: int,
    structure_length: int,
    input_type: str,
    generation_seconds: Optional[float] = None,
    pooling_seconds: Optional[float] = None,
    total_seconds: Optional[float] = None,
    species_dir: Optional[str] = None,
    reused_existing: bool = False,
    error: Optional[str] = None,
) -> None:
    rows.append(
        {
            "fasta_id": fasta_id,
            "variant": variant,
            "feature_family": feature_family,
            "status": status,
            "species_dir": species_dir or "",
            "input_type": input_type,
            "sequence_length": int(sequence_length),
            "structure_length": int(structure_length),
            "generation_seconds": float(generation_seconds) if generation_seconds is not None else np.nan,
            "pooling_seconds": float(pooling_seconds) if pooling_seconds is not None else np.nan,
            "total_seconds": float(total_seconds) if total_seconds is not None else np.nan,
            "reused_existing": int(bool(reused_existing)),
            "error": error or "",
        }
    )


def _memory_snapshot() -> Dict[str, float]:
    snapshot: Dict[str, float] = {}
    try:
        import psutil  # type: ignore

        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        snapshot["rss_gb"] = float(mem.rss) / (1024 ** 3)
        snapshot["vms_gb"] = float(mem.vms) / (1024 ** 3)
        return snapshot
    except Exception:
        pass
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = float(usage.ru_maxrss)
        if sys.platform == "darwin":
            snapshot["rss_gb"] = rss / (1024 ** 3)
        else:
            snapshot["rss_gb"] = (rss * 1024.0) / (1024 ** 3)
    except Exception:
        snapshot["rss_gb"] = float("nan")
    return snapshot


def _load_structure_context_entry(
    fasta_id: str,
    subfolder: str,
    species_dir: Optional[str],
) -> Dict[str, object]:
    from esm.sdk.api import ESMProtein
    from esm.utils.structure.protein_chain import ProteinChain

    struct_folder = os.path.join(subfolder, "structure_file")
    struct_file = select_structure_file(struct_folder)
    if not struct_file:
        return {
            "fasta_id": fasta_id,
            "error": f"No PDB files found in {struct_folder}",
        }
    try:
        chain = ProteinChain.from_pdb(struct_file)
        structure_protein = ESMProtein.from_protein_chain(chain)
        structure_protein.function_annotations = None
        structure_sequence = structure_protein.sequence or ""
    except Exception as exc:
        return {
            "fasta_id": fasta_id,
            "error": f"Failed to parse structure: {exc}",
        }
    return {
        "fasta_id": fasta_id,
        "subfolder": subfolder,
        "species_dir": species_dir,
        "structure_protein": structure_protein,
        "structure_sequence": structure_sequence,
        "struct_file": struct_file,
    }


def _worker_initialize(device: str) -> None:
    global _WORKER_DEVICE
    _WORKER_DEVICE = device


def _get_worker_esm3_model():
    global _WORKER_ESM3_MODEL
    if _WORKER_ESM3_MODEL is None:
        _WORKER_ESM3_MODEL = initialize_esm3(_WORKER_DEVICE)
        _WORKER_ESM3_MODEL.eval()
    return _WORKER_ESM3_MODEL


def _get_worker_esmc_model():
    global _WORKER_ESMC_MODEL
    if _WORKER_ESMC_MODEL is None:
        _WORKER_ESMC_MODEL = initialize_esmc(_WORKER_DEVICE)
        _WORKER_ESMC_MODEL.eval()
    return _WORKER_ESMC_MODEL


def _process_esm3_structure_bundle(
    *,
    fasta_id: str,
    subfolder: str,
    species_dir: Optional[str],
    sequence_text: str,
    structure_protein,
    structure_sequence: str,
    structure_variants: Sequence[str],
    requested_func_variants: Sequence[str],
    store_raw_artifacts: bool,
    skip_existing: bool,
    esm3_model,
    timing_rows: List[Dict[str, object]],
    failures: List[Dict[str, str]],
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, torch.Tensor], Dict[str, Dict[str, object]], Dict[str, torch.Tensor], bool]:
    function_outputs: Dict[str, Dict[str, object]] = {}
    function_vectors: Dict[str, torch.Tensor] = {}
    structure_outputs: Dict[str, Dict[str, object]] = {}
    structure_vectors: Dict[str, torch.Tensor] = {}
    sequence_succeeded = False

    if structure_protein is None:
        for variant in structure_variants:
            if str(variant) != "esm3_struc_embed":
                continue
            structure_outputs["esm3_struc_embed"] = {"variant": "esm3_struc_embed", "status": "missing_structure"}
            _append_timing_row(
                timing_rows,
                fasta_id=fasta_id,
                variant="esm3_struc_embed",
                feature_family="structure_embedding",
                status="missing_structure",
                sequence_length=len(sequence_text),
                structure_length=len(structure_sequence),
                input_type="structure",
                species_dir=species_dir,
            )
        for variant in requested_func_variants:
            input_type, _ = FUNC_LOGIT_VARIANT_SPECS[str(variant)]
            if input_type != "structure":
                continue
            function_outputs[str(variant)] = {"variant": str(variant), "status": "missing_structure"}
            _append_timing_row(
                timing_rows,
                fasta_id=fasta_id,
                variant=str(variant),
                feature_family="function_logit",
                status="missing_structure",
                sequence_length=len(sequence_text),
                structure_length=len(structure_sequence),
                input_type="structure",
                species_dir=species_dir,
            )
        return function_outputs, function_vectors, structure_outputs, structure_vectors, sequence_succeeded

    if "esm3_struc_embed" in [str(item) for item in structure_variants]:
        generation_start = time.perf_counter()
        try:
            raw_tensor, meta = infer_structure_embedding(
                fasta_id=fasta_id,
                subfolder=subfolder,
                variant="esm3_struc_embed",
                structure_protein=structure_protein,
                structure_sequence=structure_sequence,
                sequence_text=sequence_text,
                esm3_model=esm3_model,
                esmc_model=None,
                store_raw_artifacts=store_raw_artifacts,
                skip_existing=skip_existing,
            )
            pooling_start = time.perf_counter()
            pooled_tensor = pool_feature_tensor(raw_tensor)
            pooling_seconds = time.perf_counter() - pooling_start
            generation_seconds = pooling_start - generation_start
            meta["generation_seconds"] = generation_seconds
            meta["pooling_seconds"] = pooling_seconds
            meta["total_seconds"] = generation_seconds + pooling_seconds
            structure_outputs["esm3_struc_embed"] = meta
            structure_vectors["esm3_struc_embed"] = pooled_tensor
            _append_timing_row(
                timing_rows,
                fasta_id=fasta_id,
                variant="esm3_struc_embed",
                feature_family="structure_embedding",
                status=str(meta.get("status") or "generated"),
                sequence_length=len(sequence_text),
                structure_length=len(structure_sequence),
                input_type="structure",
                generation_seconds=generation_seconds,
                pooling_seconds=pooling_seconds,
                total_seconds=generation_seconds + pooling_seconds,
                species_dir=species_dir,
                reused_existing=str(meta.get("status")) == "reused_existing",
            )
            sequence_succeeded = True
        except Exception as exc:
            failures.append({"fasta_id": fasta_id, "error": f"esm3_struc_embed: {exc}"})
            structure_outputs["esm3_struc_embed"] = {
                "variant": "esm3_struc_embed",
                "status": "failed",
                "error": str(exc),
            }
            _append_timing_row(
                timing_rows,
                fasta_id=fasta_id,
                variant="esm3_struc_embed",
                feature_family="structure_embedding",
                status="failed",
                sequence_length=len(sequence_text),
                structure_length=len(structure_sequence),
                input_type="structure",
                generation_seconds=time.perf_counter() - generation_start,
                species_dir=species_dir,
                error=str(exc),
            )

    for variant in requested_func_variants:
        variant_name = str(variant)
        input_type, _ = FUNC_LOGIT_VARIANT_SPECS[variant_name]
        if input_type != "structure":
            continue
        generation_start = time.perf_counter()
        try:
            raw_tensor, meta = infer_function_variant(
                fasta_id=fasta_id,
                subfolder=subfolder,
                variant=variant_name,
                structure_protein=structure_protein,
                sequence_text=sequence_text,
                model=esm3_model,
                store_raw_artifacts=store_raw_artifacts,
                skip_existing=skip_existing,
            )
            pooling_start = time.perf_counter()
            pooled_tensor = pool_feature_tensor(raw_tensor)
            pooling_seconds = time.perf_counter() - pooling_start
            generation_seconds = pooling_start - generation_start
            meta["generation_seconds"] = generation_seconds
            meta["pooling_seconds"] = pooling_seconds
            meta["total_seconds"] = generation_seconds + pooling_seconds
            function_outputs[variant_name] = meta
            function_vectors[variant_name] = pooled_tensor
            _append_timing_row(
                timing_rows,
                fasta_id=fasta_id,
                variant=variant_name,
                feature_family="function_logit",
                status=str(meta.get("status") or "generated"),
                sequence_length=len(sequence_text),
                structure_length=len(structure_sequence),
                input_type="structure",
                generation_seconds=generation_seconds,
                pooling_seconds=pooling_seconds,
                total_seconds=generation_seconds + pooling_seconds,
                species_dir=species_dir,
                reused_existing=str(meta.get("status")) == "reused_existing",
            )
            sequence_succeeded = True
        except Exception as exc:
            failures.append({"fasta_id": fasta_id, "error": f"{variant_name}: {exc}"})
            function_outputs[variant_name] = {
                "variant": variant_name,
                "status": "failed",
                "error": str(exc),
            }
            _append_timing_row(
                timing_rows,
                fasta_id=fasta_id,
                variant=variant_name,
                feature_family="function_logit",
                status="failed",
                sequence_length=len(sequence_text),
                structure_length=len(structure_sequence),
                input_type="structure",
                generation_seconds=time.perf_counter() - generation_start,
                species_dir=species_dir,
                error=str(exc),
            )

    return function_outputs, function_vectors, structure_outputs, structure_vectors, sequence_succeeded


def _process_feature_task(task: Mapping[str, object]) -> Dict[str, object]:
    from esm.sdk.api import ESMProtein
    from esm.utils.structure.protein_chain import ProteinChain

    fasta_id = str(task["fasta_id"])
    sequence_text = str(task.get("sequence_text") or "")
    subfolder = str(task["subfolder"])
    species_dir = task.get("species_dir")
    struct_file = task.get("struct_file")
    requested_func_variants = list(task.get("requested_func_variants") or [])
    structure_variants = list(task.get("structure_variants") or [])
    store_raw_artifacts = bool(task.get("store_raw_artifacts"))
    skip_existing = bool(task.get("skip_existing"))

    timing_rows: List[Dict[str, object]] = []
    function_outputs: Dict[str, Dict[str, object]] = {}
    structure_outputs: Dict[str, Dict[str, object]] = {}
    function_vectors: Dict[str, torch.Tensor] = {}
    structure_vectors: Dict[str, torch.Tensor] = {}
    failures: List[Dict[str, str]] = []
    sequence_succeeded = False
    worker_memory_before = _memory_snapshot()

    structure_protein = None
    structure_sequence = ""
    if struct_file:
        try:
            chain = ProteinChain.from_pdb(str(struct_file))
            structure_protein = ESMProtein.from_protein_chain(chain)
            structure_protein.function_annotations = None
            structure_sequence = structure_protein.sequence or ""
        except Exception as exc:
            failures.append({"fasta_id": fasta_id, "error": f"Failed to parse structure: {exc}"})
            struct_file = None

    esm3_function_outputs, esm3_function_vectors, esm3_structure_outputs, esm3_structure_vectors, esm3_success = _process_esm3_structure_bundle(
        fasta_id=fasta_id,
        subfolder=subfolder,
        species_dir=species_dir if isinstance(species_dir, str) else None,
        sequence_text=sequence_text,
        structure_protein=structure_protein,
        structure_sequence=structure_sequence,
        structure_variants=structure_variants,
        requested_func_variants=requested_func_variants,
        store_raw_artifacts=store_raw_artifacts,
        skip_existing=skip_existing,
        esm3_model=_get_worker_esm3_model() if (structure_protein is not None and ("esm3_struc_embed" in structure_variants or any(FUNC_LOGIT_VARIANT_SPECS[str(v)][0] == "structure" for v in requested_func_variants))) else None,
        timing_rows=timing_rows,
        failures=failures,
    )
    function_outputs.update(esm3_function_outputs)
    function_vectors.update(esm3_function_vectors)
    structure_outputs.update(esm3_structure_outputs)
    structure_vectors.update(esm3_structure_vectors)
    sequence_succeeded = sequence_succeeded or esm3_success

    for variant in requested_func_variants:
        input_type, _ = FUNC_LOGIT_VARIANT_SPECS[str(variant)]
        if input_type == "structure":
            continue
        generation_start = time.perf_counter()
        try:
            raw_tensor, meta = infer_function_variant(
                fasta_id=fasta_id,
                subfolder=subfolder,
                variant=str(variant),
                structure_protein=structure_protein,
                sequence_text=sequence_text,
                model=_get_worker_esm3_model(),
                store_raw_artifacts=store_raw_artifacts,
                skip_existing=skip_existing,
            )
            pooling_start = time.perf_counter()
            pooled_tensor = pool_feature_tensor(raw_tensor)
            pooling_seconds = time.perf_counter() - pooling_start
            generation_seconds = pooling_start - generation_start
            meta["generation_seconds"] = generation_seconds
            meta["pooling_seconds"] = pooling_seconds
            meta["total_seconds"] = generation_seconds + pooling_seconds
            function_outputs[str(variant)] = meta
            function_vectors[str(variant)] = pooled_tensor
            _append_timing_row(
                timing_rows,
                fasta_id=fasta_id,
                variant=str(variant),
                feature_family="function_logit",
                status=str(meta.get("status") or "generated"),
                sequence_length=len(sequence_text),
                structure_length=len(structure_sequence),
                input_type=input_type,
                generation_seconds=generation_seconds,
                pooling_seconds=pooling_seconds,
                total_seconds=generation_seconds + pooling_seconds,
                species_dir=species_dir if isinstance(species_dir, str) else None,
                reused_existing=str(meta.get("status")) == "reused_existing",
            )
            sequence_succeeded = True
        except Exception as exc:
            failures.append({"fasta_id": fasta_id, "error": f"{variant}: {exc}"})
            function_outputs[str(variant)] = {"variant": str(variant), "status": "failed", "error": str(exc)}
            _append_timing_row(
                timing_rows,
                fasta_id=fasta_id,
                variant=str(variant),
                feature_family="function_logit",
                status="failed",
                sequence_length=len(sequence_text),
                structure_length=len(structure_sequence),
                input_type=input_type,
                generation_seconds=time.perf_counter() - generation_start,
                species_dir=species_dir if isinstance(species_dir, str) else None,
                error=str(exc),
            )

    for variant in structure_variants:
        variant = str(variant)
        if variant == "esm3_struc_embed":
            continue
        input_type = "sequence"
        generation_start = time.perf_counter()
        try:
            raw_tensor, meta = infer_structure_embedding(
                fasta_id=fasta_id,
                subfolder=subfolder,
                variant=variant,
                structure_protein=structure_protein,
                structure_sequence=structure_sequence,
                sequence_text=sequence_text,
                esm3_model=None,
                esmc_model=_get_worker_esmc_model() if variant == "esmc_seq_embed" else None,
                store_raw_artifacts=store_raw_artifacts,
                skip_existing=skip_existing,
            )
            pooling_start = time.perf_counter()
            pooled_tensor = pool_feature_tensor(raw_tensor)
            pooling_seconds = time.perf_counter() - pooling_start
            generation_seconds = pooling_start - generation_start
            meta["generation_seconds"] = generation_seconds
            meta["pooling_seconds"] = pooling_seconds
            meta["total_seconds"] = generation_seconds + pooling_seconds
            structure_outputs[variant] = meta
            structure_vectors[variant] = pooled_tensor
            _append_timing_row(
                timing_rows,
                fasta_id=fasta_id,
                variant=variant,
                feature_family="structure_embedding",
                status=str(meta.get("status") or "generated"),
                sequence_length=len(sequence_text),
                structure_length=len(structure_sequence),
                input_type=input_type,
                generation_seconds=generation_seconds,
                pooling_seconds=pooling_seconds,
                total_seconds=generation_seconds + pooling_seconds,
                species_dir=species_dir if isinstance(species_dir, str) else None,
                reused_existing=str(meta.get("status")) == "reused_existing",
            )
            sequence_succeeded = True
        except Exception as exc:
            failures.append({"fasta_id": fasta_id, "error": f"{variant}: {exc}"})
            structure_outputs[variant] = {"variant": variant, "status": "failed", "error": str(exc)}
            _append_timing_row(
                timing_rows,
                fasta_id=fasta_id,
                variant=variant,
                feature_family="structure_embedding",
                status="failed",
                sequence_length=len(sequence_text),
                structure_length=len(structure_sequence),
                input_type=input_type,
                generation_seconds=time.perf_counter() - generation_start,
                species_dir=species_dir if isinstance(species_dir, str) else None,
                error=str(exc),
            )

    return {
        "fasta_id": fasta_id,
        "species_dir": species_dir,
        "subfolder": subfolder,
        "structure_file": struct_file,
        "function_outputs": function_outputs,
        "structure_outputs": structure_outputs,
        "function_vectors": function_vectors,
        "structure_vectors": structure_vectors,
        "status": "features_generated" if sequence_succeeded else "no_features_generated",
        "failures": failures,
        "timing_rows": timing_rows,
        "worker_pid": os.getpid(),
        "worker_memory_before": worker_memory_before,
        "worker_memory_after": _memory_snapshot(),
    }


def collect_requirements(args: argparse.Namespace) -> Tuple[List[Dict[str, str]], List[str], List[str]]:
    bundle_inputs = list(args.pickled_models or [])
    if args.pickled_models_dir:
        bundle_inputs.append(args.pickled_models_dir)
    bundle_paths = discover_model_bundle_paths(bundle_inputs)

    model_specs: List[Dict[str, str]] = []
    for path in bundle_paths:
        payload = load_model_bundle(path)
        mode_alias = resolve_mode_alias(str(payload.get("mode_alias") or payload.get("mode") or ""))
        primary_variant = str(payload.get("primary_variant") or "")
        model_specs.append(
            {
                "bundle_path": path,
                "mode_alias": mode_alias,
                "primary_variant": primary_variant,
            }
        )

    requested_modes = _parse_requested_modes(args.battery_modes)
    requested_variants = _parse_requested_variants(args.function_logit_variants)

    for spec in model_specs:
        requested_modes.append(spec["mode_alias"])
        if spec["mode_alias"] in {"esm3_func_logit", "esm3_concat"} and spec["primary_variant"]:
            requested_variants.append(spec["primary_variant"])

    requested_modes = list(dict.fromkeys(requested_modes))
    requested_variants = list(dict.fromkeys(requested_variants))

    if not requested_modes:
        requested_modes = ["esmc_seq_embed"]
    if any(mode in {"esm3_func_logit", "esm3_concat"} for mode in requested_modes) and not requested_variants:
        requested_variants = ["esm3_func_logit_struc_step10"]

    return model_specs, requested_modes, requested_variants


def load_existing_tensor(path: str) -> Optional[torch.Tensor]:
    if not os.path.isfile(path):
        return None
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, torch.Tensor):
        return None
    return payload.detach().cpu().float()


def infer_function_logits(
    model: ESM3InferenceClient,
    protein: ESMProtein,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[str, int, int]]]:
    from esm.sdk.api import GenerationConfig, LogitsConfig

    function_logit_config = LogitsConfig(
        sequence=True,
        function=True,
        residue_annotations=True,
        return_embeddings=True,
        return_hidden_states=True,
    )
    with torch.no_grad():
        protein_with_function = model.generate(protein, GenerationConfig(track="function", schedule="cosine", strategy="entropy", num_steps=1))
        protein_tensor = model.encode(protein_with_function)
        logits_output = model.logits(protein_tensor, function_logit_config)
    annotations = [
        (ann.label, int(ann.start), int(ann.end))
        for ann in (protein_with_function.function_annotations or [])
    ]
    func_logits = logits_output.logits.function if logits_output.logits else None
    if func_logits is None:
        raise RuntimeError("ESM3 returned no function logits.")
    residue_logits = logits_output.residue_annotation_logits
    return func_logits.detach().cpu(), residue_logits.detach().cpu() if residue_logits is not None else None, annotations


def infer_function_variant(
    *,
    fasta_id: str,
    subfolder: str,
    variant: str,
    structure_protein: ESMProtein,
    sequence_text: Optional[str],
    model: ESM3InferenceClient,
    store_raw_artifacts: bool,
    skip_existing: bool,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    from esm.sdk.api import ESMProtein, GenerationConfig, LogitsConfig

    function_logit_config = LogitsConfig(
        sequence=True,
        function=True,
        residue_annotations=True,
        return_embeddings=True,
        return_hidden_states=True,
    )
    input_type, num_steps = FUNC_LOGIT_VARIANT_SPECS[variant]
    emb_dir = os.path.join(subfolder, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    raw_path = os.path.join(emb_dir, f"{fasta_id}_function_logits_{input_type}_steps{num_steps}.pt")
    residue_path = os.path.join(emb_dir, f"{fasta_id}_residue_annotation_logits_{input_type}_steps{num_steps}.pt")
    tokens_path = os.path.join(emb_dir, f"{fasta_id}_function_tokens_{input_type}_steps{num_steps}.txt")

    existing = load_existing_tensor(raw_path) if skip_existing else None
    if existing is not None:
        return existing, {"variant": variant, "status": "reused_existing", "raw_path": raw_path}

    if input_type == "structure":
        protein = structure_protein
    else:
        if not sequence_text:
            raise RuntimeError(
                f"Variant '{variant}' requires sequence input but no FASTA sequence was available for {fasta_id}."
            )
        protein = ESMProtein(sequence=sequence_text)
        protein.function_annotations = None

    with torch.no_grad():
        generation_config = GenerationConfig(
            track="function",
            schedule="cosine",
            strategy="entropy",
            num_steps=num_steps,
        )
        protein_with_function = model.generate(protein, generation_config)
        protein_tensor = model.encode(protein_with_function)
        logits_output = model.logits(protein_tensor, function_logit_config)

    func_logits = logits_output.logits.function if logits_output.logits else None
    if func_logits is None:
        raise RuntimeError(f"ESM3 returned no function logits for {fasta_id} ({variant}).")
    func_logits = func_logits.detach().cpu()
    residue_logits = logits_output.residue_annotation_logits
    annotations = protein_with_function.function_annotations or []

    if store_raw_artifacts:
        with open(tokens_path, "w", encoding="utf-8") as handle:
            if not annotations:
                handle.write("No function annotations predicted.\n")
            else:
                for ann in annotations:
                    handle.write(f"{ann.label}\t{ann.start}\t{ann.end}\n")
        torch.save(func_logits, raw_path)
        if residue_logits is not None:
            torch.save(residue_logits.detach().cpu(), residue_path)

    return func_logits, {
        "variant": variant,
        "status": "generated",
        "raw_path": raw_path if store_raw_artifacts else None,
    }


def infer_structure_embedding(
    *,
    fasta_id: str,
    subfolder: str,
    variant: str,
    structure_protein: ESMProtein,
    structure_sequence: str,
    sequence_text: Optional[str],
    esm3_model: Optional[ESM3InferenceClient],
    esmc_model: Optional[ESMCInferenceClient],
    store_raw_artifacts: bool,
    skip_existing: bool,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    from esm.sdk.api import ESMProtein, LogitsConfig

    esm3_feature_config = LogitsConfig(sequence=True, return_embeddings=True)
    esmc_feature_config = LogitsConfig(sequence=True, return_embeddings=True)
    output_dir = os.path.join(subfolder, "esm_out_embeddings")
    os.makedirs(output_dir, exist_ok=True)

    if variant == "esm3_struc_embed":
        raw_path = os.path.join(output_dir, f"{fasta_id}_esm3_embedding_structure_steps1.pt")
        existing = load_existing_tensor(raw_path) if skip_existing else None
        if existing is not None:
            return existing, {"variant": variant, "status": "reused_existing", "raw_path": raw_path}
        if esm3_model is None:
            raise RuntimeError("ESM3 model not initialized for structure embeddings.")
        with torch.no_grad():
            protein_tensor = esm3_model.encode(structure_protein)
            logits_output = esm3_model.logits(protein_tensor, esm3_feature_config)
        embeddings = logits_output.embeddings
        if embeddings is None:
            raise RuntimeError(f"ESM3 returned no structure embeddings for {fasta_id}.")
        tensor = embeddings[0].detach().cpu()
    elif variant == "esmc_seq_embed":
        raw_path = os.path.join(output_dir, f"{fasta_id}_esmc_embedding_sequence_steps1.pt")
        existing = load_existing_tensor(raw_path) if skip_existing else None
        if existing is not None:
            return existing, {"variant": variant, "status": "reused_existing", "raw_path": raw_path}
        if esmc_model is None:
            raise RuntimeError("ESMC model not initialized for sequence embeddings.")
        effective_sequence = sequence_text or structure_sequence
        if not effective_sequence:
            raise RuntimeError(f"No sequence available for ESMC embedding of {fasta_id}.")
        with torch.no_grad():
            protein = ESMProtein(sequence=effective_sequence)
            protein_tensor = esmc_model.encode(protein)
            logits_output = esmc_model.logits(protein_tensor, esmc_feature_config)
        embeddings = logits_output.embeddings
        if embeddings is None:
            raise RuntimeError(f"ESMC returned no sequence embeddings for {fasta_id}.")
        tensor = embeddings[0].detach().cpu()
    else:
        raise KeyError(f"Unsupported structure-embedding variant '{variant}'")

    if store_raw_artifacts:
        torch.save(tensor, raw_path)
    return tensor, {
        "variant": variant,
        "status": "generated",
        "raw_path": raw_path if store_raw_artifacts else None,
    }


def build_feature_cache(
    *,
    fasta_ids: Sequence[str],
    function_vectors_by_variant: Mapping[str, List[torch.Tensor]],
    function_fasta_ids_by_variant: Mapping[str, Sequence[str]],
    structure_vectors_by_variant: Mapping[str, List[torch.Tensor]],
    structure_fasta_ids_by_variant: Mapping[str, Sequence[str]],
    raw_records: Mapping[str, Dict[str, object]],
    requested_modes: Sequence[str],
    model_specs: Sequence[Mapping[str, str]],
    structures_dir: Optional[str],
) -> Dict[str, object]:
    function_matrix: Dict[str, torch.Tensor] = {
        variant: torch.stack(vectors, dim=0).float()
        for variant, vectors in function_vectors_by_variant.items()
        if vectors
    }
    structure_matrix: Dict[str, torch.Tensor] = {
        variant: torch.stack(vectors, dim=0).float()
        for variant, vectors in structure_vectors_by_variant.items()
        if vectors
    }
    return {
        "cache_version": CACHE_VERSION,
        "fasta_ids": list(fasta_ids),
        "source_structures_dir": structures_dir,
        "requested_modes": list(requested_modes),
        "selected_model_specs": [dict(spec) for spec in model_specs],
        "function_logit_vectors_by_variant": function_matrix,
        "function_logit_fasta_ids_by_variant": {
            variant: list(function_fasta_ids_by_variant.get(variant, []))
            for variant in function_vectors_by_variant
        },
        "structure_embedding_vectors_by_variant": structure_matrix,
        "structure_embedding_fasta_ids_by_variant": {
            variant: list(structure_fasta_ids_by_variant.get(variant, []))
            for variant in structure_vectors_by_variant
        },
        "coverage_summary": {
            "total_fasta_ids": int(len(fasta_ids)),
            "function_logit_counts_by_variant": {
                variant: int(len(function_fasta_ids_by_variant.get(variant, [])))
                for variant in function_vectors_by_variant
            },
            "structure_embedding_counts_by_variant": {
                variant: int(len(structure_fasta_ids_by_variant.get(variant, [])))
                for variant in structure_vectors_by_variant
            },
        },
        "raw_records": dict(raw_records),
    }


def main() -> None:
    args = parse_args()
    from dotenv import load_dotenv

    load_dotenv()
    args.device = resolve_device(args.device)
    if args.device != "cpu" and int(args.inference_workers) != 1:
        print("[WARNING] Inference worker parallelism is only enabled on CPU; forcing --inference_workers=1.")
        args.inference_workers = 1
    args.inference_workers = max(1, int(args.inference_workers))
    model_specs, requested_modes, requested_func_variants = collect_requirements(args)
    feature_cache_path = args.feature_cache or "ML/prediction/unchar_BAHD_features.pt"
    raw_artifacts_dir = (
        args.raw_artifacts_dir
        if args.raw_artifacts_dir
        else (args.structures_dir if args.structures_dir else "ML/prediction/unchar_raw_features")
    )
    prediction_log = args.prediction_log or os.path.join(os.path.dirname(feature_cache_path) or ".", "prediction_log.json")
    timing_tsv = args.timing_tsv or f"{os.path.splitext(feature_cache_path)[0]}.timing.tsv"

    needs_esm3 = bool(requested_func_variants) or ("esm3_struc_embed" in requested_modes)
    needs_esmc = "esmc_seq_embed" in requested_modes

    fasta_sequences = load_fasta_sequences(args.fasta)
    if not fasta_sequences:
        raise SystemExit("--fasta is required and must contain sequences for feature generation.")
    if any(variant.endswith("_seq_step1") or variant.endswith("_seq_step10") for variant in requested_func_variants):
        if not args.fasta:
            raise SystemExit(
                "Sequence-based function-logit variants were requested but --fasta was not provided."
            )
    if (needs_esm3 or "esm3_concat" in requested_modes or "esm3_struc_embed" in requested_modes) and not args.structures_dir:
        raise SystemExit(
            "Structure-aware modes require --funcpred_structures_dir. "
            "For sequence-only generation, use --battery_modes esmc_seq_embed."
        )

    esm3_model: Optional[ESM3InferenceClient] = None
    esmc_model: Optional[ESMCInferenceClient] = None
    use_cpu_worker_pool = args.device == "cpu" and args.inference_workers > 1
    if not use_cpu_worker_pool:
        if needs_esm3:
            esm3_model = initialize_esm3(args.device)
            esm3_model.eval()
        if needs_esmc:
            esmc_model = initialize_esmc(args.device)
            esmc_model.eval()

    structure_variants: List[str] = []
    if "esm3_concat" in requested_modes or "esm3_struc_embed" in requested_modes:
        structure_variants.append("esm3_struc_embed")
    if "esmc_seq_embed" in requested_modes:
        structure_variants.append("esmc_seq_embed")

    function_vectors_by_variant: Dict[str, List[torch.Tensor]] = {variant: [] for variant in requested_func_variants}
    function_fasta_ids_by_variant: Dict[str, List[str]] = {variant: [] for variant in requested_func_variants}
    structure_vectors_by_variant: Dict[str, List[torch.Tensor]] = {variant: [] for variant in structure_variants}
    structure_fasta_ids_by_variant: Dict[str, List[str]] = {variant: [] for variant in structure_variants}
    raw_records: Dict[str, Dict[str, object]] = {}
    failures: List[Dict[str, str]] = []
    timing_rows: List[Dict[str, object]] = []

    structure_context: Dict[str, Tuple[str, Optional[str], Optional[ESMProtein], str, Optional[str]]] = {}
    if args.structures_dir:
        fasta_dirs = list(iter_fasta_directories(args.structures_dir))
        if not fasta_dirs and needs_esm3:
            raise SystemExit(f"No FASTA structure folders found under {args.structures_dir}")
        parse_workers = max(1, int(args.structure_parse_workers))
        if parse_workers == 1 or len(fasta_dirs) <= 1:
            parsed_entries = [
                _load_structure_context_entry(fasta_id, subfolder, species_dir)
                for fasta_id, subfolder, species_dir in fasta_dirs
            ]
        else:
            parsed_entries = []
            with ThreadPoolExecutor(max_workers=parse_workers) as executor:
                future_map = {
                    executor.submit(_load_structure_context_entry, fasta_id, subfolder, species_dir): fasta_id
                    for fasta_id, subfolder, species_dir in fasta_dirs
                }
                for future in as_completed(future_map):
                    parsed_entries.append(future.result())
        for entry in parsed_entries:
            fasta_id = str(entry["fasta_id"])
            error = entry.get("error")
            if error:
                if needs_esm3:
                    failures.append({"fasta_id": fasta_id, "error": str(error)})
                continue
            structure_context[fasta_id] = (
                str(entry["subfolder"]),
                entry.get("species_dir"),
                entry["structure_protein"],
                str(entry["structure_sequence"]),
                str(entry["struct_file"]),
            )

    fasta_ids_to_process = sorted(fasta_sequences)
    if args.worker_smoke_test:
        fasta_ids_to_process = fasta_ids_to_process[: max(1, int(args.worker_smoke_test_count))]
        print(
            f"[INFO] Worker smoke test enabled: processing {len(fasta_ids_to_process)} sequence(s)."
        )

    tasks: List[Dict[str, object]] = []
    for fasta_id in fasta_ids_to_process:
        sequence_text = fasta_sequences.get(fasta_id)
        if not sequence_text:
            continue
        context = structure_context.get(fasta_id)
        if context is None:
            subfolder = os.path.join(raw_artifacts_dir, fasta_id)
            species_dir = None
            structure_protein = None
            structure_sequence = ""
            struct_file = None
        else:
            subfolder, species_dir, structure_protein, structure_sequence, struct_file = context

        raw_records[fasta_id] = {
            "fasta_id": fasta_id,
            "species_dir": species_dir,
            "subfolder": subfolder,
            "structure_file": struct_file,
            "function_variants": {},
            "structure_variants": {},
        }
        tasks.append(
            {
                "fasta_id": fasta_id,
                "sequence_text": sequence_text,
                "subfolder": subfolder,
                "species_dir": species_dir,
                "struct_file": struct_file,
                "requested_func_variants": list(requested_func_variants),
                "structure_variants": list(structure_variants),
                "store_raw_artifacts": bool(args.store_raw_artifacts),
                "skip_existing": bool(args.skip_existing),
            }
        )

    worker_memory_reports: List[Dict[str, object]] = []
    if use_cpu_worker_pool:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=args.inference_workers,
            initializer=_worker_initialize,
            initargs=(args.device,),
        ) as pool:
            for result in pool.imap_unordered(_process_feature_task, tasks):
                fasta_id = str(result["fasta_id"])
                raw_records[fasta_id]["function_variants"] = dict(result.get("function_outputs") or {})
                raw_records[fasta_id]["structure_variants"] = dict(result.get("structure_outputs") or {})
                raw_records[fasta_id]["status"] = str(result.get("status") or "no_features_generated")
                for variant, tensor in (result.get("function_vectors") or {}).items():
                    function_vectors_by_variant[str(variant)].append(tensor)
                    function_fasta_ids_by_variant[str(variant)].append(fasta_id)
                for variant, tensor in (result.get("structure_vectors") or {}).items():
                    structure_vectors_by_variant[str(variant)].append(tensor)
                    structure_fasta_ids_by_variant[str(variant)].append(fasta_id)
                failures.extend(list(result.get("failures") or []))
                timing_rows.extend(list(result.get("timing_rows") or []))
                worker_memory_reports.append(
                    {
                        "fasta_id": fasta_id,
                        "worker_pid": int(result.get("worker_pid") or 0),
                        "memory_before": result.get("worker_memory_before") or {},
                        "memory_after": result.get("worker_memory_after") or {},
                    }
                )
    else:
        for task in tasks:
            fasta_id = str(task["fasta_id"])
            sequence_succeeded = False
            sequence_text = str(task.get("sequence_text") or "")
            species_dir = task.get("species_dir")
            structure_sequence = ""
            struct_file = task.get("struct_file")
            structure_protein = None
            if struct_file:
                context = structure_context.get(fasta_id)
                if context is not None:
                    _, _, structure_protein, structure_sequence, _ = context
            esm3_function_outputs, esm3_function_vectors, esm3_structure_outputs, esm3_structure_vectors, esm3_success = _process_esm3_structure_bundle(
                fasta_id=fasta_id,
                subfolder=str(task["subfolder"]),
                species_dir=species_dir if isinstance(species_dir, str) else None,
                sequence_text=sequence_text,
                structure_protein=structure_protein,
                structure_sequence=structure_sequence,
                structure_variants=structure_variants,
                requested_func_variants=requested_func_variants,
                store_raw_artifacts=args.store_raw_artifacts,
                skip_existing=args.skip_existing,
                esm3_model=esm3_model,
                timing_rows=timing_rows,
                failures=failures,
            )
            raw_records[fasta_id]["function_variants"].update(esm3_function_outputs)
            raw_records[fasta_id]["structure_variants"].update(esm3_structure_outputs)
            for variant, tensor in esm3_function_vectors.items():
                function_vectors_by_variant[variant].append(tensor)
                function_fasta_ids_by_variant[variant].append(fasta_id)
            for variant, tensor in esm3_structure_vectors.items():
                structure_vectors_by_variant[variant].append(tensor)
                structure_fasta_ids_by_variant[variant].append(fasta_id)
            sequence_succeeded = sequence_succeeded or esm3_success
            for variant in requested_func_variants:
                input_type, _ = FUNC_LOGIT_VARIANT_SPECS[variant]
                if input_type == "structure":
                    continue
                generation_start = time.perf_counter()
                try:
                    raw_tensor, meta = infer_function_variant(
                        fasta_id=fasta_id,
                        subfolder=str(task["subfolder"]),
                        variant=variant,
                        structure_protein=structure_protein,
                        sequence_text=sequence_text,
                        model=esm3_model,
                        store_raw_artifacts=args.store_raw_artifacts,
                        skip_existing=args.skip_existing,
                    )
                    generation_seconds = time.perf_counter() - generation_start
                    _summarize_tensor(variant, fasta_id, raw_tensor)
                    pooling_start = time.perf_counter()
                    pooled_tensor = pool_feature_tensor(raw_tensor)
                    pooling_seconds = time.perf_counter() - pooling_start
                    function_vectors_by_variant[variant].append(pooled_tensor)
                    function_fasta_ids_by_variant[variant].append(fasta_id)
                    meta["generation_seconds"] = generation_seconds
                    meta["pooling_seconds"] = pooling_seconds
                    meta["total_seconds"] = generation_seconds + pooling_seconds
                    raw_records[fasta_id]["function_variants"][variant] = meta
                    _append_timing_row(
                        timing_rows,
                        fasta_id=fasta_id,
                        variant=variant,
                        feature_family="function_logit",
                        status=str(meta.get("status") or "generated"),
                        sequence_length=len(sequence_text),
                        structure_length=len(structure_sequence),
                        input_type=input_type,
                        generation_seconds=generation_seconds,
                        pooling_seconds=pooling_seconds,
                        total_seconds=generation_seconds + pooling_seconds,
                        species_dir=species_dir if isinstance(species_dir, str) else None,
                        reused_existing=str(meta.get("status")) == "reused_existing",
                    )
                    sequence_succeeded = True
                except Exception as exc:
                    failures.append({"fasta_id": fasta_id, "error": f"{variant}: {exc}"})
                    raw_records[fasta_id]["function_variants"][variant] = {
                        "variant": variant,
                        "status": "failed",
                        "error": str(exc),
                    }
                    _append_timing_row(
                        timing_rows,
                        fasta_id=fasta_id,
                        variant=variant,
                        feature_family="function_logit",
                        status="failed",
                        sequence_length=len(sequence_text),
                        structure_length=len(structure_sequence),
                        input_type=input_type,
                        generation_seconds=time.perf_counter() - generation_start,
                        species_dir=species_dir if isinstance(species_dir, str) else None,
                        error=str(exc),
                    )

            for variant in structure_variants:
                if variant == "esm3_struc_embed":
                    continue
                input_type = "sequence"
                generation_start = time.perf_counter()
                try:
                    raw_tensor, meta = infer_structure_embedding(
                        fasta_id=fasta_id,
                        subfolder=str(task["subfolder"]),
                        variant=variant,
                        structure_protein=structure_protein,
                        structure_sequence=structure_sequence,
                        sequence_text=sequence_text,
                        esm3_model=None,
                        esmc_model=esmc_model,
                        store_raw_artifacts=args.store_raw_artifacts,
                        skip_existing=args.skip_existing,
                    )
                    generation_seconds = time.perf_counter() - generation_start
                    _summarize_tensor(variant, fasta_id, raw_tensor)
                    pooling_start = time.perf_counter()
                    pooled_tensor = pool_feature_tensor(raw_tensor)
                    pooling_seconds = time.perf_counter() - pooling_start
                    structure_vectors_by_variant[variant].append(pooled_tensor)
                    structure_fasta_ids_by_variant[variant].append(fasta_id)
                    meta["generation_seconds"] = generation_seconds
                    meta["pooling_seconds"] = pooling_seconds
                    meta["total_seconds"] = generation_seconds + pooling_seconds
                    raw_records[fasta_id]["structure_variants"][variant] = meta
                    _append_timing_row(
                        timing_rows,
                        fasta_id=fasta_id,
                        variant=variant,
                        feature_family="structure_embedding",
                        status=str(meta.get("status") or "generated"),
                        sequence_length=len(sequence_text),
                        structure_length=len(structure_sequence),
                        input_type=input_type,
                        generation_seconds=generation_seconds,
                        pooling_seconds=pooling_seconds,
                        total_seconds=generation_seconds + pooling_seconds,
                        species_dir=species_dir if isinstance(species_dir, str) else None,
                        reused_existing=str(meta.get("status")) == "reused_existing",
                    )
                    sequence_succeeded = True
                except Exception as exc:
                    failures.append({"fasta_id": fasta_id, "error": f"{variant}: {exc}"})
                    raw_records[fasta_id]["structure_variants"][variant] = {
                        "variant": variant,
                        "status": "failed",
                        "error": str(exc),
                    }
                    _append_timing_row(
                        timing_rows,
                        fasta_id=fasta_id,
                        variant=variant,
                        feature_family="structure_embedding",
                        status="failed",
                        sequence_length=len(sequence_text),
                        structure_length=len(structure_sequence),
                        input_type=input_type,
                        generation_seconds=time.perf_counter() - generation_start,
                        species_dir=species_dir if isinstance(species_dir, str) else None,
                        error=str(exc),
                    )

            raw_records[fasta_id]["status"] = "features_generated" if sequence_succeeded else "no_features_generated"

    generated_feature_count = sum(len(vectors) for vectors in function_vectors_by_variant.values()) + sum(
        len(vectors) for vectors in structure_vectors_by_variant.values()
    )
    if generated_feature_count == 0:
        raise SystemExit("No sequences were processed successfully; feature cache not written.")

    master_fasta_ids = [str(task["fasta_id"]) for task in tasks]
    cache = build_feature_cache(
        fasta_ids=master_fasta_ids,
        function_vectors_by_variant=function_vectors_by_variant,
        function_fasta_ids_by_variant=function_fasta_ids_by_variant,
        structure_vectors_by_variant=structure_vectors_by_variant,
        structure_fasta_ids_by_variant=structure_fasta_ids_by_variant,
        raw_records=raw_records,
        requested_modes=requested_modes,
        model_specs=model_specs,
        structures_dir=args.structures_dir,
    )
    os.makedirs(os.path.dirname(feature_cache_path) or ".", exist_ok=True)
    torch.save(cache, feature_cache_path)
    os.makedirs(os.path.dirname(timing_tsv) or ".", exist_ok=True)
    with open(timing_tsv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fasta_id",
                "variant",
                "feature_family",
                "status",
                "species_dir",
                "input_type",
                "sequence_length",
                "structure_length",
                "generation_seconds",
                "pooling_seconds",
                "total_seconds",
                "reused_existing",
                "error",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(timing_rows)

    worker_memory_summary: Dict[str, object] = {}
    if worker_memory_reports:
        rss_before = [
            float(report.get("memory_before", {}).get("rss_gb"))
            for report in worker_memory_reports
            if report.get("memory_before", {}).get("rss_gb") is not None
        ]
        rss_after = [
            float(report.get("memory_after", {}).get("rss_gb"))
            for report in worker_memory_reports
            if report.get("memory_after", {}).get("rss_gb") is not None
        ]
        worker_memory_summary = {
            "reports": worker_memory_reports,
            "max_rss_before_gb": max(rss_before) if rss_before else None,
            "max_rss_after_gb": max(rss_after) if rss_after else None,
            "worker_pids": sorted(
                {int(report.get("worker_pid") or 0) for report in worker_memory_reports if int(report.get("worker_pid") or 0) > 0}
            ),
        }

    log_entry = {
        "time": datetime.utcnow().isoformat(),
        "person": args.person,
        "args": {
            "funcpred_structures_dir": args.structures_dir,
            "raw_artifacts_dir": raw_artifacts_dir,
            "fasta": args.fasta,
            "device": args.device,
            "store_raw_artifacts": bool(args.store_raw_artifacts),
            "skip_existing": bool(args.skip_existing),
            "feature_cache": feature_cache_path,
            "timing_tsv": timing_tsv,
            "structure_parse_workers": int(args.structure_parse_workers),
            "inference_workers": int(args.inference_workers),
            "worker_smoke_test": bool(args.worker_smoke_test),
            "worker_smoke_test_count": int(args.worker_smoke_test_count),
        },
        "requested_modes": requested_modes,
        "requested_function_logit_variants": requested_func_variants,
        "selected_model_specs": model_specs,
        "processed_ids": [
            fasta_id
            for fasta_id, record in raw_records.items()
            if str(record.get("status")) == "features_generated"
        ],
        "coverage_summary": cache.get("coverage_summary", {}),
        "worker_memory_summary": worker_memory_summary,
        "failures": failures,
    }
    append_prediction_log(prediction_log, "predict_step2_generate_embeddings", log_entry)
    print(
        f"[INFO] Wrote pooled feature cache for {len(master_fasta_ids)} master sequences to {feature_cache_path}"
    )
    print(f"[INFO] Wrote per-feature timing TSV to {timing_tsv}")
    if worker_memory_summary:
        print(
            "[INFO] Worker memory summary: "
            f"max_rss_before_gb={worker_memory_summary.get('max_rss_before_gb')} "
            f"max_rss_after_gb={worker_memory_summary.get('max_rss_after_gb')} "
            f"workers={worker_memory_summary.get('worker_pids')}"
        )


if __name__ == "__main__":
    main()
