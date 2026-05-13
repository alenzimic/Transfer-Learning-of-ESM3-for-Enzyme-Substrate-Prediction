#!/usr/bin/env python3
"""
extract_esm3_hidden_layer.py

For each PDB file found under --input_dir, run the same ESM3 function-
generation pass used by FuncPred-AI_step2_generate_features.py, but save the
FINAL HIDDEN LAYER (the trunk output that feeds the function regression head)
instead of the function logits.

Output per input <NAME>.pdb:
    <output_dir>/<NAME>_hidden_layer_steps<N>.pt
containing a torch.Tensor of shape (seq_len, hidden_dim).

Typical usage:
    python extract_esm3_hidden_layer.py \
        --input_dir structure/unchar_structures \
        --output_dir ML/prediction/unchar_hidden_features \
        --num_steps 10 \
        --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ESM3 last hidden layer (pre function-head) for each PDB in a folder."
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Directory containing PDB files. Searched recursively.",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory where per-input hidden-layer tensors are written.",
    )
    parser.add_argument(
        "--num_steps", type=int, default=10,
        help="ESM3 function-generation steps (matches '_step10' in your pipeline). Default: 10.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device: cuda, cpu, or mps.",
    )
    parser.add_argument(
        "--pattern", default="*.pdb",
        help="Glob pattern for input files. Default: *.pdb",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip inputs whose output file already exists.",
    )
    parser.add_argument(
        "--save_as_numpy", action="store_true",
        help="Save as .npy instead of .pt.",
    )
    parser.add_argument(
        "--id_from", choices=["filename", "parent", "grandparent"], default="filename",
        help=(
            "How to derive the output ID. 'filename' uses the PDB stem. "
            "'parent' uses the PDB's parent folder. 'grandparent' uses the folder "
            "two levels up (matches the FuncPred layout <fasta_id>/structure_file/x.pdb)."
        ),
    )
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


def hf_login_if_needed() -> None:
    import huggingface_hub
    from dotenv import load_dotenv

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if token is None:
        raise RuntimeError(
            "HF_TOKEN not found in environment. Create a .env file with HF_TOKEN=<your_token> "
            "or export HF_TOKEN before running."
        )
    huggingface_hub.login(token=token, add_to_git_credential=False)


def load_esm3(device: str):
    from esm.models.esm3 import ESM3
    try:
        model = ESM3.from_pretrained("esm3-open").to(device)
    except (FileNotFoundError, OSError):
        hf_login_if_needed()
        model = ESM3.from_pretrained("esm3-open").to(device)
    model.eval()
    return model


def extract_last_hidden(model, pdb_path: Path, num_steps: int) -> torch.Tensor:
    """Run the function-generation pass and return the final hidden-layer tensor."""
    from esm.sdk.api import ESMProtein, GenerationConfig, LogitsConfig
    from esm.utils.structure.protein_chain import ProteinChain

    chain = ProteinChain.from_pdb(str(pdb_path))
    protein = ESMProtein.from_protein_chain(chain)
    protein.function_annotations = None

    # Same config as FuncPred-AI_step2_generate_features.py.
    # return_hidden_states=True gives us the trunk representations.
    logit_config = LogitsConfig(
        sequence=True,
        function=True,
        residue_annotations=True,
        return_embeddings=True,
        return_hidden_states=True,
    )

    with torch.no_grad():
        gen_config = GenerationConfig(
            track="function",
            schedule="cosine",
            strategy="entropy",
            num_steps=num_steps,
        )
        protein_with_function = model.generate(protein, gen_config)
        protein_tensor = model.encode(protein_with_function)
        output = model.logits(protein_tensor, logit_config)

    # Prefer an explicit last hidden state when present; otherwise fall back to
    # the trunk embeddings. Both represent the final hidden layer that feeds
    # the function regression head in ESM3.
    hidden = None
    hs = getattr(output, "hidden_states", None)
    if hs is not None:
        if isinstance(hs, (list, tuple)):
            hidden = hs[-1]
        elif torch.is_tensor(hs) and hs.ndim >= 4:
            # Stacked (num_layers, batch, seq, dim) -> take last layer
            hidden = hs[-1]
        else:
            hidden = hs
    if hidden is None:
        hidden = output.embeddings
    if hidden is None:
        raise RuntimeError("ESM3 returned neither hidden_states nor embeddings.")

    # Drop a leading batch dim of 1 -> final shape is (seq_len, hidden_dim)
    if torch.is_tensor(hidden) and hidden.ndim == 3 and hidden.shape[0] == 1:
        hidden = hidden[0]

    return hidden.detach().cpu()


def derive_id(pdb_path: Path, input_dir: Path, strategy: str) -> str:
    if strategy == "filename":
        return pdb_path.stem
    if strategy == "parent":
        return pdb_path.parent.name
    if strategy == "grandparent":
        return pdb_path.parent.parent.name
    return pdb_path.stem


def main() -> None:
    args = parse_args()

    # Load .env early so HF_TOKEN is available if needed.
    from dotenv import load_dotenv
    load_dotenv()

    args.device = resolve_device(args.device)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"--input_dir not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(input_dir.rglob(args.pattern))
    if not pdb_files:
        raise SystemExit(f"No files matching {args.pattern!r} found under {input_dir}")

    print(f"[INFO] Found {len(pdb_files)} input file(s) under {input_dir}")
    print(f"[INFO] Loading ESM3 on device={args.device} ...")
    t0 = time.perf_counter()
    model = load_esm3(args.device)
    print(f"[INFO] Model ready in {time.perf_counter() - t0:.1f}s")

    suffix = ".npy" if args.save_as_numpy else ".pt"
    n_ok = n_skipped = n_failed = 0

    for i, pdb in enumerate(pdb_files, start=1):
        out_id = derive_id(pdb, input_dir, args.id_from)
        out_path = output_dir / f"{out_id}_hidden_layer_steps{args.num_steps}{suffix}"
        if args.skip_existing and out_path.exists():
            n_skipped += 1
            print(f"[{i}/{len(pdb_files)}] skip  {out_id}  (exists)")
            continue
        try:
            t = time.perf_counter()
            hidden = extract_last_hidden(model, pdb, args.num_steps)
            if args.save_as_numpy:
                import numpy as np
                np.save(out_path, hidden.numpy())
            else:
                torch.save(hidden, out_path)
            elapsed = time.perf_counter() - t
            n_ok += 1
            print(
                f"[{i}/{len(pdb_files)}] ok    {out_id}  "
                f"shape={tuple(hidden.shape)}  dtype={hidden.dtype}  {elapsed:.1f}s"
            )
        except Exception as exc:  # noqa: BLE001 - we want per-file recovery
            n_failed += 1
            print(f"[{i}/{len(pdb_files)}] FAIL  {out_id}: {exc}", file=sys.stderr)

    print(
        f"[DONE] ok={n_ok}  skipped={n_skipped}  failed={n_failed}  "
        f"outputs in {output_dir}"
    )


if __name__ == "__main__":
    main()
