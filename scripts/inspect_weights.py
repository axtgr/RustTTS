#!/usr/bin/env python3
"""
Inspect safetensors model weights.

Usage:
    python scripts/inspect_weights.py models/qwen3-tts-0.6b/model.safetensors
    python scripts/inspect_weights.py models/qwen3-tts-tokenizer/model.safetensors
"""

import sys
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    print("Error: safetensors not installed")
    print("Install with: pip install safetensors")
    sys.exit(1)


def format_size(num_bytes: float) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def inspect_safetensors(file_path: str, verbose: bool = False) -> dict:
    """Inspect a safetensors file and return info about tensors."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"Inspecting: {path}")
    print(f"File size: {format_size(path.stat().st_size)}")
    print(f"{'=' * 60}\n")

    with safe_open(str(path), framework="pt") as f:
        keys = list(f.keys())

        print(f"Total tensors: {len(keys)}\n")

        # Group by prefix
        groups: dict = {}
        total_params = 0

        for key in keys:
            tensor = f.get_tensor(key)
            shape = tuple(tensor.shape)
            dtype = str(tensor.dtype)
            num_params = tensor.numel()
            total_params += num_params

            # Get prefix (first part before .)
            prefix = key.split(".")[0]
            if prefix not in groups:
                groups[prefix] = {"count": 0, "params": 0, "tensors": []}
            groups[prefix]["count"] += 1
            groups[prefix]["params"] += num_params

            if verbose:
                groups[prefix]["tensors"].append(
                    {"name": key, "shape": shape, "dtype": dtype, "params": num_params}
                )

        # Print summary by group
        print("Tensor groups:")
        print("-" * 50)
        for prefix, info in sorted(groups.items()):
            params_str = format_size(info["params"] * 4)  # Assume float32
            print(f"  {prefix:30} {info['count']:5} tensors, {params_str:>10}")

        print("-" * 50)
        print(
            f"  {'TOTAL':30} {len(keys):5} tensors, {format_size(total_params * 4):>10}"
        )
        print(f"\nTotal parameters: {total_params:,}")

        if verbose:
            print("\n\nDetailed tensor list:")
            print("-" * 80)
            for key in sorted(keys):
                tensor = f.get_tensor(key)
                shape = tuple(tensor.shape)
                dtype = str(tensor.dtype)
                print(f"  {key:60} {str(shape):20} {dtype}")

        return {
            "path": str(path),
            "file_size": path.stat().st_size,
            "num_tensors": len(keys),
            "total_params": total_params,
            "groups": groups,
        }


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    paths = [p for p in sys.argv[1:] if not p.startswith("-")]

    for path in paths:
        inspect_safetensors(path, verbose=verbose)


if __name__ == "__main__":
    main()
