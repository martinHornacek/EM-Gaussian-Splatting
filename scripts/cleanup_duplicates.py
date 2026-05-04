"""Remove duplicate experiment output directories, keeping only the freshest run per K."""
import shutil
from pathlib import Path

BASE = Path("results/paper_comparison/per_run")

TO_REMOVE = [
    # k128: keep 061838
    BASE / "fullbatch_k128/hybrid_residual_20260503_061703",
    BASE / "fullbatch_k128/hybrid_residual_20260503_061747",
    # k256: keep 070145
    BASE / "fullbatch_k256/hybrid_residual_20260503_065959",
    BASE / "fullbatch_k256/hybrid_residual_20260503_070022",
    # k512: keep 083018
    BASE / "fullbatch_k512/hybrid_residual_20260503_082927",
    # k1024: keep 114547
    BASE / "fullbatch_k1024/hybrid_residual_20260503_114459",
    # k2048: no complete results at all
    BASE / "fullbatch_k2048",
]

for p in TO_REMOVE:
    if p.exists():
        shutil.rmtree(p)
        print(f"  removed: {p}")
    else:
        print(f"  not found (already gone?): {p}")

print("\nRemaining structure:")
for d in sorted(BASE.rglob("*")):
    if d.is_dir():
        indent = "  " * (len(d.relative_to(BASE).parts) - 1)
        print(f"  {indent}{d.name}/")
    else:
        indent = "  " * (len(d.relative_to(BASE).parts) - 1)
        print(f"  {indent}{d.name}")
