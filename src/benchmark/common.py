import sys
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = BENCHMARK_ROOT.parents[1]
QMAP_ROOT = REPOSITORY_ROOT / "src" / "qmap"
INSTANCES_ROOT = BENCHMARK_ROOT / "instances"
SOLUTIONS_ROOT = BENCHMARK_ROOT / "solutions"
ANALYSIS_ROOT = BENCHMARK_ROOT / "analysis"


def add_qmap_to_import_path() -> None:
    qmap_root = str(QMAP_ROOT)
    if qmap_root not in sys.path:
        sys.path.insert(0, qmap_root)
