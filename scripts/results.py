"""
Layout:

    results/<experiment>/<label>@<param_hash>/<timestamp>-<commit>-dirty_<build_hash>/
        data.csv
        meta.json
        source.patch      # only when the worktree was dirty

- The build hash is calculated from the compiled *.so file and serves to
  differentiate runs with different uncommitted code versions.
- If matching runs already exist, datapoints are appended instead of
  resimulating.
"""

import csv
import datetime
import functools
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path("results")


#
#
# Run identificationr
#
#


def get_param_hash(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:6]


def _run_git_cmd(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return ""
    return completed.stdout.strip()


def _is_worktree_dirty() -> bool:
    return bool(_run_git_cmd("status", "--porcelain"))


@functools.cache
def get_compiled_binary_hash() -> tuple:
    import rust_qldpc

    module = sys.modules.get("rust_qldpc.rust_qldpc")
    module_file = getattr(module, "__file__", None)
    if module_file:
        path = Path(module_file)
    else:
        path = next(iter(sorted(Path(rust_qldpc.__file__).parent.glob("*.so"))))

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    return str(path), digest.hexdigest()


def get_commit_and_binary_hash() -> str:
    commit = _run_git_cmd("rev-parse", "--short=7", "HEAD") or "nogit"
    return f"{commit}-dirty_{get_compiled_binary_hash()[1][:6]}"


def get_git_stamp() -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{get_commit_and_binary_hash()}"


def _get_stamp_revision(stamp: str) -> str:
    parts = stamp.split("-", 2)
    return parts[2] if len(parts) == 3 else ""


#
#
# Writing
#
#


class RunDir:
    def __init__(self, path: Path):
        self.path = path
        self.data_path = path / "data.csv"

    def has_p(self, p: float) -> bool:
        return any(abs(p - done) < 1e-12 for done in self.done_ps())

    def done_ps(self) -> list:
        if not self.data_path.exists():
            return []
        return [float(p) for p in pd.read_csv(self.data_path)["p"]]

    def append(self, row: dict) -> None:
        """Append one data point, flushing immediately."""

        is_new = not self.data_path.exists()
        with open(self.data_path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if is_new:
                writer.writeheader()
            writer.writerow(row)


def resolve_and_open_run(
    experiment: str, label: str, params: dict, meta=None
) -> RunDir:
    curve_dir = RESULTS_ROOT / experiment / f"{label}@{get_param_hash(params)}"
    revision = get_commit_and_binary_hash()
    build_path, build_hash = get_compiled_binary_hash()

    if curve_dir.exists():
        for stamp in sorted(p.name for p in curve_dir.iterdir() if p.is_dir()):
            if _get_stamp_revision(stamp) == revision:
                return RunDir(curve_dir / stamp)

    run_dir = curve_dir / get_git_stamp()
    run_dir.mkdir(parents=True)

    full_meta = {
        "experiment": experiment,
        "label": label,
        "params": params,
        "revision": revision,
        "fingerprint": build_hash[:6],
        "fingerprint_parts": {build_path: build_hash},
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        **(meta or {}),
    }
    with open(run_dir / "meta.json", "w") as handle:
        json.dump(full_meta, handle, indent=2, sort_keys=True, default=str)

    if _is_worktree_dirty():
        (run_dir / "source.patch").write_text(_run_git_cmd("diff", "HEAD"))

    return RunDir(run_dir)


#
#
# Reading
#
#


def load_experiment_runs(experiment: str, latest_only: bool = True) -> pd.DataFrame:
    frames = []
    for curve_dir in sorted((RESULTS_ROOT / experiment).glob("*@*")):
        run_dirs = sorted(p for p in curve_dir.iterdir() if p.is_dir())
        if latest_only:
            run_dirs = run_dirs[-1:]

        for run_dir in run_dirs:
            data_path = run_dir / "data.csv"
            if not data_path.exists():
                continue

            frame = pd.read_csv(data_path)
            frame["curve"] = curve_dir.name
            frame["label"] = curve_dir.name.split("@")[0]
            frame["revision"] = _get_stamp_revision(run_dir.name)
            frame["run"] = run_dir.name
            frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No runs found under {RESULTS_ROOT / experiment}")

    return pd.concat(frames, ignore_index=True).sort_values(["curve", "run", "p"])
