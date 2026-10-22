"""Writing an external dataset as a ready-to-open campaign folder."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from ..diagnosis import PROFILES_DIR
from ..pipeline import REFERENCE_LIST


def write_campaign(
    out_dir: Path,
    tables: dict[str, pd.DataFrame],
    profile: str,
    reference_ids: list[str],
    readme: str,
) -> None:
    """Measurements are gzip-compressed (the folders are committed so the dashboard can
    open them without the original downloads); the approved reference batches go in
    ``reference_batches.txt``; ``README.md`` carries the source and licence."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("measurements.csv", "reference_profile.json"):
        (out_dir / stale).unlink(missing_ok=True)
    for name, df in tables.items():
        if name == "measurements":  # mtime 0: the same data always gives the same file
            df.to_csv(
                out_dir / "measurements.csv.gz",
                index=False,
                compression={"method": "gzip", "mtime": 0},
            )
        else:
            df.to_csv(out_dir / f"{name}.csv", index=False)
    shutil.copy(PROFILES_DIR / f"{profile}.yaml", out_dir / "profile.yaml")
    (out_dir / REFERENCE_LIST).write_text("\n".join(reference_ids) + "\n")
    (out_dir / "README.md").write_text(readme)
