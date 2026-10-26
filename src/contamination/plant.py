"""Plant-specific settings kept beside a process profile, for the dashboard only.

A profile's ``plant`` section holds what the rule book does not use: how to word things
for that plant's scientists, the column names expected in its exports, and defaults for
the value estimate. Each default records its source and whether the plant confirmed it,
so an unconfirmed number is never shown as the plant's own figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .diagnosis import PROFILES_DIR


@dataclass
class ValueDefault:
    value: float | None
    source: str
    confirmed: bool


@dataclass
class PlantProfile:
    name: str
    status: str = ""
    wording: dict[str, str] = field(default_factory=dict)
    observation_cues: list[str] = field(default_factory=list)
    export_names: dict[str, list[str]] = field(default_factory=dict)
    value_defaults: dict[str, ValueDefault] = field(default_factory=dict)

    def unconfirmed(self) -> list[str]:
        return [k for k, v in self.value_defaults.items() if not v.confirmed]


def load_plant(profile: str | Path) -> PlantProfile | None:
    """The ``plant`` section of a packaged profile name or path (None when it has none)."""
    path = Path(profile)
    if not path.exists():
        path = PROFILES_DIR / f"{profile}.yaml"
    if not path.exists():
        return None
    d = yaml.safe_load(path.read_text())
    if set(d) == {"use"}:
        return load_plant(d["use"])
    p = d.get("plant")
    if not p:
        return None
    return PlantProfile(
        name=d.get("name", path.stem),
        status=d.get("status", ""),
        wording=p.get("wording", {}),
        observation_cues=list(p.get("observation_cues", [])),
        export_names={k: [str(x) for x in v] for k, v in p.get("export_names", {}).items()},
        value_defaults={
            k: ValueDefault(v.get("value"), str(v.get("source", "")), bool(v.get("confirmed")))
            for k, v in p.get("value_defaults", {}).items()
        },
    )
