from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


SCHEMA_VERSION = 2
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled", "interrupted"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ProjectManifest:
    name: str
    id: str = field(default_factory=lambda: str(uuid4()))
    schema_version: int = SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    input_mode: Literal["font", "glyphs"] = "font"
    global_source_fonts: list[str] = field(default_factory=list)
    regional_source_fonts: dict[str, list[str]] = field(
        default_factory=lambda: {"SC": [], "TC": [], "JP": [], "KR": []}
    )
    target_assets: list[str] = field(default_factory=list)
    base_model: str = ""
    active_checkpoint: str = ""
    style_reference_pool: list[str] = field(default_factory=list)
    charset_presets: list[str] = field(
        default_factory=lambda: ["latin-extended", "cjk-punctuation", "zh-Hans-6500"]
    )
    split_regions: bool = True
    primary_region: str = "SC"
    training: dict[str, Any] = field(default_factory=dict)
    inference: dict[str, Any] = field(default_factory=dict)
    export: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProjectManifest":
        value = dict(value)
        legacy_global = value.pop("global_source_font", "")
        if "global_source_fonts" not in value:
            value["global_source_fonts"] = [legacy_global] if legacy_global else []
        regional = value.get("regional_source_fonts") or {}
        value["regional_source_fonts"] = {
            region: (
                list(fonts)
                if isinstance(fonts, list)
                else [fonts]
                if isinstance(fonts, str) and fonts
                else []
            )
            for region, fonts in {
                "SC": regional.get("SC", []),
                "TC": regional.get("TC", []),
                "JP": regional.get("JP", []),
                "KR": regional.get("KR", []),
            }.items()
        }
        value["schema_version"] = SCHEMA_VERSION
        fields = cls.__dataclass_fields__
        return cls(**{key: item for key, item in value.items() if key in fields})

    def source_fonts_for_region(self, region: str | None = None) -> list[str]:
        regional = self.regional_source_fonts.get(region, []) if region else []
        return list(dict.fromkeys([*regional, *self.global_source_fonts]))


@dataclass
class GlyphCandidate:
    seed: int
    png_path: str
    svg_path: str = ""
    qa: dict[str, Any] = field(default_factory=dict)


@dataclass
class GlyphRecord:
    codepoint: int
    region: str
    source_font: str
    reference_glyph: str
    candidates: list[GlyphCandidate] = field(default_factory=list)
    selected_candidate: int = 0
    status: str = "pending"
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JobRecord:
    job_type: str
    command: list[str]
    cwd: str
    id: str = field(default_factory=lambda: str(uuid4()))
    schema_version: int = SCHEMA_VERSION
    project_id: str | None = None
    status: JobStatus = "queued"
    pid: int | None = None
    gpu: str | None = None
    log_path: str = ""
    progress: float = 0.0
    progress_text: str = ""
    resume_point: str = ""
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingRun:
    project_id: str
    parameters: dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid4()))
    schema_version: int = SCHEMA_VERSION
    dataset_hash: str = ""
    device: dict[str, Any] = field(default_factory=dict)
    metrics_path: str = ""
    tensorboard_path: str = ""
    checkpoint_paths: dict[str, str] = field(default_factory=dict)
    status: JobStatus = "queued"
    parent_run_id: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
