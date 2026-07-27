from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


SCHEMA_VERSION = 1
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
    global_source_font: str = ""
    regional_source_fonts: dict[str, str] = field(
        default_factory=lambda: {"SC": "", "TC": "", "JP": "", "KR": ""}
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
        fields = cls.__dataclass_fields__
        return cls(**{key: item for key, item in value.items() if key in fields})


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
