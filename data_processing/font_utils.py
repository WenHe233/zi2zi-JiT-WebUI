from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont


SUPPORTED_FONT_SUFFIXES = {".ttf", ".otf", ".TTF", ".OTF"}

CJK_RANGES = [
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF),  # Extension B
    (0x2A700, 0x2B73F),  # Extension C
    (0x2B740, 0x2B81F),  # Extension D
    (0x2B820, 0x2CEAF),  # Extension E
    (0x2CEB0, 0x2EBEF),  # Extension F
    (0x30000, 0x3134F),  # Extension G
    (0xF900, 0xFAFF),    # Compatibility Ideographs
    (0x2F800, 0x2FA1F),  # Compatibility Supplement
]


def ensure_output_directory(output_path: str) -> Path:
    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_font_file(font_path: str) -> Path:
    path = Path(font_path)
    if not path.exists():
        raise FileNotFoundError(f"Font file not found: {font_path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {font_path}")
    if path.suffix not in SUPPORTED_FONT_SUFFIXES:
        raise ValueError(
            f"Unsupported font format: {path.suffix}. Supported: {', '.join(sorted(SUPPORTED_FONT_SUFFIXES))}"
        )
    return path


def load_font(font_path: str) -> Tuple[TTFont, Path]:
    validated_path = validate_font_file(font_path)
    font = TTFont(str(validated_path))
    return font, validated_path


def is_cjk_codepoint(codepoint: int) -> bool:
    for start, end in CJK_RANGES:
        if start <= codepoint <= end:
            return True
    return False


def has_valid_outline(font: TTFont, codepoint: int) -> bool:
    cmap = font.getBestCmap()
    if cmap is None or codepoint not in cmap:
        return False

    glyph_name = cmap[codepoint]

    if "glyf" in font:
        glyf = font["glyf"]
        if glyph_name not in glyf:
            return False
        glyph = glyf[glyph_name]
        if glyph.numberOfContours == 0:
            return False
        return True

    if "CFF " in font:
        try:
            from fontTools.pens.boundsPen import BoundsPen

            glyphset = font.getGlyphSet()
            if glyph_name not in glyphset:
                return False
            pen = BoundsPen(glyphset)
            glyphset[glyph_name].draw(pen)
            return pen.bounds is not None
        except Exception:
            return False

    return False


def get_cjk_codepoints(font: TTFont, filter_empty: bool = True) -> Set[int]:
    cmap = font.getBestCmap() or {}
    cjk_codepoints = {cp for cp in cmap.keys() if is_cjk_codepoint(cp)}
    if filter_empty:
        cjk_codepoints = {cp for cp in cjk_codepoints if has_valid_outline(font, cp)}
    return cjk_codepoints


def get_outline_codepoints(font: TTFont) -> Set[int]:
    """Return every encoded Unicode codepoint backed by a non-empty outline."""
    cmap = font.getBestCmap() or {}
    return {codepoint for codepoint in cmap if has_valid_outline(font, codepoint)}


def extract_font_name(font: TTFont, fallback_path: Path) -> str:
    name_table = font.get("name")
    if not name_table:
        return fallback_path.stem

    for name_id in (4, 1):
        for record in name_table.names:
            if record.nameID != name_id:
                continue
            try:
                value = record.toUnicode().strip()
            except Exception:
                continue
            if value:
                return value
    return fallback_path.stem


class GlyphRenderer:
    SAMPLE_CHARS = [
        "中", "国", "人", "大", "小",
        "一", "二", "三", "四", "五",
        "日", "月", "水", "火", "木",
        "口", "田", "目", "耳", "手",
        "上", "下", "左", "右", "前",
        "東", "西", "南", "北", "道",
        "的", "是", "不", "了", "在",
        "有", "和", "这", "为", "我",
        "龍", "龜", "鬱", "靈", "響",
        "永", "風", "雲", "雨", "雪",
    ]

    def __init__(
        self,
        font_path: str,
        resolution: int,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        text_color: Tuple[int, int, int] = (0, 0, 0),
        sample_size: int = 50,
    ):
        self.font_path = font_path
        self.resolution = resolution
        self.background_color = background_color
        self.text_color = text_color
        self.sample_size = sample_size

        self.font_size = int(resolution * 0.8)
        self.pil_font = ImageFont.truetype(str(font_path), self.font_size)

        self._tt_font = TTFont(str(font_path))
        self._cmap = self._tt_font.getBestCmap() or {}
        self._global_offset = self._calculate_global_offset()

    def _get_sample_codepoints(self) -> list[int]:
        available = []

        for char in self.SAMPLE_CHARS:
            cp = ord(char)
            if cp in self._cmap:
                available.append(cp)
            if len(available) >= self.sample_size:
                return available[:self.sample_size]

        if len(available) < self.sample_size:
            cjk_in_font = [cp for cp in self._cmap.keys() if is_cjk_codepoint(cp) and cp not in set(available)]
            available.extend(cjk_in_font[: max(0, self.sample_size - len(available))])

        return available[:self.sample_size]

    def _calculate_global_offset(self) -> Tuple[int, int]:
        temp_image = Image.new("RGB", (self.resolution, self.resolution), self.background_color)
        draw = ImageDraw.Draw(temp_image)
        samples = self._get_sample_codepoints()

        if not samples:
            return (self.resolution // 2, self.resolution // 2)

        x_offsets = []
        y_offsets = []
        for cp in samples:
            char = chr(cp)
            try:
                bbox = draw.textbbox((0, 0), char, font=self.pil_font)
            except Exception:
                continue
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            if text_w <= 0 or text_h <= 0:
                continue
            x_offsets.append((self.resolution - text_w) // 2 - bbox[0])
            y_offsets.append((self.resolution - text_h) // 2 - bbox[1])

        if not x_offsets:
            return (self.resolution // 2, self.resolution // 2)

        return (int(sum(x_offsets) / len(x_offsets)), int(sum(y_offsets) / len(y_offsets)))

    def render(self, codepoint: int) -> Optional[Image.Image]:
        try:
            image = Image.new("RGB", (self.resolution, self.resolution), self.background_color)
            draw = ImageDraw.Draw(image)
            draw.text(self._global_offset, chr(codepoint), font=self.pil_font, fill=self.text_color)
            return image
        except Exception:
            return None


class GlyphRendererPool:
    """Render from the first font in an ordered collection that owns a glyph."""

    def __init__(
        self,
        font_paths: Iterable[str | Path],
        resolution: int,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        text_color: Tuple[int, int, int] = (0, 0, 0),
    ):
        paths = [validate_font_file(str(path)).resolve() for path in font_paths]
        if not paths:
            raise ValueError("At least one source font is required")
        self.font_paths = list(dict.fromkeys(paths))
        self.renderers = [
            GlyphRenderer(
                str(path),
                resolution,
                background_color=background_color,
                text_color=text_color,
            )
            for path in self.font_paths
        ]
        self._candidates: dict[int, list[int]] = {}
        for index, renderer in enumerate(self.renderers):
            for codepoint in renderer._cmap:
                self._candidates.setdefault(codepoint, []).append(index)
        self._owners: dict[int, int] = {}
        self._cmap = self._candidates
        self.resolution = resolution

    def _owner_for(self, codepoint: int) -> int | None:
        if codepoint in self._owners:
            return self._owners[codepoint]
        candidates = self._candidates.get(codepoint, [])
        if not candidates:
            return None
        owner = next(
            (
                index
                for index in candidates
                if has_valid_outline(self.renderers[index]._tt_font, codepoint)
            ),
            candidates[0],
        )
        self._owners[codepoint] = owner
        return owner

    def render(self, codepoint: int) -> Optional[Image.Image]:
        owner = self._owner_for(codepoint)
        if owner is None:
            return None
        return self.renderers[owner].render(codepoint)

    def source_for(self, codepoint: int) -> Path | None:
        owner = self._owner_for(codepoint)
        return self.font_paths[owner] if owner is not None else None

    def cjk_codepoints(self, filter_empty: bool = True) -> Set[int]:
        values: set[int] = set()
        for renderer in self.renderers:
            values.update(get_cjk_codepoints(renderer._tt_font, filter_empty=filter_empty))
        return values
