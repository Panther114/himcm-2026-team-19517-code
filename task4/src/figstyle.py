"""Shared Matplotlib style for Task 4.

Importing this module applies the style. Widths are inches; type sizes and
stroke widths are points. Map helpers require Cartesian coordinates in metres
and an equal-aspect axes. Output paths are relative to the project, not cwd.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import hashlib
import math
import os
import re
import tempfile
import warnings

import matplotlib as mpl
from matplotlib import font_manager, patheffects
from matplotlib.axis import Axis
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.text import Text
from matplotlib.legend import Legend
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

TEXT_WIDTH = FULL_WIDTH = 6.89
HALF_WIDTH = 3.35
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"
FONT_DIR = Path(os.environ.get("LATIN_MODERN_FONT_DIR", ""))
FONT_FILES = tuple(FONT_DIR / f"lmroman10-{face}.otf" for face in
                   ("regular", "bold", "italic", "bolditalic"))

FAN = "#2a78d6"
EVAPORATIVE = "#eb6834"
CHILLED_AIR = "#1baf7a"
NEUTRAL_GREY = "#8b8f98"
INK_PRIMARY = "#14161a"
INK_SECONDARY = "#4b4f57"
INK_MUTED = NEUTRAL_GREY
GRID = "#eceef1"
DEVICE_COLORS = {"A": FAN, "M": EVAPORATIVE, "E": EVAPORATIVE, "D": CHILLED_AIR}
DEVICE_MARKERS = {"A": "o", "M": "^", "E": "s", "D": "D"}
STATUS_COLORS = {"Safe": "#2e7d32", "Caution": "#eda100", "Danger": "#c62828"}
BLUE_RAMP = ("#eef5fe", "#0d366b")
# Section 4.2 gives no orange endpoints. Use white -> the specified family
# orange, introducing no additional categorical hue or invented endpoint.
ORANGE_RAMP = ("#ffffff", EVAPORATIVE)
DIVERGING_RAMP = ("#184f95", "#eef1f5", "#a82424")
BLUE_CMAP = LinearSegmentedColormap.from_list("task4_blue", BLUE_RAMP)
ORANGE_CMAP = LinearSegmentedColormap.from_list("task4_orange", ORANGE_RAMP)
DIVERGING_CMAP = LinearSegmentedColormap.from_list("task4_blue_grey_red", DIVERGING_RAMP)

TITLE_SIZE = 15
SUBTITLE_SIZE = 10
PANEL_SIZE = 11
AXIS_LABEL_SIZE = 9.5
TICK_SIZE = LEGEND_SIZE = DIRECT_LABEL_SIZE = 8.5
ANNOTATION_SIZE = 8
LINE_WIDTH = 1.8
MARKER_SIZE = 6
MARKER_EDGE_WIDTH = 0.8
ANNOTATION_WIDTH = 0.8


def _pdf_compatible_font(path: Path) -> Path:
    """Cache the same LM outlines as genuine TrueType for Type 42 export.

    Matplotlib's Type 42 path labels CFF-flavoured OTF data as FontFile2 /
    CIDFontType2, producing invalid PDF font metadata. Converting cubic CFF
    outlines to quadratic glyf outlines fixes the container, retaining LM's
    names, metrics, Unicode map and styles. The maximum outline error is
    0.5 font unit (0.0005 em); source fonts are never modified.
    """
    from fontTools.ttLib import TTFont, newTable
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.pens.cu2quPen import Cu2QuPen

    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    cache = Path(tempfile.gettempdir()) / 'task4_latin_modern_truetype_v1'
    cache.mkdir(exist_ok=True)
    target = cache / f'{path.stem}-{digest}.ttf'
    if target.is_file():
        return target
    with TTFont(path) as font:
        glyph_set = font.getGlyphSet()
        order = font.getGlyphOrder()
        glyf = newTable('glyf')
        glyf.glyphs = {}
        glyf.glyphOrder = order
        for name in order:
            pen = TTGlyphPen(glyph_set)
            glyph_set[name].draw(Cu2QuPen(pen, max_err=.5, reverse_direction=True))
            glyf.glyphs[name] = pen.glyph()
        font['glyf'] = glyf
        font['loca'] = newTable('loca')
        del font['CFF ']
        font.sfntVersion = '\x00\x01\x00\x00'
        font['maxp'] = newTable('maxp')
        font['maxp'].tableVersion = 0x10000
        for field in ('maxZones', 'maxTwilightPoints', 'maxStorage', 'maxFunctionDefs',
                      'maxInstructionDefs', 'maxStackElements', 'maxSizeOfInstructions'):
            setattr(font['maxp'], field, 0)
        font['maxp'].maxZones = 1
        font['post'].formatType = 2.0
        font['post'].extraNames = []
        font['post'].mapping = {}
        with tempfile.NamedTemporaryFile(dir=cache, suffix='.ttf', delete=False) as temp:
            temporary = Path(temp.name)
        try:
            font.save(temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


def setup_fonts() -> str:
    """Register all four LM faces; fall back to STIX if any is unavailable.

    Matplotlib reads the actual family name from the OTF metadata (usually
    'Latin Modern Roman'). Do not assume that the filename is a family name.
    """
    try:
        if not all(path.is_file() for path in FONT_FILES):
            raise FileNotFoundError("One or more Latin Modern Roman faces are missing")
        for path in map(_pdf_compatible_font, FONT_FILES):
            font_manager.fontManager.addfont(str(path))
        family = font_manager.FontProperties(fname=str(FONT_FILES[0])).get_name()
    except (OSError, RuntimeError) as exc:
        family = "STIXGeneral"
        warnings.warn(f"Latin Modern Roman unavailable; using STIX: {exc}",
                      RuntimeWarning, stacklevel=2)
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": list(dict.fromkeys([family, "STIXGeneral", "Palatino Linotype"])),
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.use14corefonts": False,
        "text.usetex": False,
    })
    return family


def apply_style() -> str:
    """Apply Sections 3 and 5; return the selected serif font family."""
    family = setup_fonts()
    mpl.rcParams.update({
        "font.size": TICK_SIZE,
        "text.color": INK_PRIMARY,
        "figure.facecolor": "white",
        "figure.figsize": (TEXT_WIDTH, 3.8),
        "figure.dpi": 100,
        "figure.titlesize": TITLE_SIZE,
        "figure.titleweight": "bold",
        "figure.labelsize": AXIS_LABEL_SIZE,
        "figure.autolayout": False,
        "figure.constrained_layout.use": False,
        "axes.facecolor": "white",
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK_PRIMARY,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.labelpad": 5,
        "axes.titlesize": PANEL_SIZE,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 9,
        "axes.linewidth": ANNOTATION_WIDTH,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        # Cycle over families, not classes: M and E use the same orange.
        "axes.prop_cycle": mpl.cycler(color=[FAN, EVAPORATIVE, CHILLED_AIR]),
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": MARKER_SIZE,
        "lines.markeredgecolor": "white",
        "lines.markeredgewidth": MARKER_EDGE_WIDTH,
        # White 2 pt edges create the required separation between bar fills.
        "patch.linewidth": 2,
        "patch.edgecolor": "white",
        "patch.force_edgecolor": True,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.top": False,
        "ytick.right": False,
        "xtick.major.width": ANNOTATION_WIDTH,
        "ytick.major.width": ANNOTATION_WIDTH,
        "legend.fontsize": LEGEND_SIZE,
        "legend.title_fontsize": LEGEND_SIZE,
        "legend.frameon": False,
        "legend.labelcolor": INK_PRIMARY,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "image.cmap": BLUE_CMAP.name,
    })
    return family


# Register named maps for rcParams and plt.get_cmap without overwriting a map.
for _cmap in (BLUE_CMAP, ORANGE_CMAP, DIVERGING_CMAP):
    if _cmap.name not in mpl.colormaps:
        mpl.colormaps.register(_cmap)
FONT_FAMILY = apply_style()


def diverging_norm(center: float = 0.0, vmin: float | None = None,
                   vmax: float | None = None) -> TwoSlopeNorm:
    """Map a meaningful center (e.g. AT_safe) to the neutral grey midpoint.

    Pair with DIVERGING_CMAP. Omitted bounds autoscale from the mappable data;
    explicit bounds must strictly bracket center.
    """
    if not math.isfinite(center):
        raise ValueError("center must be finite")
    for bound in (vmin, vmax):
        if bound is not None and not math.isfinite(bound):
            raise ValueError("bounds must be finite")
    return TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)


def new_figure(width: float | str = TEXT_WIDTH, height: float = 3.8):
    """Return a Figure, with room for a title/subtitle and axis labels.

    width accepts inches, 'full', or 'half'. Add axes after calling this helper;
    use title() after construction to reserve its header. Heights 2.6-4.2 in
    are standard for full-width figures.
    """
    if isinstance(width, str):
        try:
            width = {"full": TEXT_WIDTH, "half": HALF_WIDTH}[width]
        except KeyError as exc:
            raise ValueError("width must be inches, 'full', or 'half'") from exc
    if not (math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0):
        raise ValueError("figure dimensions must be positive and finite")
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(left=0.58 / width, right=1 - 0.12 / width,
                        bottom=0.52 / height, top=1 - 0.88 / height, wspace=0.4)
    return fig


def title(fig, title: str, subtitle: str):
    """Place a 15 pt bold title and 10 pt subtitle at a common top-left edge.

    Repeated calls replace the existing header. Header space is reserved via
    subplots_adjust, or the constrained-layout engine's rect when enabled.
    """
    width, height = fig.get_size_inches()
    for artist in getattr(fig, "_task4_header", ()):
        artist.remove()
    heading = fig.text(0.12 / width, 1 - 0.08 / height, title,
                       ha="left", va="top", fontsize=TITLE_SIZE,
                       fontweight="bold", color=INK_PRIMARY)
    subheading = fig.text(0.12 / width, 1 - 0.35 / height, subtitle,
                          ha="left", va="top", fontsize=SUBTITLE_SIZE,
                          color=INK_SECONDARY)
    fig._task4_header = (heading, subheading)
    top = 1 - 0.88 / height
    engine = fig.get_layout_engine()
    if isinstance(engine, mpl.layout_engine.ConstrainedLayoutEngine):
        engine.set(rect=(0, 0, 1, top))
    else:
        fig.subplots_adjust(top=top)
    return heading, subheading


def panel(ax, letter: str, text: str):
    """Set the left-aligned 11 pt bold panel label and title."""
    return ax.set_title(f"{letter}  {text}", loc="left", fontsize=PANEL_SIZE,
                        fontweight="bold", color=INK_PRIMARY, pad=9)


def colorbar(fig, ax, mappable, label: str, orientation: str = "vertical"):
    """Add a labelled colourbar. Pass units in parentheses, e.g. 'Cooling (°C)'.

    Dimensionless measures may use '(dimensionless)' or their unit symbol.
    ax can be a single axes or a list of axes sharing the quantity.
    """
    if orientation not in ("vertical", "horizontal"):
        raise ValueError("orientation must be 'vertical' or 'horizontal'")
    if not re.search(r"\([^()]+\)", label):
        raise ValueError("colourbar label must include units in parentheses")
    cb = fig.colorbar(mappable, ax=ax, orientation=orientation,
                      fraction=0.055 if orientation == "vertical" else 0.08,
                      pad=0.05 if orientation == "vertical" else 0.18,
                      aspect=22)
    if orientation == "horizontal":
        parents = [ax] if hasattr(ax, "xaxis") else list(ax)
        # Place the bar in display coordinates below the lowest rendered label.
        # A locator recomputes this on every draw, including resizing and export.
        original_locator = cb.ax.get_axes_locator()
        original_position = cb.ax.get_position(original=True).frozen()

        def below_labels(cax, renderer):
            position = (original_locator(cax, renderer) if original_locator
                        else original_position)
            bottoms = []
            for parent in parents:
                parent.xaxis._update_label_position(renderer)
                label_artist = parent.xaxis.label
                if label_artist.get_visible() and label_artist.get_text().strip():
                    bottoms.append(label_artist.get_window_extent(renderer).y0)
                else:
                    bottoms.append(parent.get_window_extent(renderer).y0)
            gap = renderer.points_to_pixels(6)
            top = fig.transFigure.inverted().transform((0, min(bottoms) - gap))[1]
            return Bbox.from_bounds(position.x0, top - position.height,
                                    position.width, position.height)

        cb.ax.set_axes_locator(below_labels)
    cb.set_label(label, fontsize=AXIS_LABEL_SIZE, color=INK_PRIMARY)
    cb.ax.tick_params(labelsize=TICK_SIZE, width=ANNOTATION_WIDTH)
    cb.outline.set_visible(False)
    cb.ax.grid(False)
    return cb


_LOCATIONS = {"lower left", "lower right", "upper left", "upper right"}


def scalebar(ax, length_m: float, location: str = "lower left"):
    """Add a true data-length scale bar; map x/y coordinates must be metres.

    Axes should use equal aspect. The white backing keeps the label legible
    over a field without decorative framing. Supports all four corner names.
    """
    if not math.isfinite(length_m) or length_m <= 0:
        raise ValueError("length_m must be positive and finite")
    if location not in _LOCATIONS:
        raise ValueError(f"location must be one of {sorted(_LOCATIONS)}")
    if ax.get_xscale() != "linear" or ax.get_yscale() != "linear":
        raise ValueError("map scale bars require linear coordinates in metres")
    bar = AnchoredSizeBar(ax.transData, length_m, f"{length_m:g} m", location,
                          pad=0.35, borderpad=0.6, sep=4,
                          frameon=True, color=INK_PRIMARY,
                          fontproperties=font_manager.FontProperties(size=ANNOTATION_SIZE))
    bar.patch.set_facecolor("white")
    bar.patch.set_edgecolor("none")
    # AnchoredSizeBar's default stroke can otherwise be backend dependent.
    for line in bar.size_bar.get_children():
        line.set_linewidth(LINE_WIDTH)
    ax.add_artist(bar)
    return bar


def north_arrow(ax, angle_deg: float = 0.0, location: str = "upper right"):
    """Add north in axes coordinates; angle is counterclockwise from page up.

    The rotation is measured in display space, independent of axes aspect.
    For projected data, supply the local north direction in this convention.
    """
    if location not in _LOCATIONS:
        raise ValueError(f"location must be one of {sorted(_LOCATIONS)}")
    if not math.isfinite(angle_deg):
        raise ValueError("angle_deg must be finite")
    theta = math.radians(angle_deg)
    direction = (-math.sin(theta), math.cos(theta))
    # Use physical points so rotation remains correct for rectangular maps.
    anchor = (0.18 if "left" in location else 0.82,
              0.20 if "lower" in location else 0.80)
    # Offsets in points, with text separated from the arrow tip.
    tail = ax.annotate("", xy=anchor, xycoords="axes fraction",
                       xytext=(-direction[0] * 18, -direction[1] * 18),
                       textcoords="offset points",
                       arrowprops={"arrowstyle": "-|>", "color": INK_SECONDARY,
                                   "lw": ANNOTATION_WIDTH, "mutation_scale": 11},
                       annotation_clip=False)
    label = ax.annotate("N", xy=anchor, xycoords="axes fraction",
                        xytext=(direction[0] * 8, direction[1] * 8),
                        textcoords="offset points", ha="center", va="center",
                        fontsize=ANNOTATION_SIZE, fontweight="bold", color=INK_PRIMARY,
                        path_effects=[patheffects.withStroke(linewidth=2, foreground="white")])
    return tail, label


def direct_label(ax, x, y, text: str, color=INK_PRIMARY):
    """Place an 8.5 pt direct label in data coordinates, with a white halo.

    Section 4.1 requires ink text, even when callers pass a series colour.
    Non-ink colors are therefore mapped to primary ink. Associate labels with
    their series through proximity, explicit names, and marker shape.
    """
    rgba = mpl.colors.to_rgba(color)
    allowed = (INK_PRIMARY, INK_SECONDARY, INK_MUTED)
    ink = next((c for c in allowed if mpl.colors.to_rgba(c) == rgba), INK_PRIMARY)
    return ax.text(x, y, text, fontsize=DIRECT_LABEL_SIZE, color=ink,
                   ha="left", va="center", clip_on=False, zorder=10,
                   path_effects=[patheffects.Stroke(linewidth=2, foreground="white", alpha=0.9),
                                 patheffects.Normal()])


def save(fig, name: str):
    """Write figures/<name>.pdf and .png; return the two absolute paths.

    PDF retains vector marks and embeds fonts. Raster artists supplied by a
    caller remain raster. PNG is 300 dpi; both use tight bounds and 0.04 in pad.
    name is a filename stem, not a path. Check overlaps before calling save.
    """
    if not name or name in {".", ".."} or any(c in name for c in '/\\:*?"<>|'):
        raise ValueError("name must be a simple filename stem")
    if Path(name).suffix.lower() in {".pdf", ".png"}:
        raise ValueError("pass a filename stem without .pdf or .png")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    png_path = FIGURES_DIR / f"{name}.png"
    with mpl.rc_context({"pdf.fonttype": 42, "pdf.use14corefonts": False}):
        for path in (pdf_path, png_path):
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04,
                        facecolor="white", transparent=False)
    return pdf_path, png_path


def check_text_overlap(fig) -> list[tuple[mpl.artist.Artist, mpl.artist.Artist]]:
    """Render and return intersecting text pairs and text/other-axes or legend pairs.

    Includes tick labels, legends, figure text, offset text, and map helpers.
    Hidden ancestors and undrawn ticks are excluded. Annotation arrows are
    excluded from text boxes. This checks rectangular text extents, not text
    versus plot marks or white halo strokes; visually inspect the PNG as well.
    Emits a RuntimeWarning for each text extending beyond the figure edge.
    """
    # PDF/SVG canvases have no get_renderer(); Agg gives consistent window
    # extents without permanently changing the caller's backend or canvas.
    original_canvas = fig.canvas
    canvas = FigureCanvasAgg(fig)
    try:
        canvas.draw()
        renderer = canvas.get_renderer()
    finally:
        fig.set_canvas(original_canvas)
    texts = []
    legends = []
    legend_members = {}
    seen = set()

    def collect(artist):
        if id(artist) in seen or not artist.get_visible():
            return
        seen.add(id(artist))
        if isinstance(artist, Legend):
            legends.append(artist)
            for member in artist.findobj(match=Text):
                legend_members[id(member)] = artist
        if isinstance(artist, Text):
            if artist.get_text().strip():
                texts.append(artist)
            return
        if isinstance(artist, Axis):
            if not artist.axes.axison:
                return
            collect(artist.label)
            collect(artist.offsetText)
            # Use the same tick selection as Axis.draw: get_children() alone
            # includes cached ticks outside the view limits that are not drawn.
            for tick in artist._update_ticks():
                if tick.get_visible():
                    collect(tick.label1)
                    collect(tick.label2)
            return
        for child in artist.get_children():
            collect(child)

    collect(fig)
    boxes = []
    edge = fig.bbox
    for text in texts:
        box = Text.get_window_extent(text, renderer=renderer)
        if not all(math.isfinite(v) for v in box.extents) or box.width <= 0 or box.height <= 0:
            continue
        boxes.append((text, box))
        if box.x0 < edge.x0 or box.y0 < edge.y0 or box.x1 > edge.x1 or box.y1 > edge.y1:
            warnings.warn(f"Text extends beyond figure edge: {text.get_text()!r}",
                          RuntimeWarning, stacklevel=2)
    def intersects(a, b):
        return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1

    overlaps = [(a, b) for (a, ba), (b, bb) in combinations(boxes, 2)
                if intersects(ba, bb)]
    for text, box in boxes:
        for other_ax in fig.axes:
            if other_ax.get_visible() and other_ax is not text.axes:
                if intersects(box, other_ax.get_window_extent(renderer)):
                    overlaps.append((text, other_ax))
        for legend in legends:
            if legend is not legend_members.get(id(text)):
                if intersects(box, legend.get_window_extent(renderer)):
                    overlaps.append((text, legend))
    return overlaps
