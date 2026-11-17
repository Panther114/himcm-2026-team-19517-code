"""
of_viz.py -- render OpenFOAM (ASCII) cell fields to PNG without ParaView.

Reads a case's polyMesh and one time directory, then draws:
  * slice  : any axis-aligned plane, coloured by any scalar field or |U|
  * cloud  : all cells as a 3-D point cloud, coloured by a field
  * profile: a line sample, plotted as a curve

Usage examples
--------------
python of_viz.py slice  --case busstop_cfd --time 800 --field T --axis y --coord 1.5 --out fig_T_plan.png
python of_viz.py slice  --case busstop_cfd --time 800 --field magU --axis z --coord 0.25 --out fig_U_vert.png
python of_viz.py cloud  --case busstop_cfd --time 800 --field T --yaw 35 --pitch 22 --out fig_T_3d.png
python of_viz.py profile --case busstop_cfd --time 800 --field T --out fig_T_axis.png

The same commands work on any other run (system A/C, sensitivity cases):
just point --case at the other folder. That is what makes it reusable for
sensitivity analysis.
"""

import argparse
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_points(path):
    txt = _read_text(path)
    lines = txt.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip().isdigit():
        i += 1
    n = int(lines[i].strip())
    body = "\n".join(lines[i + 1:])
    vals = np.array([float(x) for x in NUM.findall(body)[: 3 * n]])
    return vals.reshape(n, 3)


def read_faces(path):
    txt = _read_text(path)
    lines = txt.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip().isdigit():
        i += 1
    n = int(lines[i].strip())
    out = []
    pat = re.compile(r"^\s*(\d+)\s*\((.*)\)")
    for line in lines[i + 1:]:
        m = pat.match(line)
        if not m:
            continue
        out.append(np.array([int(v) for v in m.group(2).split()], dtype=np.int32))
        if len(out) >= n:
            break
    return out


def read_labels(path):
    txt = _read_text(path)
    lines = txt.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip().isdigit():
        i += 1
    n = int(lines[i].strip())
    vals = np.array([int(x) for x in NUM.findall("\n".join(lines[i + 1:]))[:n]], dtype=np.int32)
    return vals


def read_field(case, time, name):
    """Return (ncomp, values[ncells, ncomp])."""
    path = os.path.join(case, str(time), name)
    if not os.path.exists(path):
        return None
    txt = _read_text(path)
    m = re.search(r"internalField\s+(uniform|nonuniform)", txt)
    if not m:
        return None
    if m.group(1) == "uniform":
        mm = re.search(r"internalField\s+uniform\s+([^;]+);", txt)
        val = np.array([float(x) for x in NUM.findall(mm.group(1))])
        return val.size, val.reshape(1, -1)
    mm = re.search(r"internalField\s+nonuniform\s+List<[^>]+>\s*\n?\s*(\d+)\s*\n?\s*\(", txt)
    if not mm:
        return None
    n = int(mm.group(1))
    start = mm.end()
    body = txt[start:]
    # everything up to the closing bracket of the list
    nums = np.array([float(x) for x in NUM.findall(body)])
    # infer components from the declared class
    if re.search(r"class\s+volVectorField", txt):
        nc = 3
    elif re.search(r"class\s+volTensorField", txt):
        nc = 9
    else:
        nc = 1
    vals = nums[: n * nc].reshape(n, nc)
    return nc, vals


def load_display_field(case, time, name, n_cells=None):
    """Map a display name to a scalar cell field."""
    if name in ("magU", "U"):
        nc, vals = read_field(case, time, "U")
        if vals is None:
            raise SystemExit("field U not found in time " + str(time))
        return np.linalg.norm(vals, axis=1)
    nc, vals = read_field(case, time, name)
    if vals is None:
        raise SystemExit("field " + name + " not found in time " + str(time))
    if vals.shape[0] == 1 and n_cells:
        vals = np.repeat(vals, n_cells, axis=0)
    if nc == 3:
        return np.linalg.norm(vals, axis=1)
    return vals[:, 0]

def cell_centres(case):
    mesh = os.path.join(case, "constant", "polyMesh")
    cache = os.path.join(os.path.dirname(os.path.abspath(case)), ".viz_cache_%s.npz" % os.path.basename(case.rstrip("/\\")))
    pts_path = os.path.join(mesh, "points")
    if os.path.exists(cache) and os.path.exists(pts_path) and \
            os.path.getmtime(cache) > os.path.getmtime(pts_path):
        z = np.load(cache)
        return z["centres"], z["pts"]

    pts = read_points(pts_path)
    faces = read_faces(os.path.join(mesh, "faces"))
    owner = read_labels(os.path.join(mesh, "owner"))
    nb = os.path.join(mesh, "neighbour")
    neighbour = read_labels(nb) if os.path.exists(nb) else np.array([], dtype=np.int32)

    n_cells = int(owner.max()) + 1
    acc = np.zeros((n_cells, 3))
    w = np.zeros(n_cells)

    for fi, f in enumerate(faces):
        v = pts[f]
        fc = v.mean(axis=0)
        # area via fan triangulation
        if len(f) >= 3:
            a = 0.0
            for k in range(1, len(f) - 1):
                a += 0.5 * np.linalg.norm(np.cross(v[k] - v[0], v[k + 1] - v[0]))
        else:
            a = 1e-9
        o = owner[fi]
        acc[o] += fc * a
        w[o] += a
        if fi < len(neighbour):
            ncell = neighbour[fi]
            acc[ncell] += fc * a
            w[ncell] += a

    w[w == 0] = 1.0
    centres = acc / w[:, None]
    try:
        np.savez(cache, centres=centres, pts=pts)
    except Exception:
        pass
    return centres, pts


# --------------------------------------------------------------------------
# colour maps
# --------------------------------------------------------------------------

def colormap(x):
    """x in [0,1] -> (r,g,b) 0..255, blue->cyan->yellow->red."""
    x = np.clip(x, 0.0, 1.0)
    r = np.zeros_like(x)
    g = np.zeros_like(x)
    b = np.zeros_like(x)

    m = x < 0.33
    t = x[m] / 0.33
    r[m] = 20 + 0 * t
    g[m] = 30 + 200 * t
    b[m] = 120 + 135 * t

    m = (x >= 0.33) & (x < 0.66)
    t = (x[m] - 0.33) / 0.33
    r[m] = 20 + 235 * t
    g[m] = 230 - 30 * t
    b[m] = 255 - 230 * t

    m = x >= 0.66
    t = (x[m] - 0.66) / 0.34
    r[m] = 255
    g[m] = 200 - 170 * t
    b[m] = 25 - 20 * t

    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def get_font(size):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        for root in (r"C:\Windows\Fonts", "/usr/share/fonts/truetype/dejavu"):
            p = os.path.join(root, name)
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    pass
    return ImageFont.load_default()


# --------------------------------------------------------------------------
# slice renderer
# --------------------------------------------------------------------------

AXES = {"x": 0, "y": 1, "z": 2}


def render_slice(case, time, field, axis, coord, out, width=1100, height=460,
                 xlim=None, ylim=None, vmin=None, vmax=None, label=None,
                 cbar_label=None, tol=None, dots=None):
    centres, pts = cell_centres(case)
    f = load_display_field(case, time, field, len(centres))

    a = AXES[axis]
    bbox = pts.min(axis=0), pts.max(axis=0)

    # the two in-plane axes
    if axis == "y":      # plan view: x horizontal, z vertical
        ax, ay = 0, 2
        xlab, ylab = "x (m)", "z (m)"
    elif axis == "z":    # vertical section: x horizontal, y vertical
        ax, ay = 0, 1
        xlab, ylab = "x (m)", "y (m)"
    else:                # y-z section
        ax, ay = 2, 1
        xlab, ylab = "z (m)", "y (m)"

    if tol is None:
        tol = 0.35 * (bbox[1][a] - bbox[0][a]) / 26.0 + 0.02
    sel = np.abs(centres[:, a] - coord) <= tol
    if sel.sum() < 50:
        tol *= 3
        sel = np.abs(centres[:, a] - coord) <= tol

    if xlim is None:
        xlim = (bbox[0][ax], bbox[1][ax])
    if ylim is None:
        ylim = (bbox[0][ay], bbox[1][ay])

    # accumulate field on the pixel grid: each cell paints its footprint
    grid = np.zeros((height, width))
    cnt = np.zeros((height, width))
    xs = centres[sel, ax]
    ys = centres[sel, ay]
    vs = f[sel]
    ix = ((xs - xlim[0]) / (xlim[1] - xlim[0]) * (width - 1)).astype(int)
    iy = ((ylim[1] - ys) / (ylim[1] - ylim[0]) * (height - 1)).astype(int)

    # cell footprint in pixels (use the finest local spacing)
    n_along_x = max(1, int(round((xlim[1] - xlim[0]) / 0.0625)))
    rx = max(1, int(np.ceil(0.5 * width / n_along_x)))
    n_along_y = max(1, int(round((ylim[1] - ylim[0]) / 0.0625)))
    ry = max(1, int(np.ceil(0.5 * height / n_along_y)))

    for cx, cy, cv in zip(ix, iy, vs):
        if cx < 0 or cx >= width or cy < 0 or cy >= height:
            continue
        x0, x1 = max(0, cx - rx), min(width, cx + rx + 1)
        y0, y1 = max(0, cy - ry), min(height, cy + ry + 1)
        grid[y0:y1, x0:x1] += cv
        cnt[y0:y1, x0:x1] += 1.0
    cnt[cnt == 0] = np.nan
    img = grid / cnt

    # fill holes: iterative neighbour average
    mask = np.isnan(img)
    for _ in range(6):
        if not mask.any():
            break
        pad = np.pad(img, 1, constant_values=np.nan)
        stack = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:],
                          pad[:-2, :-2], pad[:-2, 2:], pad[2:, :-2], pad[2:, 2:]])
        with np.errstate(invalid="ignore"):
            avg = np.nanmean(stack, axis=0)
        fill = mask & ~np.isnan(avg)
        img[fill] = avg[fill]
        mask = np.isnan(img)

    # smooth
    for _ in range(3):
        pad = np.pad(img, 1, mode="edge")
        img = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:] + 4 * img) / 8.0

    if vmin is None:
        vmin = np.nanpercentile(img, 2)
    if vmax is None:
        vmax = np.nanpercentile(img, 98)
    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-9
    norm = (img - vmin) / (vmax - vmin)

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    valid = ~np.isnan(img)
    rgb[valid] = colormap(norm[valid])
    rgb[~valid] = (16, 16, 22)

    plt_w, plt_h = width, height
    margin_l, margin_r, margin_t, margin_b = 62, 120, 40, 46
    W = plt_w + margin_l + margin_r
    H = plt_h + margin_t + margin_b
    canvas = Image.new("RGB", (W, H), (12, 12, 16))
    canvas.paste(Image.fromarray(rgb), (margin_l, margin_t))
    d = ImageDraw.Draw(canvas)
    f12 = get_font(14)
    f16 = get_font(16)

    # frame
    d.rectangle([margin_l, margin_t, margin_l + plt_w, margin_t + plt_h], outline=(90, 90, 100))

    # ticks
    def nice_ticks(lo, hi, n=5):
        span = hi - lo
        step = 10 ** np.floor(np.log10(span / n))
        for mult in (1, 2, 5, 10):
            if span / (step * mult) <= n:
                step *= mult
                break
        t = np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)
        return t

    for t in nice_ticks(xlim[0], xlim[1]):
        xp = margin_l + (t - xlim[0]) / (xlim[1] - xlim[0]) * plt_w
        d.line([xp, margin_t + plt_h, xp, margin_t + plt_h + 5], fill=(200, 200, 200))
        d.text((xp - 10, margin_t + plt_h + 8), f"{t:g}", fill=(200, 200, 200), font=f12)
    for t in nice_ticks(ylim[0], ylim[1]):
        yp = margin_t + (ylim[1] - t) / (ylim[1] - ylim[0]) * plt_h
        d.line([margin_l - 5, yp, margin_l, yp], fill=(200, 200, 200))
        d.text((6, yp - 7), f"{t:g}", fill=(200, 200, 200), font=f12)
    d.text((margin_l + plt_w / 2 - 30, H - 20), xlab, fill=(230, 230, 230), font=f12)
    # rotated y-axis label
    tmp = Image.new("RGB", (240, 26), (12, 12, 16))
    td = ImageDraw.Draw(tmp)
    td.text((0, 0), ylab, fill=(230, 230, 230), font=f12)
    bb = td.textbbox((0, 0), ylab, font=f12)
    lab_img = tmp.crop((bb[0], bb[1], bb[2], bb[3])).rotate(90, expand=True)
    canvas.paste(lab_img, (6, margin_t + plt_h // 2 - lab_img.size[1] // 2))

    # title
    ttl = label or f"{field}  ({axis} = {coord:g} m, t = {time})"
    d.text((margin_l, 8), ttl, fill=(245, 245, 245), font=f16)

    # annotations: waiting positions and device
    if dots:
        for (px, py, name) in dots:
            cx = margin_l + (px - xlim[0]) / (xlim[1] - xlim[0]) * plt_w
            cy = margin_t + (ylim[1] - py) / (ylim[1] - ylim[0]) * plt_h
            d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], outline=(255, 255, 255), width=2)
            d.text((cx + 7, cy - 16), name, fill=(255, 255, 255), font=f12)

    # colour bar
    cb_x0, cb_x1 = margin_l + plt_w + 25, margin_l + plt_w + 55
    for i in range(plt_h):
        v = 1 - i / (plt_h - 1)
        col = tuple(int(c) for c in colormap(np.array([v]))[0])
        d.line([cb_x0, margin_t + i, cb_x1, margin_t + i], fill=col)
    d.rectangle([cb_x0, margin_t, cb_x1, margin_t + plt_h], outline=(120, 120, 130))
    for t in nice_ticks(vmin, vmax, 5):
        yp = margin_t + (vmax - t) / (vmax - vmin) * plt_h
        d.line([cb_x1, yp, cb_x1 + 5, yp], fill=(200, 200, 200))
        d.text((cb_x1 + 8, yp - 7), f"{t:.3g}", fill=(200, 200, 200), font=f12)
    if cbar_label:
        d.text((cb_x0 + 10, margin_t - 26), cbar_label, fill=(230, 230, 230), font=f12)
    d.text((10, H - 20), "generated by of_viz.py", fill=(90, 90, 100), font=f12)

    canvas.save(out)
    print("wrote", out, f"[{sel.sum()} cells in slab, tol={tol:.3f} m]")


# --------------------------------------------------------------------------
# 3-D cloud
# --------------------------------------------------------------------------

def render_cloud(case, time, field, out, yaw=35.0, pitch=22.0, size=900, vmin=None, vmax=None):
    centres, pts = cell_centres(case)
    f = load_display_field(case, time, field, len(centres))

    c = centres - centres.mean(axis=0)
    # yaw about vertical (y), pitch about x
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    x = c[:, 0] * cy - c[:, 2] * sy
    z = c[:, 0] * sy + c[:, 2] * cy
    y = c[:, 1]
    y2 = y * cp - z * sp
    z2 = y * sp + z * cp

    span = max(np.ptp(x), np.ptp(y2))
    xp = (x - x.mean()) / span
    yp = (y2 - y2.mean()) / span
    depth = z2

    if vmin is None:
        vmin = np.percentile(f, 2)
    if vmax is None:
        vmax = np.percentile(f, 98)
    norm = np.clip((f - vmin) / (vmax - vmin + 1e-12), 0, 1)

    W = H = size
    canvas = Image.new("RGB", (W, H), (10, 10, 14))
    d = ImageDraw.Draw(canvas)
    order = np.argsort(depth)
    px = (xp * 0.72 + 0.5) * W
    py = (0.5 - yp * 0.72) * H
    cols = colormap(norm)
    for i in order:
        col = tuple(int(v) for v in cols[i])
        r = 2 if depth[i] > np.median(depth) else 3
        d.ellipse([px[i] - r, py[i] - r, px[i] + r, py[i] + r], fill=col)
    f16 = get_font(16)
    d.text((14, 10), f"{field} cloud  (yaw {yaw:g}, pitch {pitch:g}, t={time})", fill=(240, 240, 240), font=f16)
    canvas.save(out)
    print("wrote", out)


# --------------------------------------------------------------------------
# line profile
# --------------------------------------------------------------------------

def render_profile(case, time, field, out, y=1.5, z=0.25, vmin=None, vmax=None):
    centres, pts = cell_centres(case)
    f = load_display_field(case, time, field, len(centres))

    tol = 0.034
    sel = (np.abs(centres[:, 1] - y) < tol) & (np.abs(centres[:, 2] - z) < tol)
    if sel.sum() < 10:
        sel = (np.abs(centres[:, 1] - y) < 3 * tol) & (np.abs(centres[:, 2] - z) < 3 * tol)
    xs = centres[sel, 0]
    fs = f[sel]
    order = np.argsort(xs)
    xs, fs = xs[order], fs[order]

    # average into 5 cm bins so the curve is readable
    bins = np.arange(xs.min(), xs.max() + 0.05, 0.05)
    idx = np.digitize(xs, bins)
    bx, bf = [], []
    for b in range(1, len(bins)):
        m = idx == b
        if m.any():
            bx.append(xs[m].mean())
            bf.append(fs[m].mean())
    xs, fs = np.array(bx), np.array(bf)

    W, H = 1000, 460
    ml, mr, mt, mb = 80, 40, 40, 55
    canvas = Image.new("RGB", (W, H), (12, 12, 16))
    d = ImageDraw.Draw(canvas)
    f14, f16 = get_font(14), get_font(16)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = (min(fs), max(fs)) if vmin is None else (vmin, vmax)
    if y1 - y0 < 1e-9:
        y1 = y0 + 1e-9
    pad = 0.08 * (y1 - y0)
    y0 -= pad
    y1 += pad

    def X(v):
        return ml + (v - x0) / (x1 - x0) * (W - ml - mr)

    def Y(v):
        return mt + (y1 - v) / (y1 - y0) * (H - mt - mb)

    d.rectangle([ml, mt, W - mr, H - mb], outline=(80, 80, 90))
    d.line([X(x0), Y(0) if y0 < 0 < y1 else Y(y0), X(x1), Y(0) if y0 < 0 < y1 else Y(y0)],
           fill=(60, 60, 70))

    pts2 = [(X(a), Y(b)) for a, b in zip(xs, fs)]
    d.line(pts2, fill=(80, 200, 255), width=3)
    for a, b in zip(xs, fs):
        d.ellipse([X(a) - 2, Y(b) - 2, X(a) + 2, Y(b) + 2], fill=(255, 255, 255))

    for xv, name in ((1.5, "x1"), (2.5, "x2"), (3.5, "x3")):
        if x0 <= xv <= x1:
            d.line([X(xv), mt, X(xv), H - mb], fill=(120, 120, 60))
            d.text((X(xv) - 8, mt + 6), name, fill=(230, 230, 150), font=f14)

    for t in np.linspace(x0, x1, 6):
        d.line([X(t), H - mb, X(t), H - mb + 5], fill=(200, 200, 200))
        d.text((X(t) - 12, H - mb + 10), f"{t:.1f}", fill=(200, 200, 200), font=f14)
    for t in np.linspace(y0, y1, 5):
        d.line([ml - 5, Y(t), ml, Y(t)], fill=(200, 200, 200))
        d.text((10, Y(t) - 7), f"{t:.2f}", fill=(200, 200, 200), font=f14)

    d.text((W / 2 - 40, H - 26), "x (m)", fill=(230, 230, 230), font=f14)
    d.text((10, 10), f"{field} along the device axis (y={y:g} m, z={z:g} m)", fill=(240, 240, 240), font=f16)
    canvas.save(out)
    print("wrote", out, f"[{sel.sum()} samples]")


# --------------------------------------------------------------------------


def render_volume(case, time, field, out, yaw=35.0, pitch=20.0, size=900, vmin=None, vmax=None, k=0.6):
    """Additive volume rendering: every cell contributes colour, cold core shows through."""
    centres, pts = cell_centres(case)
    f = load_display_field(case, time, field, len(centres))

    c = centres - centres.mean(axis=0)
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    x = c[:, 0] * cy - c[:, 2] * sy
    z = c[:, 0] * sy + c[:, 2] * cy
    y = c[:, 1]
    y2 = y * cp - z * sp

    span = max(np.ptp(x), np.ptp(y2))
    xp = (x - x.mean()) / span
    yp = (y2 - y2.mean()) / span

    if vmin is None:
        vmin = np.percentile(f, 2)
    if vmax is None:
        vmax = np.percentile(f, 98)
    norm = np.clip((f - vmin) / (vmax - vmin + 1e-12), 0, 1)
    cols = colormap(norm).astype(float)

    W = H = size
    px = ((xp * 0.72 + 0.5) * (W - 1)).astype(int)
    py = ((0.5 - yp * 0.72) * (H - 1)).astype(int)
    ok = (px >= 2) & (px < W - 2) & (py >= 2) & (py < H - 2)

    acc = np.zeros((H, W, 3))
    wt = np.zeros((H, W))
    for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        xs = px[ok] + dx
        ys = py[ok] + dy
        np.add.at(acc, (ys, xs, np.zeros_like(xs)), cols[ok] * 0.2)
        np.add.at(wt, (ys, xs), 0.2)

    alpha = 1.0 - np.exp(-k * wt * 8.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        colour = acc / np.maximum(wt, 1e-9)[..., None]
    bg = np.array([10, 10, 14], dtype=float)
    out_rgb = colour * alpha[..., None] + bg * (1 - alpha[..., None])
    img = Image.fromarray(np.clip(out_rgb, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    f16 = get_font(16)
    d.text((14, 10), f"{field} volume  (yaw {yaw:g}, pitch {pitch:g}, t={time})", fill=(240, 240, 240), font=f16)
    img.save(out)
    print("wrote", out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["slice", "cloud", "volume", "profile"])
    ap.add_argument("--case", required=True)
    ap.add_argument("--time", default="800")
    ap.add_argument("--field", default="T")
    ap.add_argument("--axis", default="y")
    ap.add_argument("--coord", type=float, default=1.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--yaw", type=float, default=35)
    ap.add_argument("--pitch", type=float, default=22)
    ap.add_argument("--y", type=float, default=1.5)
    ap.add_argument("--z", type=float, default=0.25)
    a = ap.parse_args()

    if a.mode == "slice":
        dots = []
        if a.axis == "y":
            dots = [(1.5, 0.25, "x1"), (2.5, 0.25, "x2"), (3.5, 0.25, "x3")]
        elif a.axis == "z":
            dots = [(1.5, 1.5, "x1"), (2.5, 1.5, "x2"), (3.5, 1.5, "x3")]
        render_slice(a.case, a.time, a.field, a.axis, a.coord, a.out,
                     vmin=a.vmin, vmax=a.vmax, label=a.label, dots=dots)
    elif a.mode == "volume":
        render_volume(a.case, a.time, a.field, a.out, yaw=a.yaw, pitch=a.pitch, vmin=a.vmin, vmax=a.vmax)
    elif a.mode == "cloud":
        render_cloud(a.case, a.time, a.field, a.out, yaw=a.yaw, pitch=a.pitch,
                     vmin=a.vmin, vmax=a.vmax)
    else:
        render_profile(a.case, a.time, a.field, a.out, y=a.y, z=a.z,
                       vmin=a.vmin, vmax=a.vmax)


if __name__ == "__main__":
    main()
