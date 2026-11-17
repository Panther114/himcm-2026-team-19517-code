"""Eight figures built from the saved venue results (results/ and inputs/)."""
import glob
import json
import os
import math

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "venue_scale", "figures")
os.makedirs(OUT, exist_ok=True)
SC = json.load(open(os.path.join(ROOT, "inputs", "scenarios.json")))
VM = np.load(os.path.join(ROOT, "results", "venue_maps.npz"))


def kern(cid, cls, state, variant=""):
    pat = os.path.join(ROOT, "results", "kernels", f"{cid}_{cls}_{variant}_{state}_*.npz" if variant else f"{cid}_{cls}_{state}_*.npz")
    files = [f for f in glob.glob(pat) if "previous" not in f]
    return np.load(files[0])


def pair(cid, cls, variant=""):
    on = kern(cid, cls, "on", variant); off = kern(cid, cls, "off", variant)
    valid = on["valid"] & off["valid"]
    return dict(x=on["x"], y=on["y"], z=on["z"],
                dT=np.where(valid, off["T"] - on["T"], 0.0),
                Ton=on["T"] - 273.15, Toff=off["T"] - 273.15,
                wind=np.where(valid, np.linalg.norm(on["U"], axis=-1), 0.0),
                U=on["U"])


def grid_points(xy, area, values, h):
    x = np.arange(xy[:, 0].min(), xy[:, 0].max() + h, h)
    y = np.arange(xy[:, 1].min(), xy[:, 1].max() + h, h)
    ix = np.clip(np.round((xy[:, 0] - x[0]) / h).astype(int), 0, len(x) - 1)
    iy = np.clip(np.round((xy[:, 1] - y[0]) / h).astype(int), 0, len(y) - 1)
    g = np.zeros((len(x), len(y))); w = np.zeros((len(x), len(y)))
    np.add.at(g, (ix, iy), values * area); np.add.at(w, (ix, iy), area)
    g = np.divide(g, w, out=np.full_like(g, np.nan), where=w > 0)
    return x, y, g


def panel_map(ax, x, y, g, title, cmap, lo, hi, label):
    ext = [x[0], x[-1], y[0], y[-1]]
    pcm = ax.imshow(g.T, origin="lower", extent=ext, aspect="equal", cmap=cmap, vmin=lo, vmax=hi)
    ax.set_title(title, fontsize=8); ax.set_xlabel("x (m)", fontsize=7); ax.set_ylabel("y (m)", fontsize=7)
    cb = plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02); cb.set_label(label, fontsize=7); cb.ax.tick_params(labelsize=6)


def at_change(name, title, vmax):
    xy = VM[name + "_xy"]; area = VM[name + "_area"]; d = VM[name + "_dAT"]
    h = 1.0 if name == "bayfront" else 2.0
    x, y, g = grid_points(xy, area, d, h)
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw={"width_ratios": [2.1, 1, 1]})
    panel_map(axs[0], x, y, g, title, "Blues", 0, vmax, "cooling (K)")
    axs[0].text(0.02, 0.97, "max %.2f K" % d.max(), transform=axs[0].transAxes, va="top", fontsize=7,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.6"))
    axs[1].hist(d[d > 0.01], bins=60, color="tab:blue", alpha=0.85); axs[1].set_yscale("log")
    axs[1].set_xlabel("cooling (K)", fontsize=7); axs[1].set_ylabel("cells (log)", fontsize=7)
    axs[1].set_title("distribution", fontsize=8); axs[1].tick_params(labelsize=6); axs[1].grid(alpha=0.3)
    th = np.linspace(0, max(0.05, d.max()), 80)
    axs[2].plot(th, [(d > t).sum() * (area.mean()) / 1e4 for t in th], color="tab:red")
    axs[2].set_xlabel("threshold (K)", fontsize=7); axs[2].set_ylabel("area above (ha)", fontsize=7)
    axs[2].set_title("area vs threshold", fontsize=8); axs[2].tick_params(labelsize=6); axs[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "at_change_%s.png" % name), dpi=160); plt.close(fig)


def device_kernels():
    bf = pair("bf", "D"); md = pair("md", "M")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.0))
    jobs = [(bf, 1.1, "Bayfront chilled-air unit, 1.1 m"),
            (md, 1.1, "Medina misting fan, 1.1 m (body)"),
            (md, 3.5, "Medina misting fan, 3.5 m (delivery)")]
    for ax, (k, zz, title) in zip(axs, jobs):
        iz = int(np.argmin(abs(k["z"] - zz)))
        C = k["dT"][:, :, iz]
        ext = [k["x"][0], k["x"][-1], k["y"][0], k["y"][-1]]
        vmax = max(0.25, float(np.percentile(C, 99.8)))
        pcm = ax.imshow(C.T, origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        step = 14
        Xg, Yg = np.meshgrid(k["x"], k["y"], indexing="ij")
        ax.quiver(Xg[::step, ::step], Yg[::step, ::step], k["U"][::step, ::step, iz, 0],
                  k["U"][::step, ::step, iz, 1], scale=60, width=0.002, color="0.25")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("x (m)", fontsize=7); ax.set_ylabel("y (m)", fontsize=7); ax.tick_params(labelsize=6)
        plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02, label="K").ax.tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "device_kernels.png"), dpi=160); plt.close(fig)


def positions():
    bf = json.load(open(os.path.join(ROOT, "inputs", "bayfront_cooler_positions.json")))
    bfx = [p["x"] for p in bf]; bfy = [p["y"] for p in bf]
    md = json.load(open(os.path.join(ROOT, "inputs", "medina_umbrella_layout.json")))["umbrellas"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    axs[0].scatter(bfx, bfy, s=18); axs[0].set_title("Bayfront: %d candidate positions" % len(bfx), fontsize=8)
    axs[1].scatter([u["x"] for u in md], [u["y"] for u in md], s=6, c="tab:green")
    axs[1].set_title("Medina: %d umbrella columns" % len(md), fontsize=8)
    for ax in axs:
        ax.set_aspect("equal"); ax.set_xlabel("x (m)", fontsize=7); ax.set_ylabel("y (m)", fontsize=7); ax.tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "positions_compare.png"), dpi=160); plt.close(fig)


def venue_bf():
    xy = VM["bayfront_xy"]; area = VM["bayfront_area"]
    x, y, on = grid_points(xy, area, VM["bayfront_AT_on"], 1.0)
    _, _, d = grid_points(xy, area, VM["bayfront_dAT"], 1.0)
    _, _, w = grid_points(xy, area, VM["bayfront_wind10"], 1.0)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    panel_map(axs[0], x, y, on, "AT, devices on (degC)", "inferno_r", np.nanmin(on), np.nanmax(on), "degC")
    panel_map(axs[1], x, y, d, "AT reduction (K)", "Blues", 0, 10, "K")
    panel_map(axs[2], x, y, w, "wind at 10 m (m/s)", "viridis", 1.5, 7.0, "m/s")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "venue_bf.png"), dpi=160); plt.close(fig)


def venue_md_detail():
    xy = VM["medina_xy"]; area = VM["medina_area"]
    x, y, d = grid_points(xy, area, VM["medina_dAT"], 2.0)
    _, _, on = grid_points(xy, area, VM["medina_AT_on"], 2.0)
    _, _, w = grid_points(xy, area, VM["medina_wind10"], 2.0)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    panel_map(axs[0], x, y, on, "AT, devices on (degC)", "inferno_r", np.nanmin(on), np.nanmax(on), "degC")
    panel_map(axs[1], x, y, d, "AT reduction at 1.1 m (K)", "Blues", 0, 0.25, "K")
    panel_map(axs[2], x, y, w, "wind at 10 m (m/s)", "viridis", 6.2, 6.5, "m/s")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "venue_md_detail.png"), dpi=160); plt.close(fig)


def slice_at(k, z):
    zs = k["z"]
    zc = float(np.clip(z, zs[0], zs[-1]))
    i1 = int(np.searchsorted(zs, zc)); i0 = max(0, i1 - 1); i1 = min(len(zs) - 1, i1)
    a = 0.0 if i1 == i0 else (zc - zs[i0]) / (zs[i1] - zs[i0])
    return (1 - a) * k["dT"][:, :, i0] + a * k["dT"][:, :, i1]


def placed_layer(k, xy, positions, angle, z, weights=None, cap=None):
    interp = RegularGridInterpolator((k["x"], k["y"]), slice_at(k, z), bounds_error=False, fill_value=0.0)
    weights = np.ones(len(positions)) if weights is None else np.asarray(weights)
    out = np.zeros(len(xy)); th = math.radians(angle); cs, sn = math.cos(th), math.sin(th)
    for (px, py), w in zip(positions, weights):
        dx = xy[:, 0] - px; dy = xy[:, 1] - py
        local = np.column_stack([cs * dx + sn * dy, -sn * dx + cs * dy])
        sel = ((local[:, 0] >= k["x"][0]) & (local[:, 0] <= k["x"][-1]) &
               (local[:, 1] >= k["y"][0]) & (local[:, 1] <= k["y"][-1]))
        out[sel] += w * interp(local[sel])
    cool = out
    if cap is not None:
        cool = np.minimum(cool, cap)
    return cool


def layers(name, cid, cls, angle, positions, weights, xs, ys, cap, out_png, agg, levels, gap=1.0):
    k = pair(cid, cls)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    xy = np.column_stack([X.ravel(), Y.ravel()])
    zs = list(levels)
    cool = [placed_layer(k, xy, positions, angle, z, weights, cap).reshape(X.shape) for z in zs]
    if agg > 1:
        def bm(A, f):
            ny, nx = A.shape; ny2, nx2 = (ny // f) * f, (nx // f) * f
            return A[:ny2, :nx2].reshape(ny2 // f, f, nx2 // f, f).max(axis=(1, 3))
        cool = [bm(c, agg) for c in cool]
        xs2 = xs[:(len(xs) // agg) * agg].reshape(-1, agg).mean(axis=1)
        ys2 = ys[:(len(ys) // agg) * agg].reshape(-1, agg).mean(axis=1)
    else:
        xs2, ys2 = xs, ys
    X2, Y2 = np.meshgrid(xs2, ys2, indexing="ij")
    vmax = max(1.0, float(np.max([c.max() for c in cool])) * 0.9)
    norm = colors.Normalize(vmin=0.3, vmax=vmax); cmap = plt.get_cmap("Blues").copy(); cmap.set_under("0.92")
    zd = np.arange(len(zs)) * gap
    fig = plt.figure(figsize=(12, 8)); ax = fig.add_subplot(111, projection="3d")
    for z, C in zip(zd, cool):
        ax.plot_surface(X2, Y2, np.full_like(X2, z), facecolors=cmap(norm(C)), rstride=1, cstride=1, shade=False, antialiased=False)
        ax.plot_wireframe(X2, Y2, np.full_like(X2, z), rstride=max(1, len(xs2)//16), cstride=max(1, len(ys2)//16), color="0.75", lw=0.2, alpha=0.4)
    ax.set_zticks(zd); ax.set_zticklabels(["%.1f" % z for z in zs], fontsize=7)
    ax.set_zlim(-0.4, zd[-1] + 0.4)
    ax.set_box_aspect((xs2[-1] - xs2[0], ys2[-1] - ys2[0], 6.0 * (ys2[-1] - ys2[0]) * np.sin(np.deg2rad(18))))
    ax.set_xlabel("x (m)", fontsize=7); ax.set_ylabel("y (m)", fontsize=7); ax.set_zlabel("height (m)", fontsize=7)
    ax.view_init(elev=18, azim=-62); ax.set_title(name + ": cooling by height (K)", fontsize=9)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.5, label="cooling (K)", pad=0.08)
    fig.savefig(out_png, dpi=160); plt.close(fig)


if __name__ == "__main__":
    at_change("bayfront", "Bayfront: AT reduction at 1.1 m", 10)
    at_change("medina", "Medina: AT reduction at 1.1 m", 0.25)
    device_kernels()
    positions()
    venue_bf()
    venue_md_detail()

    bf_pos = np.array([[p["x"], p["y"]] for p in json.load(open(os.path.join(ROOT, "inputs", "bayfront_cooler_positions.json")))])
    layers("Bayfront (all candidate positions)", "bf", "D", SC["bayfront"]["flow_angle_deg_from_x"], bf_pos,
           np.ones(len(bf_pos)), np.arange(0, 131.4, 1.0), np.arange(0, 160.3, 1.0), 9.0,
           os.path.join(OUT, "layers_bayfront_T.png"), agg=2,
           levels=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5], gap=1.0)

    md = json.load(open(os.path.join(ROOT, "inputs", "medina_umbrella_layout.json")))["umbrellas"]
    md_pos = np.array([[u["x"], u["y"]] for u in md])
    w = 436.0 / 243.0
    layers("Medina detail", "md", "M", SC["medina"]["flow_angle_deg_from_x"], md_pos,
           np.full(len(md_pos), w), np.arange(60, 201, 1.0), np.arange(80, 201, 1.0), 22.3,
           os.path.join(OUT, "layers_medina_zoom_T.png"), agg=1,
           levels=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], gap=1.0)
    print("done")
