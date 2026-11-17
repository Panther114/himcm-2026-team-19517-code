"""Plaza grid, candidate device slots, shading by buildings and roof, and occupancy.

Plaza frame: origin at the plaza centroid, x along the long axis (16.3 deg north of east), y towards the north-north-west side.
"""
import os, sys, math, json
import numpy as np
from shapely.geometry import Point, Polygon, MultiPoint
from shapely import affinity, prepared
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import venue_cfd

CELL = 2.0


def build(cell=CELL):
    sc = venue_cfd.load_scene()
    P = sc['plaza']
    x0, y0, x1, y1 = P.bounds
    xs = np.arange(x0 + cell / 2, x1, cell); ys = np.arange(y0 + cell / 2, y1, cell)
    pp = prepared.prep(P.buffer(-0.5))
    pts = [(x, y) for y in ys for x in xs if pp.contains(Point(x, y))]
    xy = np.array(pts)
    sc['xy'] = xy
    sc['cell'] = cell
    sc['slots'] = slots(P)
    return sc


def slots(P, spacing=12.0, inset=3.0):
    """Candidate mounting slots: along the inside of the plaza boundary every ~12 m, 3 m in from the edge (arcade line)."""
    ring = P.buffer(-inset).exterior
    n = int(ring.length // spacing)
    return np.array([ring.interpolate(i * ring.length / n).coords[0] for i in range(n)])


def world_to_plaza_deg(world_math_deg, plaza_angle):
    return world_math_deg - plaza_angle


def sun_vector_plaza(azimuth_deg, plaza_angle):
    """Unit horizontal vector pointing towards the sun, in the plaza frame. Azimuth is clockwise from north."""
    math_deg = 90 - azimuth_deg
    a = math.radians(math_deg - plaza_angle)
    return np.array([math.cos(a), math.sin(a)])


def shadow_mask(sc, azimuth, zenith):
    """True where a plaza cell is shaded by buildings (roof open)."""
    if zenith >= 88:
        return np.ones(len(sc['xy']), bool)
    tanalt = math.tan(math.radians(90 - zenith))
    sv = sun_vector_plaza(azimuth, sc['angle_deg'])
    shadows = []
    for b in sc['buildings']:
        L = b['h'] / max(tanalt, 1e-3)
        P0 = b['poly']; dx, dy = -sv[0] * L, -sv[1] * L
        moved = affinity.translate(P0, dx, dy)
        cs = list(P0.exterior.coords)
        quads = [Polygon([cs[i], cs[i + 1], (cs[i + 1][0] + dx, cs[i + 1][1] + dy), (cs[i][0] + dx, cs[i][1] + dy)]).buffer(0)
                 for i in range(len(cs) - 1)]
        shadows.append(unary_union([P0, moved] + quads))   # exact sweep (a convex hull would fill concave recesses)
    U = prepared.prep(unary_union(shadows))
    return np.array([U.contains(Point(x, y)) for x, y in sc['xy']])


# ------------------------------------------------------------------ occupancy (ASSUMED; scaled in the sensitivity analysis)
# persons in the plaza by hour of day, summer weekday; weekend (Fri, Sat) x1.5. Evening peak after sunset (about 18:30),
# low use in the outdoor-work-ban hours (10:00-15:30), late-night use typical of Doha summer.
BASE = np.array([90, 50, 20, 8, 5, 10, 25, 30, 30, 30, 30, 35, 40, 40, 40, 45, 70, 110, 170, 240, 280, 280, 230, 160], float)
WEEKEND = 1.5


def occupancy_total(hour, dow, scale=1.0):
    return BASE[hour] * (WEEKEND if dow in (4, 5) else 1.0) * scale


def spatial_weights(sc, seat_frac=0.5, seat_width=8.0):
    """Half of the visitors within 8 m of the long edges (arcades, outlets and seating), half spread evenly."""
    P = sc['plaza']; xy = sc['xy']
    # long edges: the two boundary sides parallel to the plaza axis (|y| large in the plaza frame)
    ymin, ymax = P.bounds[1], P.bounds[3]
    near = (xy[:, 1] - ymin <= seat_width) | (ymax - xy[:, 1] <= seat_width)
    w = np.full(len(xy), (1 - seat_frac) / len(xy)) + np.where(near, seat_frac / near.sum(), 0.0)
    return w / w.sum()


if __name__ == '__main__':
    sc = build()
    print('cells', len(sc['xy']), 'area', len(sc['xy']) * CELL ** 2, 'slots', len(sc['slots']))
    print('daily persons-hours weekday', BASE.sum(), 'peak density', BASE.max() * WEEKEND / sc['plaza'].area)
