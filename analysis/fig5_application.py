#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fig. 5 -- Application example: audit-and-repair loop for the Na/Na3SbS4 model.

(A) Vacuum dimer energy curves for Na-Na, S-S and Na-S: DFT reference points
    versus the fine-tuned model before repair (v1), after the first repair (v2)
    and after the second repair (v2.1).  dE = E(d) - E(3.5 A) for every curve
    (each curve is shifted by its own value at the largest distance).  The
    spurious Na-Na minimum of v2 and the DFT minimum are marked.  Curves are
    clipped at 8 eV (say so in the caption; the y ticks stop at 6).  Legend
    entries are one word each, as in Fig. 3 (FIRE / Random / ...): define
    v1 = before repair, v2 = first repair, v2.1 = repaired in the caption.
    Source: data/dimerA_results.json (90 dimers = 6 pairs x 15 distances).
(B) Five-model audit summary: pair-scan cells passed (of 16).  Bars are keyed
    to model identity exactly as in (C): foundation model (tick "MPA-0" =
    MACE-MPA-0, define in the caption) = sky '\\' hatch, fine-tuned models =
    '//' hatch, filled navy when the audit passed and left white (an emptied
    navy bar) when it failed.  All three fills are keyed in the legend.  The
    hole-hunt outcome is the second line under every model name ("3/50" = 3 of
    50 random starts collapsed; the row is headed "hunt" left of the axes;
    define in the caption).
    Source: data/g2_report.json (criteria v1.1).
(C) Deep-compression wall at 0.9 A (Li-Li at 1.2 A, marked *) for seven DFT-
    anchored pairs of five systems: fine-tuned model / foundation model / DFT
    (headline, second, reference -- the Fig. 2C sequence), each relative to the
    minimum of its own curve, log scale.  Pairs are grouped by system: the pair
    groups of one system sit one unit apart and systems are separated by an
    extra 0.3 unit, with one system name per system under the pair names (as
    Fig. 2C separates its per-system groups); LPS/LBO = Li3PS4/Li3B11O18 and
    LLZO = Li7La3ZrxMxO12 (define in the caption).  Value labels carry one
    decimal throughout (state the rule in the caption); labels are centred on
    their bars and, where neighbouring bars are too close in height for the
    labels to clear each other, the label of the taller bar is stepped up just
    enough to clear its neighbour; a label that had to be lifted well above its
    own bar is tied to it by a thin grey leader (reported in the printout).
    Sources: data/kmod_curves.json (DFT, converged points only) and
             data/kdimer_model.json (models on the same grid).

House style (matches the original Figs 2-4 of the manuscript):
    Times New Roman everywhere (registered from --fonts if present, otherwise
    the system copy, otherwise Liberation Serif / STIXGeneral); tick labels,
    legends, annotations and second tick lines 9 pt, sub-panel titles 11 pt
    (the "n = 500" titles of Fig. 3), axis labels 10.5 pt, footnotes 8 pt,
    value labels 7.5 pt (bold for the headline series); italic variables
    (d, E) with upright Delta; chemical subscripts raised and enlarged to the
    Fig. 2C proportions; bold navy "(A)" panel labels outside the axes
    (baseline-aligned with the sub-panel titles in the top row); boxed axes
    with outward ticks left/bottom; frameless legends inside the axes.  Line
    series follow Fig. 3 (red filled circles, blue filled squares, green open
    inverted triangles, 1.0-pt lines, long legend handles), raw DFT data
    follow Fig. 4 (small grey dots drawn ON TOP of the model lines, so a grey
    centre inside a red marker means the model coincides with DFT), arrows
    follow Fig. 4C (slate-blue hairline, small filled head, stopping short of
    the marker stack), bars follow Fig. 2C (steel-blue '//' headline, sky '\\'
    second, pale '..' reference; fine black pinstripe / light stipple, NO
    outline; value labels above every bar; thin 3.5:1 legend swatches).

Usage:  python fig5_application.py [--data DIR] [--out DIR] [--fonts DIR]
Writes  <out>/fig5_application.pdf and <out>/fig5_application.png (300 dpi)
and prints every number that is drawn (for the caption / SI tables).
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MultipleLocator, LogLocator, NullFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def first_existing(cands, fallback):
    for c in cands:
        if os.path.isdir(c):
            return c
    return fallback


ap = argparse.ArgumentParser()
ap.add_argument("--data", default=first_existing(
    [os.path.join(HERE, "data"), os.path.join(HERE, "..", "data")],
    os.path.join(HERE, "data")))
ap.add_argument("--out", default=first_existing(
    [os.path.join(HERE, "..", "figures"), os.path.join(HERE, "figures")],
    os.path.join(HERE, "..", "figures")))
ap.add_argument("--fonts", default=first_existing(
    [os.path.join(HERE, "fonts"), os.path.join(HERE, "..", "fonts")], ""))
args = ap.parse_args()
DATA, OUT = args.data, args.out
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------- house style
if args.fonts and os.path.isdir(args.fonts):       # real Times faces (times*.ttf)
    for f in sorted(glob.glob(os.path.join(args.fonts, "*.[ot]tf"))):
        fm.fontManager.addfont(f)
_avail = {f.name for f in fm.fontManager.ttflist}
SERIF = next((f for f in ("Times New Roman", "Liberation Serif", "STIXGeneral")
              if f in _avail), "DejaVu Serif")

# Chemical-formula subscripts of Fig. 2C are 0.79 of the cap height and drop only
# 0.21 cap below the baseline; mathtext's fixed 0.7 shrink and generic font
# constants give 0.65 / 0.35.  Private API (names exist in matplotlib 3.6-3.10);
# skipped silently elsewhere.
SUBSCRIPT_TWEAK = False
try:
    import matplotlib._mathtext as _mt

    _mt.SHRINK_FACTOR = 0.84        # GaF3 '3': 20.5 px on a 26-px cap (0.79, Fig. 2C)

    class _SerifConstants(_mt.FontConstantsBase):
        # a subscript after plain text has an empty nucleus, so its drop is sub1 * xHeight
        # (mathtext's pclt xHeight is ~32 px here): 0.18 -> '3' bottom ~6 px (0.21 cap, Fig. 2C)
        sub1 = 0.18
        sub2 = 0.3
        subdrop = 0.32

    _mt._font_constant_mapping[SERIF] = _SerifConstants
    SUBSCRIPT_TWEAK = True
except Exception:                                   # pragma: no cover
    pass

NAVY = "#003366"        # panel labels (sampled from Figs 2 and 3)
BAR_NAVY = "#3d699e"    # headline bar fill of Fig. 2C (steel blue under black hatch)
SKY = "#97d2f0"         # second bar series of Fig. 2C
PALE = "#cee2ee"        # reference bar series of Fig. 2C
RED, BLUE, GREEN = "#ee0000", "#31639c", "#38c672"   # FIRE / Random / Vanilla of Fig. 3
GREY = "#696969"        # raw data dots of Fig. 4
GRID = "#d9d9d9"        # faint dotted grid of Fig. 2C (bar panels only)
ARROW = "#7d90aa"       # annotation arrows of Fig. 4C
LEADER = "#a0a0a0"      # value-label leaders (C)

TICK_PT = 9.0           # tick labels = legends = annotations = second tick lines
TITLE_PT = 11.0         # sub-panel titles ("n = 500" of Fig. 3: 1.25 x tick size)
TITLE_PAD = 7.0         # points between the top spine and the title baseline
VAL_PT = 8.0            # value labels (B and C): 0.9 x tick digits, as Fig. 2C
FOOT_PT = 9.0           # footnotes share the tick/legend size (no sixth size in the figure)
ANN_PT = TICK_PT        # annotations
LINE2_PT = TICK_PT      # second tick line (system names under the pair names)

# Fig. 2C hatch, calibrated with one metric on both images at matched resolution
# (tick digits 14 px: ref_fig2.png as is, this figure downsampled to 150 dpi).
# Fig. 2C bars: '/' pitch 0.50 digit, stroke 0.18, ink 0.35; '\' pitch 0.83,
# stroke 0.24, ink 0.29; '.' 8-10 dots per digit^2 of 0.22-0.23 digit diameter.
# Matplotlib at 300 dpi: '/' pitch = 100/n px, '.' rows = 50/n px, 0.8-pt stroke:
# n = 7 / 4 / 5 give 0.51 / 0.89 / (8.3 dots per digit^2, 0.24) -- see probe log.
HATCH_HEAD = "/" * 7      # headline series (fine-tuned)          pitch 0.51 digit
HATCH_SECOND = "\\" * 4   # second series (foundation model)      pitch 0.89 digit
HATCH_REF = "." * 5       # reference series (DFT)                8.3 dots / digit^2

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [SERIF, "Times New Roman", "Liberation Serif", "STIXGeneral"],
    "mathtext.fontset": "custom",
    "mathtext.rm": SERIF,
    "mathtext.it": SERIF + ":italic",
    "mathtext.bf": SERIF + ":bold",
    "mathtext.cal": SERIF,
    "mathtext.sf": SERIF,
    "mathtext.tt": SERIF,
    "mathtext.default": "it",      # $d$, $E$ italic; digits and Delta stay upright
    "font.size": TICK_PT,
    "axes.labelsize": 10.5,
    "axes.titlesize": TITLE_PT,
    "xtick.labelsize": TICK_PT,
    "ytick.labelsize": TICK_PT,
    "legend.fontsize": TICK_PT,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "black",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.top": False,
    "ytick.right": False,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "lines.linewidth": 1.0,
    "legend.frameon": False,
    "hatch.linewidth": 0.8,         # 3.3 px at 300 dpi: Fig. 2C's stroke (0.17-0.25 digit after downsampling)
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

MODEL_LW = 1.2          # model curves of (A) (Fig. 3 series lines: ~0.18 tick digit)
BAR_LW = 0.0            # Fig. 2C bars carry no outline: pure fill + black hatch


def bars(ax, x, h, w, face, hatch, zorder=3):
    # edgecolor sets the hatch colour; linewidth 0 suppresses the rim only
    return ax.bar(x, h, w, facecolor=face, edgecolor="black", linewidth=BAR_LW,
                  hatch=hatch, zorder=zorder)


def swatch(face, hatch):
    return Patch(facecolor=face, edgecolor="black", linewidth=BAR_LW, hatch=hatch)


SWATCH_KW = dict(handlelength=2.8, handleheight=0.8, handletextpad=0.4,
                 labelspacing=0.45, borderaxespad=0.5, borderpad=0.1)   # Fig. 2C legend (3.5:1 swatches)


def value_label(ax, x, y, text, bold=False, dy=2.5):
    """horizontal value label above a bar (Fig. 2C); dy in points"""
    return ax.annotate(text, xy=(x, y), xytext=(0, dy), textcoords="offset points",
                       ha="center", va="bottom", fontsize=VAL_PT,
                       fontweight="bold" if bold else "normal", zorder=6)


def step_up_labels(fig, ax, texts, gap_pts=1.2, push_frac=0.6):
    """Value labels are centred on their bars.  Where neighbouring bars are so
    close in height that their labels would touch, the label of the taller bar
    (processed left to right) is stepped up just far enough to clear every
    label it overlaps.  A label that is wider than its bar and would print over
    the body of a taller neighbouring bar is pushed sideways, away from that
    bar, by at most push_frac bar widths (lifted above it only if that is not
    enough); a label walled by taller bars on both sides is lifted above the
    taller wall.  Returns [(text, push_pts, lift_pts)] for the printout."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    g = gap_pts * px
    bars_px = [p.get_window_extent(r) for c in ax.containers for p in c
               if p.get_height() > 0]
    wbar = min(b.width for b in bars_px)
    boxes = [t.get_window_extent(r) for t in texts]
    moves = [[0.0, 0.0] for _ in texts]                # (push, lift) in points
    order = sorted(range(len(texts)), key=lambda i: boxes[i].x0)

    def hit(a, b):
        return b.x0 < a.x1 + g and a.x0 < b.x1 + g and b.y0 < a.y1 + g and a.y0 < b.y1 + g

    placed = []
    for i in order:
        for _ in range(20):
            b = boxes[i]
            hits = [boxes[j] for j in placed if hit(b, boxes[j])]
            if hits:                                    # step up above the labels it touches
                up = max(h.y1 + g - b.y0 for h in hits)
                boxes[i] = b.translated(0, up)
                moves[i][1] += up / px
                continue
            walls = [w for w in bars_px if hit(b, w)]
            if not walls:
                break
            cx = 0.5 * (b.x0 + b.x1)
            left = [w for w in walls if 0.5 * (w.x0 + w.x1) < cx]
            right = [w for w in walls if 0.5 * (w.x0 + w.x1) >= cx]
            if left and right:                          # walled on both sides: lift above the taller wall
                up = max(w.y1 for w in walls) + g - b.y0
                boxes[i] = b.translated(0, up)
                moves[i][1] += up / px
                continue
            w = (left or right)[0]
            dx = (w.x1 + g - b.x0) if left else (w.x0 - g - b.x1)   # push away from the wall
            if abs(moves[i][0] * px + dx) <= push_frac * wbar:
                boxes[i] = b.translated(dx, 0)
                moves[i][0] += dx / px
            else:                                       # cannot push far enough: lift above the bar
                up = w.y1 + g - b.y0
                boxes[i] = b.translated(0, up)
                moves[i][1] += up / px
        placed.append(i)
    for t, (push, lift) in zip(texts, moves):
        if push or lift:
            dx, dy = t.xyann
            t.xyann = (dx + push, dy + lift)
    return [(t.get_text(), round(m[0], 1), round(m[1], 1)) for t, m in zip(texts, moves)]


def draw_leaders(fig, ax, texts, patches, moves, min_lift_pt=8.0, top_pad_pt=1.5, bot_pad_pt=1.0):
    """A value label that had to be lifted more than about one label height above
    its own bar (Fig. 2C never separates the two by more) is tied to the bar by a
    thin grey leader from the bar top to just below the label.  Returns
    [(text, leader_length_pts)]."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    inv = ax.transData.inverted()
    out = []
    for t, p, (txt, push, lift) in zip(texts, patches, moves):
        if lift < min_lift_pt:
            continue
        pb, tb = p.get_window_extent(r), t.get_window_extent(r)
        x = 0.5 * (pb.x0 + pb.x1)
        y0, y1 = pb.y1 + top_pad_pt * px, tb.y0 - bot_pad_pt * px
        (xd0, yd0), (xd1, yd1) = inv.transform([(x, y0), (x, y1)])
        ax.plot([xd0, xd1], [yd0, yd1], lw=0.4, color=LEADER, zorder=5, solid_capstyle="butt")
        out.append((txt, round((y1 - y0) / px, 1)))
    return out


def legend_marker_clearance_cm(fig, ax, legend):
    """smallest distance (cm) between the legend box and the edge of any visible
    plotted marker of the axes (negative = overlap); (distance, series label)"""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    lb, ab = legend.get_window_extent(r), ax.get_window_extent(r)
    px = fig.dpi / 72.0
    best = (np.inf, None)
    for ln in ax.lines:
        if ln.get_marker() in (None, "None", ""):
            continue
        rad = (0.5 * ln.get_markersize() + ln.get_markeredgewidth()) * px
        for x, y in ax.transData.transform(np.column_stack([ln.get_xdata(), ln.get_ydata()])):
            if not (ab.x0 <= x <= ab.x1 and ab.y0 <= y <= ab.y1):
                continue                                # clipped marker: not drawn
            dx = max(lb.x0 - x, x - lb.x1, 0.0)
            dy = max(lb.y0 - y, y - lb.y1, 0.0)
            dist = np.hypot(dx, dy) - rad
            if dist < best[0]:
                best = (dist, ln.get_label())
    return best[0] / fig.dpi * 2.54, best[1]


def below_axes(ax, xfrac, text, dy, **kw):
    """text hung below the axes at an axes-fraction x (footnotes)"""
    ax.annotate(text, xy=(xfrac, 0), xycoords="axes fraction", xytext=(0, dy),
                textcoords="offset points", va="top", annotation_clip=False, **kw)


def second_line(ax, x, text, dy=-17.5, **kw):
    """extra line under an x tick label, x in data units"""
    ax.annotate(text, xy=(x, 0), xycoords=("data", "axes fraction"), xytext=(0, dy),
                textcoords="offset points", ha="center", va="top", annotation_clip=False, **kw)


MM = 1.0 / 25.4
PT_CM = 2.54 / 72.0
EN = "–"   # en dash
ANG = "Å"  # Angstrom sign
W_CM, H_CM = 16.0, 11.0


def rect(x0, y0, w, h):
    """axes rectangle given in cm from the lower-left corner of the figure"""
    return [x0 / W_CM, y0 / H_CM, w / W_CM, h / H_CM]


def J(name):
    with open(os.path.join(DATA, name)) as fh:
        return json.load(fh)


dimers = J("dimerA_results.json")
g2 = J("g2_report.json")
kmod = J("kmod_curves.json")
kdim = J("kdimer_model.json")


def pretty(pair):
    return pair.replace("-", EN)


# ----------------------------------------------------------------------- figure
fig = plt.figure(figsize=(W_CM * 10 * MM, H_CM * 10 * MM), dpi=300)

# =============================================================================== (A)
PAIRS_A = ["Na-Na", "S-S", "Na-S"]
# key, legend label (one word, as Fig. 3), colour, marker, marker face, size, z-order
MODELS_A = [("v1", "FIRE", GREEN, "v", "white", 4.8, 3.0),
            ("v2", "FIRE+SR", BLUE, "s", BLUE, 3.8, 3.2),
            ("v21", "FIRE+SR′", RED, "o", RED, 4.2, 5.0)]
DFT_MS = 3.8            # grey reference dots drawn UNDER the model curves (Fig. 4: grey data, model line on top)
DFT_Z = 1.5
YLIM_A = {"Na-Na": (-5.0, 8.0), "S-S": (-6.5, 8.0), "Na-S": (-3.5, 8.0)}   # per-pair range (legend/annotations fit)
YTICKS_A = {"Na-Na": np.arange(-4, 6.1, 2), "S-S": np.arange(-6, 6.1, 2),
            "Na-S": np.arange(-2, 6.1, 2)}          # ticks stop at 6: curves are clipped at 8
A_X0, A_W, A_GAP, A_Y0, A_H = 1.40, 4.10, 1.10, 6.75, 3.50
LEGEND_PANEL_A = 2      # Na-S: its upper-right quadrant (d > 2.2 A, dE > 1 eV) is empty at the common scale
LEGEND_CLEAR_CM = 0.25  # no plotted marker may come closer than this to the legend box

curves = {}
axes_a = []
for i, p in enumerate(PAIRS_A):
    rows = sorted([r for r in dimers if r["pair"] == p], key=lambda r: r["d"])
    assert len(rows) == 15, (p, len(rows))
    d = np.array([r["d"] for r in rows], float)
    E = {"dft": np.array([r["E_dft"] for r in rows], float)}
    for m in MODELS_A:
        E[m[0]] = np.array([r["E_" + m[0]] for r in rows], float)
    dE = {k: v - v[-1] for k, v in E.items()}          # reference: E(3.5 A)
    curves[p] = (d, dE, E)

    ax = fig.add_axes(rect(A_X0 + i * (A_W + A_GAP), A_Y0, A_W, A_H))
    axes_a.append(ax)
    for m, lab, col, mk, mfc, ms, z in MODELS_A:
        ax.plot(d, dE[m], "-", marker=mk, ms=ms, color=col, mfc=mfc, mec=col, mew=0.9,
                lw=MODEL_LW, label=lab, zorder=z)
    ax.plot(d, dE["dft"], "o", ms=DFT_MS, color=GREY, mfc=GREY, mec=GREY, mew=0.0,
            label="DFT", zorder=DFT_Z)                  # grey reference dots beneath the models (Fig. 4)
    ax.set_xlim(0.85, 3.65)
    ax.set_ylim(*YLIM_A[p])
    ax.set_yticks(YTICKS_A[p])
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.set_xticks([1.0, 2.0, 3.0])
    ax.set_xticklabels(["1", "2", "3"])                 # integer ticks, as in Figs 3 and 4
    ax.set_title(pretty(p), pad=TITLE_PAD)
    ax.set_xlabel("$d$ (%s)" % ANG, labelpad=2)
    if i == 0:
        ax.set_ylabel(r"$\Delta E$ (eV)", labelpad=3)

    if p == "Na-Na":
        i_v2 = int(np.argmin(E["v2"]))
        i_dft = int(np.argmin(E["dft"]))
        # Fig. 4C arrows: hairline shaft, small filled head, 0.5-1 cm long, text one em inside the box.
        # patchA=None: without it Annotation clips the shaft against the text box padded by 2 pt,
        # which at a shallow angle eats ~6 pt of the tail on top of shrinkA.
        arrow = dict(arrowstyle="-|>", color=ARROW, lw=0.8, mutation_scale=7,
                     shrinkA=2, shrinkB=1.5, patchA=None)
        # tail 0.2 of the way along the top edge of the text block, tip on the tail-to-marker
        # line ~0.1 cm short of the v2 square so that the head points at the marker
        ax.annotate("FIRE+SR spurious\nminimum, %.2f %s" % (d[i_v2], ANG),
                    xy=(d[i_v2] + 0.02, dE["v2"][i_v2] - 0.20), xytext=(3.50, -2.9),
                    fontsize=ANN_PT, ha="right", va="center", color="black",
                    linespacing=1.2, arrowprops=dict(relpos=(0.2, 1.0), **arrow))
        ax.annotate("DFT minimum\n%.2f %s" % (d[i_dft], ANG),
                    xy=(d[i_dft], dE["dft"][i_dft] + 0.60), xytext=(2.55, 5.0),
                    fontsize=ANN_PT, ha="center", va="center", color="black",
                    linespacing=1.2, arrowprops=dict(relpos=(0.85, 0.0), **arrow))

handles = [Line2D([], [], marker="o", ls="none", ms=DFT_MS, color=GREY, mfc=GREY, mec=GREY, mew=0.0, label="DFT")]
for m, lab, col, mk, mfc, ms, z in MODELS_A:
    handles.append(Line2D([], [], ls="-", marker=mk, ms=ms, color=col, mfc=mfc, mec=col,
                          mew=0.9, lw=MODEL_LW, label=lab))
leg_a = axes_a[LEGEND_PANEL_A].legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0),
                                      handlelength=1.8, handletextpad=0.4, labelspacing=0.28,
                                      borderaxespad=0.45, borderpad=0.15)   # Fig. 3 handles, compact pitch

# =============================================================================== (B)
MODELS_B = [("mpa0", "MPA-0", False), ("v1", "FIRE", True), ("v2", "+SR", True),
            ("v21", "+SR′", True), ("v22", "+3B", True)]      # key, tick label (short form), fine-tuned?
B_X0, B_W, B_Y0, B_H = 1.40, 4.60, 1.10, 4.00
ax_b = fig.add_axes(rect(B_X0, B_Y0, B_W, B_H))
xb = np.arange(len(MODELS_B))
npass, nscan, hunt_bad, hunt_n, ok_all = [], [], [], [], []
for m, lab, ft in MODELS_B:
    rep = g2[m]
    n_ok = sum(1 for v in rep["scans"].values() if v["pass"])
    n_tot = len(rep["scans"])
    assert rep["summary"]["scans_pass"] == "%d/%d" % (n_ok, n_tot)
    hh = rep["hole_hunt"]
    npass.append(n_ok)
    nscan.append(n_tot)
    hunt_bad.append(hh["collapsed_dmin"])
    hunt_n.append(hh["n"])
    ok_all.append(n_ok == n_tot and hh["pass"])
assert len(set(nscan)) == 1 and len(set(hunt_n)) == 1
NSCAN, NHUNT = nscan[0], hunt_n[0]

ax_b.yaxis.grid(True, color=GRID, ls=":", lw=0.6, zorder=0)
ax_b.set_axisbelow(True)
# bar key = model identity, as in (C): foundation sky '\\'; fine-tuned '//' (navy = passed, light grey = failed)
FAIL_GREY = "#e3e3e3"   # a solid fill keeps the silhouette (every Fig. 2C bar has one); grey = greyed out
STYLE_B = {"found": (SKY, HATCH_SECOND), "ft_pass": (BAR_NAVY, HATCH_HEAD), "ft_fail": (FAIL_GREY, HATCH_HEAD)}
labels_b = []
for x, n, ok, hb, (m, lab, ft) in zip(xb, npass, ok_all, hunt_bad, MODELS_B):
    face, hatch = STYLE_B["found" if not ft else ("ft_pass" if ok else "ft_fail")]
    bars(ax_b, x, n, 0.62, face, hatch)
    labels_b.append(value_label(ax_b, x, n, "%d/%d" % (n, NSCAN), bold=(ok and ft)))   # bold = navy series only
# hole-hunt outcomes are stated in the caption (no second tick line)
ax_b.axhline(NSCAN, color="0.5", lw=0.7, ls="--", zorder=2)
ax_b.set_xticks(xb)
ax_b.set_xticklabels([lab for _, lab, _ in MODELS_B])
ax_b.set_ylabel("Pair-scan cells passed", labelpad=3)
B_YMAX = 25.0                             # bars fill ~64% of the box; two-row legend above the "16/16" labels
ax_b.set_ylim(0, B_YMAX)
ax_b.set_yticks([0, 4, 8, 12, 16])
ax_b.set_xlim(-0.5, len(MODELS_B) - 0.5)
# every fill present in the panel is keyed (Fig. 2C): one-row legend along the top, one-word entries
hb_leg = [swatch(*STYLE_B["ft_pass"]), swatch(*STYLE_B["ft_fail"]), swatch(*STYLE_B["found"])]
leg_b = ax_b.legend(handles=hb_leg, labels=["Passed", "Failed", "Foundation"],
                    loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, columnspacing=1.0,
                    **dict(SWATCH_KW, handlelength=2.0, borderaxespad=0.35))

# =============================================================================== (C)
# key, pair label, system label (Fig. 2C spelling where it fits), probe distance, x position
# pairs of one system sit 1.0 apart; systems are separated by an extra 0.3 (Fig. 2C's group gaps)
PAIRS_C = [("FLi", "F" + EN + "Li", "LiCl/GaF$_3$", 0.9, 0.0),
           ("GaGa", "Ga" + EN + "Ga", "LiCl/GaF$_3$", 0.9, 1.0),
           ("CO", "C" + EN + "O", "Li$_2$CO$_3$/LiF", 0.9, 2.3),
           ("SS", "S" + EN + "S", "Li/Li$_5$PS$_5$Cl", 0.9, 3.6),
           ("LiLi", "Li" + EN + "Li", "Li/Li$_5$PS$_5$Cl", 1.2, 4.6),
           ("BO", "B" + EN + "O", "LPS/LBO", 0.9, 5.9),
           ("LiO", "Li" + EN + "O", "LLZO", 0.9, 7.2)]


def wall_dft(key, dprobe):
    pts = [p for p in kmod[key] if p.get("converged") and "E0_eV" in p]
    emin = min(p["E0_eV"] for p in pts)
    e = [p["E0_eV"] for p in pts if abs(p["d"] - dprobe) < 1e-9]
    assert len(e) == 1, (key, dprobe)
    dmin = [p["d"] for p in pts if p["E0_eV"] == emin][0]
    return e[0] - emin, dmin


def wall_model(key, which, dprobe):
    d = np.asarray(kdim[key]["d"], float)
    E = np.asarray(kdim[key]["E_" + which], float)
    i = int(np.argmin(np.abs(d - dprobe)))
    assert abs(d[i] - dprobe) < 1e-9
    return float(E[i] - E.min()), float(d[int(np.argmin(E))])


C_X0, C_W, C_Y0, C_H = 7.55, 8.35, 1.10, 4.00
ax_c = fig.add_axes(rect(C_X0, C_Y0, C_W, C_H))
xc = np.array([x for *_, x in PAIRS_C], float)
wc = 0.36               # two touching bars fill 0.72 of the pair pitch (Fig. 2C: 0.71 of the group pitch)
walls = {"DFT": [], "base": [], "ft": []}
facts_c = {}
for key, lab, sysn, dp, _x in PAIRS_C:
    wd, dmd = wall_dft(key, dp)
    wm, dmm = wall_model(key, "mpa0", dp)
    wf, dmf = wall_model(key, "ft", dp)
    walls["DFT"].append(wd)
    walls["base"].append(wm)
    walls["ft"].append(wf)
    facts_c[key] = dict(probe_A=dp, DFT=round(wd, 2), base=round(wm, 2), ft=round(wf, 2),
                        ft_vs_DFT_pct=round(100 * (wf - wd) / wd, 1),
                        base_vs_DFT_pct=round(100 * (wm - wd) / wd, 1),
                        dmin_DFT=dmd, dmin_base=dmm, dmin_ft=dmf, system=sysn)


def fmt(v):
    """one decimal throughout (equal precision, as Fig. 2C's two decimals)"""
    return "%.1f" % v


ax_c.yaxis.grid(True, which="major", color=GRID, ls=":", lw=0.6, zorder=0)
ax_c.set_axisbelow(True)
# Fig. 2C sequence: headline first, reference last (bars and legend)
# wall height as a percentage of the DFT wall: two bars per pair, DFT = dashed line at 100
SERIES_C = [(-wc / 2, "ft", BAR_NAVY, HATCH_HEAD, "Fine-tuned", True),    # headline first (Fig. 2C)
            (wc / 2, "base", SKY, HATCH_SECOND, "Foundation", False)]
hc_leg, hc_lab = [], []
labels_c, labels_c_key, patches_c = [], [], []
for off, name, face, hatch, lab, bold in SERIES_C:
    vals = [100.0 * w / wd for w, wd in zip(walls[name], walls["DFT"])]
    cont = bars(ax_c, xc + off, vals, wc, face, hatch)
    hc_leg.append(swatch(face, hatch))
    hc_lab.append(lab)
    for i, (x, v, patch) in enumerate(zip(xc + off, vals, cont)):
        labels_c.append(value_label(ax_c, x, v, "%.0f" % v, bold=bold))
        labels_c_key.append((PAIRS_C[i][0], name))
        patches_c.append(patch)
ax_c.axhline(100, color="0.5", lw=0.7, ls="--", zorder=2)
for t in labels_c:                       # white backing so the DFT line does not strike through the numbers
    t.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.6))
    t.set_zorder(6)
hc_leg.append(Line2D([], [], ls="--", color="0.5", lw=0.7))
hc_lab.append("DFT")
ax_c.set_ylim(0, 128)
ax_c.set_yticks([0, 25, 50, 75, 100])
ax_c.set_xticks(xc)
ax_c.set_xticklabels([lab for _, lab, _, _, _ in PAIRS_C])
# one system name per system, centred under its pair groups (Fig. 2C per-system labels)
groups = []
for i, (_, _, sysn, _, x) in enumerate(PAIRS_C):
    if groups and groups[-1][0] == sysn:
        groups[-1][1].append(x)
    else:
        groups.append((sysn, [x]))
# system names are given in the caption (no second tick line)
ax_c.set_xlim(xc[0] - 0.80, xc[-1] + 0.80)   # first/last value labels clear the spines (Fig. 2C >= 0.3 cm)
ax_c.set_ylabel("Wall retained (% of DFT)", labelpad=3)
leg_c = ax_c.legend(handles=hc_leg, labels=hc_lab, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                    ncol=3, columnspacing=1.0, **dict(SWATCH_KW, handlelength=2.0, borderaxespad=0.35))
FOOT_C = "*Li%sLi probed at 1.2 %s" % (EN, ANG)
FOOT_DY = -34           # hugs the two-line tick block (0.27 cm below it)
# footnote moved to the caption

# panel letters: bold navy serif, outside the axes at the top-left corner of each panel
# (15 pt: cap height 1.55 x the tick digits, the centre of the originals' 1.36-1.72 spread)
PL = dict(fontsize=14, fontweight="bold", color=NAVY, va="baseline", ha="left")   # cap height ~1.5 x tick digits (Figs 2-4: 1.36-1.7)
fig.text(0.12 / W_CM, (A_Y0 + A_H + TITLE_PAD * PT_CM) / H_CM, "(A)", **PL)   # baseline = title baseline
fig.text(0.12 / W_CM, (B_Y0 + B_H + 0.15) / H_CM, "(B)", **PL)               # 0.15 cm above the top spine
fig.text(6.45 / W_CM, (C_Y0 + C_H + 0.15) / H_CM, "(C)", **PL)               # 0.45 cm right of B's spine

# value-label collision handling (needs a renderer: after every artist exists)
# labels stay exactly centred above their own bars: with one label per bar (B) and two-digit
# percentages over 0.34-cm bars (C) no collision handling is needed, and none is applied
lifts_b, lifts_c, leaders_c = [], [], []
fig.canvas.draw()
_r = fig.canvas.get_renderer()
_cm = 2.54 / fig.dpi
_lb = leg_c.get_window_extent(_r)
_bad = [t.get_text() for t in labels_c if t.get_window_extent(_r).overlaps(_lb)]
print("legend (C) x0 = %.2f cm from the left spine; labels under it: %s"
      % ((_lb.x0 - ax_c.get_window_extent(_r).x0) * _cm, _bad))
assert not _bad, "value label under the legend of (C)"
_near = [t for t in labels_c if t.get_window_extent(_r).y1 > _lb.y0]        # labels reaching the legend band
if _near:
    print("legend (C) clears the nearest value label in its band by %.2f cm (%s)"
          % (min(_lb.x0 - t.get_window_extent(_r).x1 for t in _near) * _cm,
             min(_near, key=lambda t: _lb.x0 - t.get_window_extent(_r).x1).get_text()))
else:
    print("legend (C): no value label reaches the legend band")
_lbb = leg_b.get_window_extent(_r)
print("legend (B) clears the value labels below it by %.2f cm; bars fill %.0f%% of the box height"
      % (min(_lbb.y0 - t.get_window_extent(_r).y1 for t in labels_b) * _cm,
         100.0 * NSCAN / ax_b.get_ylim()[1]))
assert _lbb.y0 > max(t.get_window_extent(_r).y1 for t in labels_b), "legend of (B) over a value label"
_dist, _who = legend_marker_clearance_cm(fig, axes_a[LEGEND_PANEL_A], leg_a)
_la = leg_a.get_window_extent(_r)
_aa = axes_a[LEGEND_PANEL_A].get_window_extent(_r)
print("legend (A) in %s: %.2f x %.2f cm (%.0f%% x %.0f%% of the sub-panel); nearest marker %s at %.2f cm"
      % (PAIRS_A[LEGEND_PANEL_A], _la.width * _cm, _la.height * _cm,
         100 * _la.width / _aa.width, 100 * _la.height / _aa.height, _who, _dist))
assert _dist >= LEGEND_CLEAR_CM, "a plotted marker comes within %.2f cm of the legend of (A)" % LEGEND_CLEAR_CM
# the two Na-Na annotation blocks stay >= 0.2 cm inside the box (Fig. 4C: about one 9-pt em);
# Annotation.get_window_extent would include the arrow, so the text box is taken from Text
from matplotlib.text import Text as _Text  # noqa: E402
_axn = axes_a[0].get_window_extent(_r)
for t in axes_a[0].texts:
    tb = _Text.get_window_extent(t, _r)
    ap_ = t.arrow_patch.get_window_extent(_r)
    print("annotation %-28r text clears spines by L %.2f R %.2f B %.2f T %.2f cm; arrow %.2f cm long"
          % (t.get_text().replace("\n", " / "), (tb.x0 - _axn.x0) * _cm, (_axn.x1 - tb.x1) * _cm,
             (tb.y0 - _axn.y0) * _cm, (_axn.y1 - tb.y1) * _cm, np.hypot(ap_.width, ap_.height) * _cm))
    assert min(tb.x0 - _axn.x0, _axn.x1 - tb.x1, tb.y0 - _axn.y0, _axn.y1 - tb.y1) * _cm >= 0.2

pdf = os.path.join(OUT, "fig5_application.pdf")
png = os.path.join(OUT, "fig5_application.png")
fig.savefig(pdf)
fig.savefig(png, dpi=300)
print("saved", pdf, "and", png)
print("serif font used: %s; mathtext subscript tweak active: %s" % (SERIF, SUBSCRIPT_TWEAK))

# ------------------------------------------------------------------------ facts
print("\n== (A) dimer facts ==")
for p in PAIRS_A:
    d, dE, E = curves[p]
    s = "%-6s" % p
    for m in ["dft"] + [m[0] for m in MODELS_A]:
        i = int(np.argmin(E[m]))
        s += "  %s: min %.2f A (dE %+.2f eV)" % (m, d[i], dE[m][i])
    print(s)
print("caption: dE = E(d) - E(3.5 A); curves clipped at 8 eV (y ticks stop at 6);",
      "v1 = before repair, v2 = first repair, v2.1 = repaired; grey dots = DFT")
print("\n== (B) audit facts ==")
for (m, lab, ft), n, hb in zip(MODELS_B, npass, hunt_bad):
    fails = [k for k, v in g2[m]["scans"].items() if not v["pass"]]
    print("%-10s scans %2d/%d  hole-hunt collapsed %d/%d  worst dmin %.2f A  fails: %s"
          % (lab, n, NSCAN, hb, NHUNT, g2[m]["hole_hunt"]["worst_dmin_A"], fails))
print("caption: MPA-0 = MACE-MPA-0 foundation model; second tick line ('hunt' k/%d) = k of %d random"
      " hole-hunt starts collapsed; light-grey '//' bars = fine-tuned models that failed the audit" % (NHUNT, NHUNT))
print("label moves (B):", [x for x in lifts_b if x[1] or x[2]])
print("\n== (C) wall facts (3 s.f. for the SI table; drawn labels carry one decimal) ==")
for k, v in facts_c.items():
    print(k, v)
print("label moves (C): sideways push / lift in points (lift is above the standard 2.5-pt offset):")
for (k, name), (txt, push, lift) in zip(labels_c_key, lifts_c):
    if push or lift:
        print("   %-5s %-4s %-6s push %+.1f pt  lift %.1f pt" % (k, name, txt, push, lift))
print("leaders (C): %s" % leaders_c)
print("footnote:", FOOT_C, "| caption: LPS/LBO = Li3PS4/Li3B11O18, LLZO = Li7La3ZrxMxO12;",
      "value labels one decimal throughout")
