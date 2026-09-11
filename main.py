import flet as ft
import math
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sympy import Interval, S, oo, FiniteSet, EmptySet, Union


def format_interval(iv):
    if iv == S.EmptySet or iv == EmptySet:
        return "∅"
    if isinstance(iv, FiniteSet):
        vals = sorted(iv.args, key=lambda p: float(p))
        return "{" + ", ".join(str(v) for v in vals) + "}"
    if isinstance(iv, Interval):
        lb = "-∞" if iv.start == -oo else str(iv.start)
        rb = "+∞" if iv.end == oo else str(iv.end)
        left = "]" if iv.left_open else "["
        right = "[" if iv.right_open else "]"
        return f"{left}{lb}, {rb}{right}"
    if isinstance(iv, Union):
        def _sort_key(p):
            if hasattr(p, "start") and p.start != -oo:
                return float(p.start)
            return float("-inf")
        parts = sorted(iv.args, key=_sort_key)
        return " ∪ ".join(format_interval(p) for p in parts)
    return str(iv)


def parse_bound(text):
    t = str(text).strip().lower()
    if t in ("inf", "infinity", "oo", "+oo", "+inf"):
        return oo
    if t in ("-inf", "-infinity", "-oo"):
        return -oo
    try:
        return int(t)
    except ValueError:
        return float(t)


def fmt_tick(v):
    f = float(v)
    if math.isfinite(f) and f.is_integer():
        return str(int(f))
    return f"{f:g}"


def finite_bounds(iv):
    out = []
    if iv in (S.EmptySet, EmptySet):
        return out
    if isinstance(iv, FiniteSet):
        return [float(p) for p in iv.args]
    if hasattr(iv, "start") and hasattr(iv, "end"):
        if iv.start != -oo:
            out.append(float(iv.start))
        if iv.end != oo:
            out.append(float(iv.end))
        return out
    if hasattr(iv, "args"):
        for piece in iv.args:
            if hasattr(piece, "start") and hasattr(piece, "end"):
                if piece.start != -oo:
                    out.append(float(piece.start))
                if piece.end != oo:
                    out.append(float(piece.end))
    return out


def compute_view_range(*ivs):
    vals = []
    for iv in ivs:
        vals.extend(finite_bounds(iv))
    if not vals:
        return -5.0, 5.0
    lo, hi = min(vals), max(vals)
    span = hi - lo
    pad = max(span * 0.5, 1.0) if span > 1e-9 else max(1.0, abs(lo) * 0.3)
    return lo - pad, hi + pad


def render_number_lines(A, B, result_iv, result_label):
    vmin, vmax = compute_view_range(A, B)
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.0), dpi=110,
                             gridspec_kw={"hspace": 1.0})
    fig.patch.set_facecolor("#ffffff")
    configs = [
        (axes[0], A, "#2563eb", "A"),
        (axes[1], B, "#db2777", "B"),
        (axes[2], result_iv, "#059669", result_label),
    ]
    for ax, iv, color, label in configs:
        _draw_axis(ax, iv, color, label, vmin, vmax)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    return buf.getvalue()


def _draw_axis(ax, iv, color, label, vmin, vmax):
    ax.clear()
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(-0.75, 0.85)
    ax.get_yaxis().set_visible(False)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([])
    ax.tick_params(axis="x", length=0, labelbottom=False)

    span = vmax - vmin
    is_empty = (iv == S.EmptySet) or (iv == EmptySet)
    is_finite_set = (not is_empty) and isinstance(iv, FiniteSet)

    if span <= 12:
        step = 1
    elif span <= 30:
        step = 2
    elif span <= 60:
        step = 5
    elif span <= 150:
        step = 10
    else:
        step = 10 ** max(0, int(math.floor(math.log10(span / 10))))

    tick_vals = set()
    t = math.ceil(vmin / step) * step
    while t <= vmax + 1e-9:
        if vmin + 1e-9 < t < vmax - 1e-9:
            tick_vals.add(round(t, 6))
        t += step

    bound_vals = set()
    for b in finite_bounds(iv):
        b = round(b, 6)
        if vmin + 1e-9 < b < vmax - 1e-9:
            tick_vals.add(b)
            bound_vals.add(b)

    for t in sorted(tick_vals):
        is_bound = t in bound_vals
        ax.axvline(t, color="#eef2f7", linewidth=1, zorder=0)
        ax.plot([t, t], [-0.05, 0.05], color="#94a3b8", linewidth=1.2, zorder=3)
        ax.text(t, -0.22, fmt_tick(t),
                ha="center", va="top",
                color=("#1e293b" if is_bound else "#64748b"),
                fontdict={"weight": "bold" if is_bound else "normal",
                          "size": 11}, zorder=6)

    ax.plot([vmin, vmax], [0, 0], color="#94a3b8",
            linewidth=1.4, zorder=1, solid_capstyle="round")

    ax.annotate("", xy=(vmin, 0), xytext=(vmin + span * 0.04, 0),
                arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=1.4),
                zorder=6, annotation_clip=False)
    ax.annotate("", xy=(vmax, 0), xytext=(vmax - span * 0.04, 0),
                arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=1.4),
                zorder=6, annotation_clip=False)

    ax.text(vmin, -0.30, "-∞", ha="left", va="top", color="#475569",
            fontdict={"weight": "bold", "size": 12}, zorder=6, clip_on=False)
    ax.text(vmax, -0.30, "+∞", ha="right", va="top", color="#475569",
            fontdict={"weight": "bold", "size": 12}, zorder=6, clip_on=False)

    if is_empty:
        ax.text((vmin + vmax) / 2, 0, "∅   empty set",
                ha="center", va="center", color=color,
                fontdict={"weight": "bold", "size": 12}, zorder=5)
        return

    if is_finite_set:
        pts = sorted(float(p) for p in iv.args)
        for p in pts:
            ax.plot(p, 0, marker="o", markerfacecolor=color,
                    markeredgecolor=color, markeredgewidth=2,
                    markersize=9, zorder=5)
        mid = sum(pts) / len(pts)
        lx = min(max(mid, vmin + span * 0.06), vmax - span * 0.06)
        ax.text(lx, 0.30, label, color=color, ha="center",
                fontdict={"weight": "bold", "size": 12}, zorder=5)
        return

    if hasattr(iv, "start") and hasattr(iv, "end"):
        pieces = [iv]
    else:
        pieces = [p for p in iv.args
                  if hasattr(p, "start") and hasattr(p, "end")]

    mids = []
    for piece in pieces:
        st, en = piece.start, piece.end
        ps = vmin if st == -oo else float(st)
        pe = vmax if en == oo else float(en)
        ps_draw = max(ps, vmin)
        pe_draw = min(pe, vmax)
        ax.plot([ps_draw, pe_draw], [0, 0], color=color,
                linewidth=6, zorder=2, solid_capstyle="round")
        if st != -oo and vmin < float(st) < vmax:
            ax.plot(float(st), 0, marker="o",
                    markerfacecolor=("white" if piece.left_open else color),
                    markeredgecolor=color, markeredgewidth=2,
                    markersize=9, zorder=5)
        if en != oo and vmin < float(en) < vmax:
            ax.plot(float(en), 0, marker="o",
                    markerfacecolor=("white" if piece.right_open else color),
                    markeredgecolor=color, markeredgewidth=2,
                    markersize=9, zorder=5)
        mids.append((ps_draw + pe_draw) / 2)

    if mids:
        label_x = sum(mids) / len(mids)
        label_x = min(max(label_x, vmin + span * 0.06), vmax - span * 0.06)
        ax.text(label_x, 0.30, label, color=color, ha="center",
                fontdict={"weight": "bold", "size": 12}, zorder=5)


def main(page):
    # ---- ultra-minimal Flet API only. No enums, no renamed functions. ----
    page.title = "Interval Analyzer & Plotter"
    page.bgcolor = "#f8f9fa"

    U = S.Reals

    # ---- build one interval input block ----
    def make_input(name, d_s, d_e, d_l, d_r):
        se = ft.TextField(label="Start", value=d_s, width=120)
        ee = ft.TextField(label="End", value=d_e, width=120)
        lo = ft.Dropdown(value=d_l, width=170,
                         options=[ft.dropdown.Option("Closed ( [ )"),
                                  ft.dropdown.Option("Open ( ] )")])
        ro = ft.Dropdown(value=d_r, width=170,
                         options=[ft.dropdown.Option("Open ( [ )"),
                                  ft.dropdown.Option("Closed ( ] )")])

        r1 = ft.Row([ft.Text("Left Bound:"), lo,
                     ft.Text("Start:"), se], spacing=8)
        r2 = ft.Row([ft.Text("End:"), ee,
                     ft.Text("Right Bound:"), ro], spacing=8)

        inner = ft.Container(content=ft.Column([r1, r2], spacing=6),
                             padding=12, bgcolor="#e2e8f0")
        return {
            "start": se, "end": ee,
            "left_open": lo, "right_open": ro,
            "block": ft.Column([ft.Text(f"Interval {name}", size=14),
                                inner], spacing=2),
        }

    A_in = make_input("A", "-2", "inf", "Closed ( [ )", "Open ( [ )")
    B_in = make_input("B", "-inf", "3", "Open ( ] )", "Closed ( ] )")

    # ---- graph choice ----
    graph_dd = ft.Dropdown(
        value="Intersection (A ∩ B)", width=280,
        options=[ft.dropdown.Option("Intersection (A ∩ B)"),
                 ft.dropdown.Option("Union (A ∪ B)"),
                 ft.dropdown.Option("Difference (A - B)"),
                 ft.dropdown.Option("Difference (B - A)")])

    # ---- results ----
    labels = {}
    rows = []
    for key, title in [
        ("union", "Union (A ∪ B):"),
        ("intersection", "Intersection (A ∩ B):"),
        ("a_minus_b", "Difference (A - B):"),
        ("b_minus_a", "Difference (B - A):"),
        ("a_dash", "Complement (A'):"),
        ("b_dash", "Complement (B'):"),
    ]:
        v = ft.Text("", color="#059669", selectable=True)
        labels[key] = v
        rows.append(ft.Row([ft.Text(title, width=200), v], spacing=8))

    results_box = ft.Container(content=ft.Column(rows, spacing=4),
                               padding=12, bgcolor="#e2e8f0")

    # ---- plot image ----
    plot_image = ft.Image(src=base64.b64encode(b"").decode(),
                          width=640, height=600)
    plot_box = ft.Container(content=plot_image, padding=10, bgcolor="#e2e8f0")

    # ---- helpers ----
    def build_interval(ip):
        v1 = parse_bound(ip["start"].value)
        v2 = parse_bound(ip["end"].value)
        lo = "Open" in ip["left_open"].value
        ro = "Open" in ip["right_open"].value
        if v1 != -oo and v2 != -oo and v1 != oo and v2 != oo and v1 > v2:
            v1, v2 = v2, v1
            lo, ro = ro, lo
        if v1 == -oo:
            lo = True
        if v2 == oo:
            ro = True
        return Interval(v1, v2, left_open=lo, right_open=ro)

    def calculate(e):
        try:
            A = build_interval(A_in)
            B = build_interval(B_in)
        except Exception as ex:
            page.snack_bar = ft.SnackBar(ft.Text(f"Input Error: {ex}"))
            page.snack_bar.open = True
            page.update()
            return

        ops = {
            "union": A.union(B),
            "intersection": A.intersect(B),
            "a_minus_b": A - B,
            "b_minus_a": B - A,
            "a_dash": U - A,
            "b_dash": U - B,
        }
        for k, res in ops.items():
            labels[k].value = format_interval(res)

        gc = graph_dd.value
        if "Union" in gc:
            r_iv, r_lbl = ops["union"], "A ∪ B"
        elif "Intersection" in gc:
            r_iv, r_lbl = ops["intersection"], "A ∩ B"
        elif "A - B" in gc:
            r_iv, r_lbl = ops["a_minus_b"], "A - B"
        else:
            r_iv, r_lbl = ops["b_minus_a"], "B - A"

        png = render_number_lines(A, B, r_iv, r_lbl)
        plot_image.src_base64 = base64.b64encode(png).decode()
        page.update()

    def clear_all(e):
        for ip in (A_in, B_in):
            ip["start"].value = ""
            ip["end"].value = ""
        for lbl in labels.values():
            lbl.value = ""
        plot_image.src_base64 = ""
        page.update()

    # ---- assemble page ----
    page.add(ft.Text("Interval Analyzer & Plotter", size=24))
    page.add(A_in["block"])
    page.add(B_in["block"])
    page.add(ft.Row([graph_dd], spacing=8))
    page.add(ft.Row([
        ft.ElevatedButton("Analyze & Plot", on_click=calculate,
                          bgcolor="#2563eb", color="white"),
        ft.ElevatedButton("Clear All", on_click=clear_all,
                          bgcolor="#ef4444", color="white"),
    ], spacing=20))
    page.add(results_box)
    page.add(plot_box)

    calculate(None)


if __name__ == "__main__":
    ft.app(target=main)
