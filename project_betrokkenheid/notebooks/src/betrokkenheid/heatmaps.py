import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
sns.set_theme(style="white")
# =====================================================
# 5. Plot hoofdeffect-heatmap (1 kolom)
# =====================================================

def plot_hoofdeffect_heatmap(
    delta_main_pct,
    sig_main,
    labels_full,
    effect_sizes=None,
    topk_effect=3,
    title_suffix=""
):
    """
    1-koloms heatmap:
    - X: "Gen X vs Niet-Gen X"
    - Y: vragen
    Tekst toont:
    - delta% + '*' bij significantie
    - '★' bij top-k laagste effect sizes (d) (meest negatief)
    """

    # Zorg dat alles aligned is op dezelfde volgorde
    # (labels_full hoort dezelfde volgorde te volgen als delta_main_pct.index)
    Z = delta_main_pct.values.reshape(1, -1)

    sig_arr = np.asarray(sig_main)
    sig_mask = sig_arr.reshape(1, -1)

    fig, ax = plt.subplots(figsize=(5, 8))

    cmap = plt.get_cmap("RdBu").copy()
    absmax = np.nanmax(np.abs(Z)) if np.isfinite(Z).any() else 1
    norm = TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)

    Z_color = np.ma.array(Z, mask=~sig_mask)
    im = ax.imshow(
        Z_color.T,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        origin="upper",
        interpolation="none"
    )

    n_rows = len(labels_full)

    ax.grid(False)
    ax.tick_params(which="both", length=0)

    # Grenzen tussen rijen
    for y in range(n_rows + 1):
        ax.plot([-0.5, 0.5], [y - 0.5, y - 0.5], color="gray", linewidth=0.6)

    # --- NEW: bepaal welke items een ★ krijgen (bottom-k d) ---
    star_idx = set()
    if effect_sizes is not None and topk_effect and topk_effect > 0:
        # effect_sizes is pd.Series met index = vragen
        # align op delta_main_pct.index
        d_aligned = effect_sizes.reindex(delta_main_pct.index)

        d_vals = d_aligned.values
        valid = np.isfinite(d_vals)
        if valid.any():
            order = np.argsort(d_vals[valid])  # laagste eerst
            order = order[:min(topk_effect, len(order))]
            valid_idx = np.where(valid)[0]
            chosen = valid_idx[order]
            star_idx = set(chosen)

    # Tekst
    for i, val in enumerate(delta_main_pct.values):
        if not np.isfinite(val):
            continue

        is_sig = bool(sig_arr[i])
        text_color = "white" if is_sig else "black"

        sig_star = "*" if is_sig else ""
        eff_star = "★" if i in star_idx else ""

        txt = f"{val:.1f}%{sig_star}{eff_star}"
        weight = "bold" if i == 0 else "normal"  # Betrokkenheid vet

        ax.text(
            0, i, txt,
            ha="center", va="center",
            color=text_color,
            fontsize=9,
            fontweight=weight
        )

    ax.set_xticks([0])
    ax.set_xticklabels(["Gen X vs. Niet-Gen X"])

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(labels_full)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Betrokkenheid-label vet (als die echt de eerste is)
    if len(ax.get_yticklabels()) > 0:
        ax.get_yticklabels()[0].set_fontweight("bold")

    # Lijn onder Betrokkenheid
    ax.plot([-0.5, 0.5], [0.5, 0.5], color="black", linewidth=1.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Afwijking t.o.v. referentie (Niet-Gen X) — % van 1–5-schaal")

    titel = "Generatie X vs. Niet-Gen X — hoofdeffect per vraag"
    if title_suffix:
        titel += f" ({title_suffix})"
    ax.set_title(titel)

    fig.text(
        0.5, -0.03,
        "* = FDR-gecorrigeerd p < 0.05  |  ★ = top 3 laagste effect size (d)  |  Blauw = positiever, Rood = negatiever",
        ha="center", fontsize=9, style="italic"
    )

    plt.tight_layout()
    plt.show()


def bottomk_effect_positions(d_sub, k=3):
    """
    d_sub: DataFrame (groepen x vragen) al gefilterd op top5_questions
    return: set van (gi, qi) posities van de k laagste (meest negatieve) d's
    """
    vals = d_sub.values
    flat = vals.ravel()

    valid = np.isfinite(flat)
    flat_valid = flat[valid]

    # laagste k
    order = np.argsort(flat_valid)  # ascending
    top_idx_valid = order[:k]

    # indices terug naar original flatten index
    valid_flat_indices = np.where(valid)[0]
    chosen_flat = valid_flat_indices[top_idx_valid]

    n_rows, n_cols = vals.shape
    pos = set()
    for fi in chosen_flat:
        gi = fi // n_cols
        qi = fi % n_cols
        pos.add((gi, qi))
    return pos


def plot_top5_heatmap(
    delta, sig, top5_labels, label_to_question,
    effect_sizes=None, topk_effect=3,
    group_label="Groep", title_suffix="",
    show_nonsig_values=True,
    nonsig_text_color="#777777"
):
    """
    Witte achtergrond, alleen significante cellen ingekleurd.
    - Significant: kleur + dikgedrukt + '*'
    - Top-k laagste effect size: '★' + onderstreping
    """

    # =========================================================
    # HARD RESET VAN ALLE DARK MODE / STYLES
    # =========================================================
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use("default")

    # ---------------------------------------------------------
    # 1) labels -> originele vragen
    # ---------------------------------------------------------
    top5_questions = [label_to_question[lbl] for lbl in top5_labels]

    # ---------------------------------------------------------
    # 2) filter delta + sig naar top-5 vragen
    # ---------------------------------------------------------
    delta_top5 = delta[top5_questions]
    col_idx = [delta.columns.get_loc(q) for q in top5_questions]
    sig_top5 = sig[:, col_idx]  # shape (n_groups, 5)

    # korte labels voor plot
    delta_plot = delta_top5.copy()
    delta_plot.columns = top5_labels

    Z = delta_plot_query = delta_plot.values
    n_groups, n_themes = Z.shape

    # thema's op Y, groepen op X
    Z_plot = Z.T
    sig_plot = sig_top5.T

    # ---------------------------------------------------------
    # 3) FIGUUR (WIT)
    # ---------------------------------------------------------
    fig_w = 1.25 * n_groups + 6
    fig_h = 1.05 * n_themes + 2.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")

    # ---------------------------------------------------------
    # 4) COLORMAP – alleen significant
    # ---------------------------------------------------------
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad(color="white")  # niet-significant = wit

    absmax = np.nanmax(np.abs(Z_plot)) if np.isfinite(Z_plot).any() else 1.0
    norm = TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)

    Z_color = np.ma.array(Z_plot, mask=~sig_plot)

    im = ax.imshow(
        Z_color,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        origin="upper",
        interpolation="none"
    )

    # ---------------------------------------------------------
    # 5) ASSEN & GRID
    # ---------------------------------------------------------
    ax.set_xticks(np.arange(n_groups))
    ax.set_xticklabels(delta_plot.index, fontsize=10)

    ax.set_yticks(np.arange(n_themes))
    ax.set_yticklabels(top5_labels, fontsize=11)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    ax.set_xticks(np.arange(-.5, n_groups, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_themes, 1), minor=True)
    ax.grid(which="minor", color="#E0E0E0", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    # ---------------------------------------------------------
    # 6) TOP-K EFFECT SIZE POSITIES
    # ---------------------------------------------------------
    effect_star_pos = set()
    if effect_sizes is not None and topk_effect and topk_effect > 0:
        d_sub = effect_sizes.loc[delta.index, top5_questions]
        effect_star_pos = bottomk_effect_positions(d_sub, k=topk_effect)
        # verwacht: set van (gi, ti)

    # ---------------------------------------------------------
    # 7) ANNOTATIES
    # ---------------------------------------------------------
    for gi in range(n_groups):
        for ti in range(n_themes):
            val = Z[gi, ti]
            if not np.isfinite(val):
                continue

            is_sig = bool(sig_top5[gi, ti])
            is_top_effect = (gi, ti) in effect_star_pos

            if not is_sig and not show_nonsig_values:
                continue

            txt = f"{val:+.1f}%"
            if is_sig:
                txt += "*"
            if is_top_effect:
                txt += "★"

            if is_sig:
                color = "black"
                fontsize = 11
                weight = "bold"
            else:
                color = nonsig_text_color
                fontsize = 10
                weight = "normal"

            ax.text(
                gi, ti, txt,
                ha="center", va="center",
                color=color,
                fontsize=fontsize,
                fontweight=weight,
                zorder=6
            )

            # underline bij top-3 laagste effect size
            if is_top_effect:
                ax.plot(
                    [gi - 0.32, gi + 0.32],
                    [ti + 0.34, ti + 0.34],
                    color=color,
                    linewidth=2.2,
                    solid_capstyle="round",
                    zorder=7
                )

    # ---------------------------------------------------------
    # 8) COLORBAR (OOK WIT)
    # ---------------------------------------------------------
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label(
        "Afwijking (Niet-Gen X – Gen X) — % van 1–5-schaal",
        fontsize=10,
        color="black"
    )
    cbar.ax.set_facecolor("white")
    cbar.ax.tick_params(colors="black")
    cbar.outline.set_edgecolor("black")

    # ---------------------------------------------------------
    # 9) TITEL & VOETNOOT
    # ---------------------------------------------------------
    title = f"Niet-Gen X vs. Gen X per {group_label} — Top 5 thema's"
    if title_suffix:
        title += f" ({title_suffix})"

    ax.set_title(title, fontsize=13, pad=14, color="black")

    fig.text(
        0.5, 0.02,
        "* = FDR-gecorrigeerd p < 0.05  |  ★ + onderstreept = top 3 laagste effect size (d) binnen dit blok  |  Alleen significante cellen ingekleurd",
        ha="center", fontsize=9, style="italic", color="black"
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


import matplotlib as mpl

def plot_hoofdeffect_heatmap_adjusted(
    eff_pct,
    sig_main,
    labels_full,
    title="",
    effect_sizes=None,          # optional: array-like same length as eff_pct
    topk_effect=0,              # set to 3 to mark top-3 lowest d
    ax=None,
    show_colorbar=False,
    vlim=20,                    # fixed scale [-20, 20]
    show_nonsig_values=True,
    nonsig_text_color="#777777",
    add_footnote=False,
    footnote_note=""
):
    """
    1-koloms heatmap (Voor/Na) met:
    - Witte achtergrond (force reset)
    - Alleen significante cellen ingekleurd (RdBu)
    - Significant: bold + '*'
    - Optioneel: top-k laagste effect size: '★' + underline
    - Footnote optioneel (typisch 1x voor hele fig)
    """

    # ---- hard reset van dark mode / rcparams (alleen voor deze plot) ----
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use("default")

    eff_vals = np.asarray(eff_pct, dtype=float).reshape(-1)
    sig_main = np.asarray(sig_main, dtype=bool).reshape(-1)

    n_rows = len(labels_full)
    if eff_vals.shape[0] != n_rows or sig_main.shape[0] != n_rows:
        raise ValueError("eff_pct, sig_main en labels_full moeten dezelfde lengte hebben.")

    # Z is (n_rows, 1)
    Z = eff_vals.reshape(n_rows, 1)
    sig_mask = sig_main.reshape(n_rows, 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 8))
    else:
        fig = ax.figure

    # ---- force white everywhere ----
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("#B0B0B0")  # subtiel

    # ---- colormap: alleen significant kleuren ----
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad(color="white")  # masked (niet-significant) = wit
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)

    Z_color = np.ma.array(Z, mask=~sig_mask)  # niet-sig gemaskeerd
    im = ax.imshow(
        Z_color,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        origin="upper",
        interpolation="none"
    )

    # ---- subtiele gridlijnen tussen rijen ----
    ax.set_xticks(np.arange(-.5, 1, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#E0E0E0", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    # ---- labels ----
    ax.set_xticks([])  # geen x ticks; title boven is duidelijker
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(labels_full, fontsize=11, color="black")

    # ---- top-k effect size posities (optioneel) ----
    # We markeren "laagste d" -> kleinste waarden (meest negatief) OF echt "laagste" zoals jij definieert.
    # Hier: laagste numerieke waarde.
    underline_pos = set()
    if effect_sizes is not None and topk_effect and topk_effect > 0:
        d = np.asarray(effect_sizes, dtype=float).reshape(-1)
        if d.shape[0] != n_rows:
            raise ValueError("effect_sizes moet dezelfde lengte hebben als eff_pct.")
        # indices van k kleinste d (negeer NaN)
        valid = np.isfinite(d)
        idx = np.argsort(d[valid])[:topk_effect]
        underline_pos = set(np.where(valid)[0][idx].tolist())

    # ---- tekst in cellen ----
    for i, val in enumerate(eff_vals):
        if not np.isfinite(val):
            continue

        is_sig = bool(sig_main[i])
        is_top = i in underline_pos

        if (not is_sig) and (not show_nonsig_values):
            continue

        txt = f"{val:+.1f}%"
        if is_sig:
            txt += "*"
        if is_top:
            txt += "★"

        if is_sig:
            color = "black"
            weight = "bold"
            fontsize = 11
        else:
            color = nonsig_text_color
            weight = "normal"
            fontsize = 10

        ax.text(
            0, i, txt,
            ha="center", va="center",
            color=color, fontsize=fontsize, fontweight=weight,
            zorder=6
        )

        # underline voor top-k effect size
        if is_top:
            ax.plot(
                [-0.18, 0.18],          # onderstreep breedte binnen cel
                [i + 0.32, i + 0.32],    # net onder tekst
                color=color,
                linewidth=2.2,
                solid_capstyle="round",
                zorder=7
            )

    # ---- title ----
    ax.set_title(title, fontsize=13, pad=12, color="black")

    # ---- colorbar (optioneel, meestal alleen op rechter subplot) ----
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.06)
        cbar.set_label("Afwijking Gen X t.o.v. Niet-Gen X (% van 1–5 schaal)", fontsize=10, color="black")
        cbar.ax.set_facecolor("white")
        cbar.ax.tick_params(colors="black")
        cbar.outline.set_edgecolor("#B0B0B0")

    # ---- footnote (liefst 1x per figure) ----
    if add_footnote:
        foot = "* = p < 0.05"
        foot += "\n★ + onderstreept = top 3 laagste effect size (d)" if (effect_sizes is not None and topk_effect) else ""
        if footnote_note:
            foot += f"\n{footnote_note}"
        foot += "\nBlauw = Gen X positiever, Rood = Gen X negatiever (alleen significant ingekleurd)"
        fig.text(0.5, 0.01, foot, ha="center", fontsize=9, style="italic", color="black")

    return ax


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def plot_corr_highlight(
    df: pd.DataFrame,
    cols: list[str],
    label_map: dict | None = None,
    thr: float = 0.5,
    method: str = "pearson",
    figsize=(10, 8),
    decimals: int = 2,
    title: str | None = None,
):
    """
    Correlatiematrix plotten met:
    - highlight alleen |r| >= thr (rest wit)
    - annotaties (correlatiewaarden) in de cellen
    - colorbar -1..1 met wit rond 0

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe met data
    cols : list[str]
        Kolommen die je in de correlatiematrix wilt
    label_map : dict | None
        Mapping van volledige kolomnamen naar korte labels
    thr : float
        Highlight drempel (|r| >= thr)
    method : str
        "pearson" of "spearman"
    decimals : int
        Aantal decimalen in annotaties
    """

    # 1) Corr berekenen (numeriek maken)
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = X.corr(method=method)

    # 2) Mask: alles onder threshold wit
    Z = corr.values.astype(float)
    Z_masked = np.ma.array(Z, mask=np.abs(Z) < thr)

    # 3) Plot
    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad(color="white")  # gemaskeerde cellen -> wit
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    im = ax.imshow(Z_masked, cmap=cmap, norm=norm, aspect="auto")

    # 4) Labels
    labels = [label_map.get(c, c) if label_map else c for c in corr.columns]
    n = len(labels)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    # 5) Annotaties (alleen in highlighted cellen)
    fmt = f"{{:.{decimals}f}}"
    for i in range(n):
        for j in range(n):
            val = corr.iat[i, j]
            if np.isfinite(val) and abs(val) >= thr:
                # kies tekstkleur op basis van intensiteit
                txt_color = "white" if abs(val) > 0.65 else "black"
                ax.text(j, i, fmt.format(val), ha="center", va="center",
                        fontsize=8, color=txt_color)

    # 6) Titel + colorbar
    if title is None:
        title = f"Correlatiematrix (highlight |r| ≥ {thr})"
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{method.title()} correlatie (r)")

    plt.tight_layout()
    plt.show()

    return corr  # handig om ook als tabel terug te hebben

