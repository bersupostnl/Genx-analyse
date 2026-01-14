import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# =====================================================
# Analyse hoofdeffect (zonder dienstjaren) + effect sizes
# =====================================================

def analyse_hoofdeffect(df_genx, df_nonx, questions_full):
    """
    Retourneert:
    - delta_main_pct: Series met Δ% per vraag (Gen X – Niet-Gen X)
    - sig_main: boolean-array met significantie (FDR < 0.05)
    - effect_sizes: Series met Cohen's d per vraag
    """

    means_genx = df_genx[questions_full].mean()
    means_nonx = df_nonx[questions_full].mean()

    delta_main = means_genx - means_nonx

    # Welch t-test per vraag
    pvals_main = []
    effect_sizes = {}

    for q in questions_full:
        gvals = pd.to_numeric(df_genx[q], errors="coerce").dropna()
        nvals = pd.to_numeric(df_nonx[q], errors="coerce").dropna()

        # T-test
        if len(gvals) > 2 and len(nvals) > 2:
            _, p = ttest_ind(gvals, nvals, equal_var=False, nan_policy="omit")
            pvals_main.append(p)
        else:
            pvals_main.append(np.nan)

        # Effect size (Cohen's d)
        if len(gvals) > 1 and len(nvals) > 1:
            pooled_sd = np.sqrt((gvals.var() + nvals.var()) / 2)
            d = (gvals.mean() - nvals.mean()) / pooled_sd
        else:
            d = np.nan

        effect_sizes[q] = d

    # FDR correctie
    pvals_main = np.array(pvals_main)
    mask = ~np.isnan(pvals_main)
    pvals_corr_main = np.full_like(pvals_main, np.nan)

    if mask.sum() > 0:
        _, corr_main, _, _ = multipletests(
            pvals_main[mask], method="fdr_bh", alpha=0.05
        )
        pvals_corr_main[mask] = corr_main

    delta_main_pct = (delta_main / 4.0) * 100.0
    sig_main = pvals_corr_main < 0.05

    effect_sizes = pd.Series(effect_sizes)

    return delta_main_pct, sig_main, effect_sizes

import pandas as pd

def maak_effectsamenvatting(
    delta_main_pct,
    sig_main,
    effect_sizes,
    top_n=None
):
    """
    Maakt een overzichtelijke samenvattingstabel van hoofdeffectresultaten.

    Parameters
    ----------
    delta_main_pct : pd.Series
        Percentuele verandering per groep/variabele.

    sig_main : pd.Series
        Significantiefilter (True/False) op basis van FDR < 0.05.

    effect_sizes : pd.Series
        Effect size (Cohen's d).

    top_n : int, optional
        Aantal rijen om terug te geven, gesorteerd op absolute effectgrootte.
        Laat None om alles terug te geven.

    Returns
    -------
    pd.DataFrame
        Gesorteerde tabel met Delta, Significantie en Effect Size.
    """

    summary = pd.DataFrame({
        "Delta (%)": delta_main_pct,
        "Significant (FDR<0.05)": sig_main,
        "Effect Size (d)": effect_sizes
    })

    # sorteer op absolute effect size
    summary_sorted = summary.reindex(
        summary["Effect Size (d)"].abs().sort_values(ascending=False).index
    )

    if top_n is not None:
        return summary_sorted.head(top_n)

    return summary_sorted


def analyse_grouped(df_genx, df_nonx, questions_full, group_col, group_order):
    """
    Algemene analysefunctie:
    Gen X vs Niet Gen X binnen categorieën van group_col.

    Retourneert:
    - delta_disp: Δ% matrix (Non-X – Gen X)
    - sig: boolean significantiematrix (FDR < 0.05)
    - group_order: volgorde van groepen
    - d_matrix: effect sizes (Cohen's d)
    """

    # Gemiddelden per groep
    means_nonx = (
        df_nonx.groupby(group_col)[questions_full]
        .mean()
        .reindex(group_order)
    )

    means_genx = (
        df_genx.groupby(group_col)[questions_full]
        .mean()
        .reindex(group_order)
    )

    # Verschillen
    delta = means_genx - means_nonx

    # p-waardes + Cohen’s d
    pvals = pd.DataFrame(index=group_order, columns=questions_full, dtype=float)
    d_matrix = pd.DataFrame(index=group_order, columns=questions_full, dtype=float)

    for g in group_order:

        a_nonx = df_nonx[df_nonx[group_col] == g]
        a_genx = df_genx[df_genx[group_col] == g]

        for q in questions_full:

            va = pd.to_numeric(a_nonx[q], errors="coerce").dropna()
            vx = pd.to_numeric(a_genx[q], errors="coerce").dropna()

            # t-test
            if len(va) > 2 and len(vx) > 2:
                _, p = ttest_ind(vx, va, equal_var=False, nan_policy="omit")
                pvals.loc[g, q] = p

            # Cohen’s d
            if len(va) > 1 and len(vx) > 1:
                pooled = np.sqrt(
                    ((len(va)-1)*va.var(ddof=1) + (len(vx)-1)*vx.var(ddof=1)) /
                    (len(va) + len(vx) - 2)
                )
                d = (vx.mean() - va.mean()) / pooled if pooled > 0 else np.nan
            else:
                d = np.nan

            d_matrix.loc[g, q] = d

    # FDR correctie
    flat = pvals.values.ravel()
    mask = ~np.isnan(flat)
    corr = np.full_like(flat, np.nan)
    if mask.sum() > 0:
        _, fdr_vals, _, _ = multipletests(flat[mask], method="fdr_bh", alpha=0.05)
        corr[mask] = fdr_vals
    sig = np.nan_to_num(corr.reshape(pvals.shape) < 0.05)

    # Δ % schaal
    delta_disp = pd.DataFrame((delta.values / 4.0) * 100.0,
                              index=group_order,
                              columns=questions_full)

    return delta_disp, sig, group_order, d_matrix


import pandas as pd


def bouw_effect_summary(
        delta: pd.DataFrame,
        sig: pd.DataFrame | np.ndarray,
        effect_sizes: pd.DataFrame,
        groups: list | pd.Index,
        questions: list | pd.Index,
        top_n: int = 10,
        filter_questions: list | None = None,
        sig_name: str = "Significant",
        delta_name: str = "Delta (%)",
        effect_name: str = "Effect Size (d)",
) -> pd.DataFrame:
    """
    Bouwt een gesorteerde samenvattingstabel met delta, significantie en effectgrootte.

    Parameters
    ----------
    delta : pd.DataFrame
        DataFrame met delta-waarden, index = groepen, columns = vragen.
    sig : pd.DataFrame of np.ndarray
        Significantiematrix (True/False of p<cutoff). Wordt in DataFrame gezet
        met dezelfde shape als delta (via groups & questions).
    effect_sizes : pd.DataFrame
        DataFrame met effect sizes (bijv. Cohen's d), zelfde index/columns als delta.
    groups : list/pd.Index
        Rijlabels (groepen) voor de significantiematrix.
    questions : list/pd.Index
        Kolomlabels (vragen) voor de significantiematrix.
    top_n : int, default 10
        Aantal rijen om terug te geven na sorteren.
    filter_questions : list, optional
        Beperk de output tot deze subset van vragen (bijv. top5_questions).
    sig_name, delta_name, effect_name : str
        Kolomnamen in de output.

    Returns
    -------
    pd.DataFrame
        Lange, gesorteerde tabel met MultiIndex (groep, vraag).
    """

    # 1. Significance matrix netjes in DataFrame
    sig_df = pd.DataFrame(sig, index=groups, columns=questions)

    # 2. Lange summary opbouwen
    summary = (
        delta.stack().rename(delta_name).to_frame()
        .join(sig_df.stack().rename(sig_name))
        .join(effect_sizes.stack().rename(effect_name))
    )

    # 3. Optioneel filteren op subset vragen
    if filter_questions is not None:
        summary = summary[
            summary.index.get_level_values(1).isin(filter_questions)
        ]

    # 4. Sorteren op absolute effect size
    summary_sorted = summary.reindex(
        summary[effect_name].abs().sort_values(ascending=False).index
    )

    # 5. Top N teruggeven
    if top_n is not None:
        summary_sorted = summary_sorted.head(top_n)

    return summary_sorted

import pandas as pd
import statsmodels.formula.api as smf

def compute_enps_driver_table(df_all, questions_full, labels_full,
                              enps_key="Hoe waarschijnlijk is het dat je PostNL als werkgever zou aanbevelen bij anderen?",
                              alpha=0.05):
    """
    Bouwt een 'driveranalyse'-tabel voor eNPS.

    Parameters
    ----------
    df_all : pd.DataFrame
        Gecombineerde dataset (GenX en niet-GenX).
    questions_full : list of str
        Volledige kolomnamen van de themavragen, incl. 'Betrokkenheid (gem)' en eNPS.
    labels_full : list of str
        Korte labels in dezelfde volgorde als questions_full.
    enps_key : str
        Kolomnaam van de eNPS-vraag.
    alpha : float
        Significantieniveau (default 0.05).

    Returns
    -------
    driver_eNPS : pd.DataFrame
        Tabel met per predictor:
        - originele vraagnaam
        - kort label
        - regressiecoëfficiënt
        - absolute coëfficiënt
        - p-waarde
        - Significant (True/False)
    """

    # 1. Vragen uitsluiten die niet als predictor mogen
    exclude = ["Betrokkenheid (gem)", enps_key]

    # 2. Predictors = alle themavragen behalve eNPS en Betrokkenheid (gem)
    predictors = [q for q in questions_full if q not in exclude]

    # 3. Formule bouwen
    formula = f"Q('{enps_key}') ~ " + " + ".join([f"Q('{p}')" for p in predictors])

    # 4. Regressie draaien
    model = smf.ols(formula, data=df_all).fit()

    # 5. Coefs & p-waardes (zonder intercept)
    params = model.params.drop("Intercept")
    pvals = model.pvalues.drop("Intercept")

    # 6. Map van lange naar korte labels
    label_map = dict(zip(questions_full, labels_full))

    # Helper om de kolomnaam Q('…') weer schoon te maken
    def _strip_q(name: str) -> str:
        if name.startswith("Q('") and name.endswith("')"):
            return name[3:-2]
        return name

    clean_names = [_strip_q(v) for v in params.index]

    driver_eNPS = pd.DataFrame({
        "Vraag": clean_names,
        "Thema (kort)": [label_map.get(v, v) for v in clean_names],
        "Coef": params.values,
        "AbsCoef": params.abs().values,
        "p-value": pvals.values,
        "Significant": pvals.values < alpha,
    })

    # Sorteren op grootte van het effect
    driver_eNPS = driver_eNPS.sort_values("AbsCoef", ascending=False).reset_index(drop=True)

    return driver_eNPS
