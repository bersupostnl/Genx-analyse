import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

def run_regressies_op_vragen(
    df: pd.DataFrame,
    questions: list,
    group_var: str,
    covariates: list | None = None,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Voert OLS-regressies uit voor meerdere vragen (themes) t.o.v. een group-variabele.

    Returns
    -------
    pd.DataFrame met:
    - Thema
    - Coef_<group_var>   (β-coëfficiënt)
    - p_value_<group_var>
    - Significant        (p < alpha)
    """

    rows = []

    # RHS van formule bouwen
    if covariates:
        rhs_parts = [group_var]
        for c in covariates:
            if df[c].dtype == "object" or df[c].dtype.name == "category":
                rhs_parts.append(f"C({c})")
            else:
                rhs_parts.append(c)
        rhs = " + ".join(rhs_parts)
    else:
        rhs = group_var

    # Regressies draaien
    for q in questions:
        formula = f"Q('{q}') ~ {rhs}"
        model = smf.ols(formula, data=df).fit()

        coef = model.params.get(group_var, np.nan)
        pval = model.pvalues.get(group_var, np.nan)

        rows.append({
            "Thema": q,
            f"Coef_{group_var}": coef,
            f"p_value_{group_var}": pval,
            "Significant": (pval < alpha) if np.isfinite(pval) else False
        })

    return pd.DataFrame(rows)

def compute_theme_importance(df_genx, df_nonx, top5_questions, top_k=None):
    """
    Voert regressies uit per thema en geeft per thema de belangrijkste variabelen terug.
    Neemt GenX, Geslacht, Dienstjarengroep_samengevat, Salarisschaalgroep en
    Contracturen_per_week_groepen mee.
    """

    # 1) Combineer data
    df_all = pd.concat([
        df_genx.assign(GenX=1),
        df_nonx.assign(GenX=0)
    ])

    # 2) Regressieformule
    rhs = (
        "GenX"
        " + C(Geslacht)"
        " + C(Dienstjarengroep_samengevat)"
        " + C(Salarisschaalgroep)"
        " + C(Contracturen_per_week_groepen)"
    )

    all_results = []

    # 3) Loop over thema's
    for theme in top5_questions:
        formula = f"Q('{theme}') ~ {rhs}"
        model = smf.ols(formula, data=df_all).fit()

        params = model.params.drop("Intercept")
        pvals = model.pvalues.drop("Intercept")

        rows = []
        for var_name, coef in params.items():
            rows.append({
                "Thema": theme,
                "Variabele": var_name,
                "Coef": coef,
                "p-value": pvals[var_name],
                "Significant": pvals[var_name] < 0.05,
            })

        # sorteren op |coëfficiënt| zonder kolom op te slaan
        rows_sorted = sorted(rows, key=lambda r: abs(r["Coef"]), reverse=True)

        if top_k is None:
            all_results.extend(rows_sorted)
        else:
            all_results.extend(rows_sorted[:top_k])

    importance_df = pd.DataFrame(all_results)

    # 4) Kolom "Label" verwijderen (als die eventueel nog ergens zou ontstaan)
    if "Label" in importance_df.columns:
        importance_df = importance_df.drop(columns=["Label"])

    return importance_df

import pandas as pd

import pandas as pd

import pandas as pd

def bepaal_belangrijkste_factor_per_thema(importance_df):
    """
    Bepaalt per thema welke basisvariabele het sterkste negatieve effect heeft,
    op basis van significante en NEGATIEVE dummies.

    Returned een DataFrame met per thema:
    - Thema
    - Basisvariabele
    - MaxNegCoef  -> meest negatieve coef (dus laagste waarde)
    - AnySignificant
    """

    df = importance_df.copy()

    # 0. Converteer Significant naar booleans
    df["Significant"] = (
        df["Significant"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    # 1. Filter: alleen significante dummies
    df_sig = df[df["Significant"]].copy()

    # 2. Extra filter: alleen NEGATIEVE coefs
    df_sig = df_sig[df_sig["Coef"] < 0].copy()

    # Als er niets overblijft → lege output
    if df_sig.empty:
        return pd.DataFrame(columns=["Thema", "Basisvariabele", "MaxNegCoef", "AnySignificant"])

    # 3. Basisvariabelen extraheren
    df_sig["Basisvariabele"] = (
        df_sig["Variabele"]
        .str.replace(r"\[T.*", "", regex=True)
        .str.replace(r"C\(", "", regex=True)
        .str.replace(r"\)", "", regex=True)
    )

    # 4. Aggregatie per thema + basisvariabele
    #    Kies per variabele de meest negatieve waarde (min())
    agg = (
        df_sig.groupby(["Thema", "Basisvariabele"])
        .agg(
            MaxNegCoef=("Coef", "min"),     # meest negatieve coef
            AnySignificant=("Significant", "any")
        )
        .reset_index()
    )

    # 5. Selecteer per thema de basisvariabele met *de meest negatieve* coef
    top_factor = (
        agg.sort_values(["Thema", "MaxNegCoef"], ascending=[True, True])  # True → meest negatief eerst
           .groupby("Thema")
           .head(1)
           .reset_index(drop=True)
    )

    return top_factor

import pandas as pd
import statsmodels.formula.api as smf

def compute_driver_table(df_all, questions_full, four_keys):
    """
    Bouwt een tabel met regressiecoëfficiënten, p-waardes en significantie
    voor alle predictors die Betrokkenheid (gem) verklaren.
    """

    # Predictors = alle thema's behalve de engagement items en de target
    predictors = [q for q in questions_full
                  if q not in four_keys and q != "Betrokkenheid (gem)"]

    # Regressie-formule (alles in één model)
    rhs = " + ".join([f"Q('{p}')" for p in predictors])
    formula = f"Q('Betrokkenheid (gem)') ~ {rhs}"

    model = smf.ols(formula, data=df_all).fit()

    # Parameter output (excl. intercept)
    params = model.params.drop("Intercept")
    pvals = model.pvalues.drop("Intercept")

    rows = []
    for var, coef in params.items():
        rows.append({
            "Predictor": var,
            "Coef": coef,
            "AbsCoef": abs(coef),
            "p-value": pvals[var],
            "Significant": pvals[var] < 0.05
        })

    # Sorteren op sterkte impact
    table = pd.DataFrame(rows).sort_values("AbsCoef", ascending=False)

    return table



