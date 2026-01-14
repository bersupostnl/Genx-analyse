import numpy as np
import pandas as pd

## =====================================================
# 1. Data inlezen en voorbereiden
# =====================================================
file_path = r"C:\Users\TurkmenBersu\OneDrive - PostNL\Analyse betrokkenheidsmonitor\ruwedatawbscumq2q3.xlsx"

def load_data(file_path: str):

    """
    Laadt de Excel, fixt de headers en maakt:
    - df: volledige dataset
    - df_genx: alleen Generatie X
    - df_nonx: alle andere generaties
    - questions_full: lijst kolomnamen (incl. Betrokkenheid)
    - labels_full: korte labelnamen (voor de Y-as)
    """

    # Header-fix
    header_rows = pd.read_excel(file_path, sheet_name="Data Bewerkt", nrows=2, header=None)
    h0 = header_rows.iloc[0].tolist()
    h1 = header_rows.iloc[1].tolist()

    for i in range(36, 49):  # AK..AX
        h1[i] = h0[i]

    df = pd.read_excel(file_path, sheet_name="Data Bewerkt", header=1)
    df.columns = h1

    generation_col = "Crossing: Generatie"

    # Vraagkolommen
    questions_orig = list(df.columns[36:49])
    questions_short = [
        "Gevoel te passen","Energie","Plezier","Trots",
        "Informatie toekomstplannen","Bijdragen aan toekomstplannen",
        "Waardering voor de bijdrage","Werkdruk (goed)",
        "Klantgerichtheid","Luisteren",
        "Ideeën en voorstellen gebruiken","Uitleg keuzes","eNPS (aanbeveling)"
    ]

    for q in questions_orig:
        df[q] = pd.to_numeric(df[q], errors="coerce")

    # Draai 1-5 schaal om → 1 slecht, 5 goed
    for q in questions_orig:
        df[q] = 6 - df[q]

    # Dienstjaren samenvatten
    dienstjaren_mapping = {
        "Tot 2 jaar": "<2", "0-2": "<2",
        "2 tot 5 jaar": "2-5", "2-5": "2-5",
        "5 tot 10 jaar": "5-10", "5-10": "5-10",
        "10 tot 20 jaar": "10-20", "10-20": "10-20",
        "20 tot 30 jaar": "20-30", "20-30": "20-30",
        "30 tot 40 jaar": "30-40", "30-40": "30-40",
        "Vanaf 40 jaar": "40+", "40+": "40+",
    }

    df["Dienstjarengroep_samengevat"] = (
        df["Dienstjarengroep"].replace(dienstjaren_mapping).fillna(df["Dienstjarengroep"])
    )

    valid = df.dropna(subset=[generation_col, "Dienstjarengroep_samengevat"])
    is_genx = valid[generation_col].astype(str).str.contains("X", case=False, na=False)

    df_genx = valid[is_genx].copy()
    df_nonx = valid[~is_genx].copy()

    # Samengestelde Betrokkenheid
    four_keys = [
        "Ik heb het gevoel dat ik bij PostNL pas",
        "Mijn werk geeft mij energie",
        "Ik heb plezier in mijn werk",
        "Ik ben trots op PostNL",
    ]

    valid["Betrokkenheid (gem)"]   = valid[four_keys].mean(axis=1, skipna=True)
    df_genx["Betrokkenheid (gem)"] = df_genx[four_keys].mean(axis=1, skipna=True)
    df_nonx["Betrokkenheid (gem)"] = df_nonx[four_keys].mean(axis=1, skipna=True)

    questions_full = ["Betrokkenheid (gem)"] + questions_orig
    labels_full    = ["Betrokkenheid"]       + questions_short

    return df, df_genx, df_nonx, questions_full, labels_full