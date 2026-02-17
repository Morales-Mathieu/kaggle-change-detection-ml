from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from sklearn.preprocessing import MultiLabelBinarizer


def fix_date_column_names(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    rename_map = {}

    # Renommage des dates
    for i in range(5):
        old_col = f"date{i}"
        new_col = f"date{i+1}"
        assert old_col in df.columns
        if old_col in df.columns:
            rename_map[old_col] = new_col

    # Renommage des statuts
    for i in range(5):
        old_col = f"change_status_date{i}"
        new_col = f"change_status_date{i+1}"
        assert old_col in df.columns
        if old_col in df.columns:
            rename_map[old_col] = new_col

    return df.rename(columns=rename_map)


# =========================
# Missing values strategies
# =========================

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd


@dataclass(frozen=True)
class MissingConfig:
    date_cols: List[str]  # ex: ["date1", ..., "date5"]
    status_cols: List[str]  # ex: ["change_status_date1", ..., "change_status_date5"]
    numeric_cols: Optional[List[str]] = None  # si None -> auto détecté
    add_indicators: bool = True  # ajoute has_missing_date / nb_missing_dates
    # Valeur sentinelle pour remplacer les dates manquantes (évite les NaN qui font crash)
    date_sentinel: str = "1900-01-01"


def _ensure_cols_exist(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour {name}: {missing}")


def add_missing_date_indicators(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
    """Ajoute des indicateurs liés aux dates manquantes."""
    _ensure_cols_exist(df, date_cols, "date_cols")
    out = df.copy()
    out["has_missing_date"] = out[date_cols].isna().any(axis=1)
    out["nb_missing_dates"] = out[date_cols].isna().sum(axis=1)
    out["n_dates_valid"] = out[date_cols].notna().sum(axis=1)
    return out


def drop_rows_with_missing_dates(
    df: pd.DataFrame, date_cols: List[str]
) -> pd.DataFrame:
    """Supprime les lignes ayant au moins une date manquante."""
    _ensure_cols_exist(df, date_cols, "date_cols")
    return df.loc[~df[date_cols].isna().any(axis=1)].copy()


def _fit_impute_params(
    train_df: pd.DataFrame,
    numeric_cols: List[str],
    cat_cols: List[str],
) -> Tuple[pd.Series, Dict[str, object]]:
    """
    Calcule paramètres d'imputation à partir d'un dataframe (typiquement le train).
    - numériques: médiane
    - catégorielles: mode (ou 'UNKNOWN' si aucun mode)
    """
    _ensure_cols_exist(train_df, numeric_cols, "numeric_cols")
    _ensure_cols_exist(train_df, cat_cols, "cat_cols")

    medians = train_df[numeric_cols].median(numeric_only=True)

    modes: Dict[str, object] = {}
    for c in cat_cols:
        s = train_df[c]
        mode = s.mode(dropna=True)
        modes[c] = mode.iloc[0] if len(mode) > 0 else "UNKNOWN"
    return medians, modes


def _apply_impute_params(
    df: pd.DataFrame,
    numeric_cols: List[str],
    cat_cols: List[str],
    medians: pd.Series,
    modes: Dict[str, object],
) -> pd.DataFrame:
    """Applique une imputation (médianes/modes) sur un df."""
    out = df.copy()

    # Numériques
    for c in numeric_cols:
        if c in out.columns:
            out[c] = out[c].fillna(medians.get(c, out[c].median()))

    # Catégorielles
    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].fillna(modes.get(c, "UNKNOWN"))

    return out


def _fill_missing_dates_with_sentinel(
    df: pd.DataFrame, date_cols: List[str], sentinel: str
) -> pd.DataFrame:
    """
    Remplace les dates manquantes par une valeur sentinelle afin d'éviter tout NaN.
    On n'interprète pas la sentinelle comme une vraie date : les indicateurs portent l'information de manque.
    """
    _ensure_cols_exist(df, date_cols, "date_cols")
    out = df.copy()
    for c in date_cols:
        out[c] = out[c].fillna(sentinel)
    return out


def build_missing_strategies(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: MissingConfig,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Construit 4 stratégies (train, test) sans NaN en sortie, prêtes pour la modélisation.

    Stratégies retournées:
      - S1_trainfit_keep
      - S2_trainfit_droptrain
      - S3_indep_keep
      - S4_indep_droptrain_classmates
    """
    _ensure_cols_exist(train_df, cfg.date_cols, "date_cols(train)")
    _ensure_cols_exist(test_df, cfg.date_cols, "date_cols(test)")
    _ensure_cols_exist(train_df, cfg.status_cols, "status_cols(train)")
    _ensure_cols_exist(test_df, cfg.status_cols, "status_cols(test)")

    # Numeric cols auto
    if cfg.numeric_cols is None:
        numeric_cols = train_df.select_dtypes(include=["number"]).columns.tolist()
        if "change_type" in numeric_cols:
            numeric_cols.remove("change_type")
    else:
        numeric_cols = cfg.numeric_cols

    strategies: Dict[str, Dict[str, pd.DataFrame]] = {}

    # ================
    # Strategy S1: Train-fit, keep all rows
    # ================
    tr = train_df.copy()
    te = test_df.copy()

    med_tr, mode_tr = _fit_impute_params(tr, numeric_cols, cfg.status_cols)
    tr = _apply_impute_params(tr, numeric_cols, cfg.status_cols, med_tr, mode_tr)
    te = _apply_impute_params(te, numeric_cols, cfg.status_cols, med_tr, mode_tr)

    tr = _fill_missing_dates_with_sentinel(tr, cfg.date_cols, cfg.date_sentinel)
    te = _fill_missing_dates_with_sentinel(te, cfg.date_cols, cfg.date_sentinel)

    if cfg.add_indicators:
        tr = add_missing_date_indicators(tr, cfg.date_cols)
        te = add_missing_date_indicators(te, cfg.date_cols)

    strategies["S1_trainfit_keep"] = {"train": tr, "test": te}

    # ================
    # Strategy S2: Train-fit, drop train missing dates
    # ================
    tr = drop_rows_with_missing_dates(train_df, cfg.date_cols)
    te = test_df.copy()

    med_tr, mode_tr = _fit_impute_params(tr, numeric_cols, cfg.status_cols)
    tr = _apply_impute_params(tr, numeric_cols, cfg.status_cols, med_tr, mode_tr)
    te = _apply_impute_params(te, numeric_cols, cfg.status_cols, med_tr, mode_tr)

    # Dates: train complet sur dates, test peut en manquer → sentinelle
    tr = _fill_missing_dates_with_sentinel(tr, cfg.date_cols, cfg.date_sentinel)
    te = _fill_missing_dates_with_sentinel(te, cfg.date_cols, cfg.date_sentinel)

    if cfg.add_indicators:
        tr = add_missing_date_indicators(tr, cfg.date_cols)
        te = add_missing_date_indicators(te, cfg.date_cols)

    strategies["S2_trainfit_droptrain"] = {"train": tr, "test": te}

    # ================
    # Strategy S3: Independent imputation (train params for train, test params for test), keep all rows
    # ================
    tr = train_df.copy()
    te = test_df.copy()

    med_tr, mode_tr = _fit_impute_params(tr, numeric_cols, cfg.status_cols)
    tr = _apply_impute_params(tr, numeric_cols, cfg.status_cols, med_tr, mode_tr)

    med_te, mode_te = _fit_impute_params(te, numeric_cols, cfg.status_cols)
    te = _apply_impute_params(te, numeric_cols, cfg.status_cols, med_te, mode_te)

    tr = _fill_missing_dates_with_sentinel(tr, cfg.date_cols, cfg.date_sentinel)
    te = _fill_missing_dates_with_sentinel(te, cfg.date_cols, cfg.date_sentinel)

    if cfg.add_indicators:
        tr = add_missing_date_indicators(tr, cfg.date_cols)
        te = add_missing_date_indicators(te, cfg.date_cols)

    strategies["S3_indep_keep"] = {"train": tr, "test": te}

    # ================
    # Strategy S4: Drop train missing dates + independent test imputation (closest to classmates)
    # ================
    tr = drop_rows_with_missing_dates(train_df, cfg.date_cols)
    te = test_df.copy()

    # (train can be imputed with its own stats to avoid any remaining NaN)
    med_tr, mode_tr = _fit_impute_params(tr, numeric_cols, cfg.status_cols)
    tr = _apply_impute_params(tr, numeric_cols, cfg.status_cols, med_tr, mode_tr)

    # test imputed with its own stats (as in classmates approach)
    med_te, mode_te = _fit_impute_params(te, numeric_cols, cfg.status_cols)
    te = _apply_impute_params(te, numeric_cols, cfg.status_cols, med_te, mode_te)

    tr = _fill_missing_dates_with_sentinel(tr, cfg.date_cols, cfg.date_sentinel)
    te = _fill_missing_dates_with_sentinel(te, cfg.date_cols, cfg.date_sentinel)

    if cfg.add_indicators:
        tr = add_missing_date_indicators(tr, cfg.date_cols)
        te = add_missing_date_indicators(te, cfg.date_cols)

    strategies["S4_indep_droptrain_classmates"] = {"train": tr, "test": te}

    return strategies


def split_multi_tokens(x: object) -> List[str]:
    """
    Transforme une cellule type "River, Hills" en liste de tokens nettoyés.
    - gère NaN
    - strip
    - enlève tokens vides
    - enlève tokens très courts ('A', 'N')
    """
    if pd.isna(x):
        return []
    tokens = [t.strip() for t in str(x).split(",")]
    tokens = [t for t in tokens if t]
    tokens = [t for t in tokens if len(t) > 1]
    return tokens


def multihot_encode_column(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    col: str,
    prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, MultiLabelBinarizer]:
    """
    Encodage multi-hot d'une colonne multi-valuée (séparée par virgules).
    Fit sur le train, transform sur train+test pour garantir les mêmes colonnes.
    """
    if col not in train_df.columns or col not in test_df.columns:
        raise ValueError(f"Colonne '{col}' absente du train ou du test.")

    mlb = MultiLabelBinarizer()

    train_tokens = train_df[col].apply(split_multi_tokens)
    test_tokens = test_df[col].apply(split_multi_tokens)

    train_ohe = pd.DataFrame(
        mlb.fit_transform(train_tokens),
        columns=[f"{prefix}__{c}" for c in mlb.classes_],
        index=train_df.index,
    )
    test_ohe = pd.DataFrame(
        mlb.transform(test_tokens),
        columns=[f"{prefix}__{c}" for c in mlb.classes_],
        index=test_df.index,
    )

    train_out = pd.concat([train_df.drop(columns=[col]), train_ohe], axis=1)
    test_out = pd.concat([test_df.drop(columns=[col]), test_ohe], axis=1)

    return train_out, test_out, mlb


def reorder_temporal_blocks(
    df: pd.DataFrame,
    n_dates: int = 5,
    date_prefix: str = "date",
    status_prefix: str = "change_status_date",
    img_prefix: str = "img_",
    date_format: str | None = None,
) -> pd.DataFrame:
    """
    Réordonne (par ligne) les blocs temporels date_i en ordre croissant, et applique la même permutation à:
    - status_i
    - toutes les colonnes image/statistiques qui finissent par _date{i}

    Notes:
    - Fonction conçue pour être robuste: elle détecte automatiquement les colonnes img_*_date{i}.
    - Les dates manquantes (NaN) sont conservées et placées en fin d’ordre (via tri avec NaT).
    """
    out = df.copy()

    # Colonnes dates / statuts attendues
    date_cols = [f"{date_prefix}{i}" for i in range(1, n_dates + 1)]
    status_cols = [f"{status_prefix}{i}" for i in range(1, n_dates + 1)]

    for c in date_cols:
        if c not in out.columns:
            raise ValueError(f"Colonne manquante: {c}")
    for c in status_cols:
        if c not in out.columns:
            raise ValueError(f"Colonne manquante: {c}")

    # Colonnes image liées aux dates : toutes celles qui se terminent par _date{i}
    img_cols_by_i: Dict[int, List[str]] = {}
    for i in range(1, n_dates + 1):
        suffix = f"_date{i}"
        cols_i = [
            c for c in out.columns if c.startswith(img_prefix) and c.endswith(suffix)
        ]
        img_cols_by_i[i] = cols_i

    # Conversion dates -> datetime (si format fourni on l'utilise, sinon pandas infère)
    if date_format is None:
        dates_dt = out[date_cols].apply(
            lambda s: pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")
        )

    else:
        dates_dt = out[date_cols].apply(
            lambda s: pd.to_datetime(s, format=date_format, errors="coerce")
        )

    # Ordre trié par ligne (NaT à la fin)
    order = np.argsort(dates_dt.to_numpy(dtype="datetime64[ns]"), axis=1)

    # Helper: réordonner un bloc de colonnes (par ligne) suivant order
    def _reorder_block(block_cols: List[str]) -> pd.DataFrame:
        arr = out[block_cols].to_numpy()
        # réordonner chaque ligne selon "order"
        new_arr = np.take_along_axis(arr, order, axis=1)
        return pd.DataFrame(new_arr, columns=block_cols, index=out.index)

    # 1) Réordonner dates (on réécrit dans les mêmes colonnes date1..date5)
    out[date_cols] = _reorder_block(date_cols)

    # 2) Réordonner statuts
    out[status_cols] = _reorder_block(status_cols)

    # 3) Réordonner les features image : pour chaque type de feature, on doit permuter date1..date5
    # On le fait en reconstruisant un tableau par "feature base" (sans le suffix _date{i})
    # Exemple: img_red_mean_date1..5 => base img_red_mean_
    # Pour faire simple et robuste: on regroupe par "préfixe sans le suffix date{i}"
    # puis on réordonne ce groupe.
    # On utilise une extraction de base via rsplit("_date", 1).
    base_to_cols: Dict[str, List[str]] = {}
    for i in range(1, n_dates + 1):
        for c in img_cols_by_i[i]:
            base = c.rsplit("_date", 1)[0]  # ex: img_red_mean
            base_to_cols.setdefault(base, [])
            base_to_cols[base].append(c)

    # Chaque base doit avoir n_dates colonnes; on les trie dans l'ordre date1..dateN
    for base, cols in base_to_cols.items():
        cols_sorted = sorted(cols, key=lambda x: int(x.rsplit("_date", 1)[1]))
        out[cols_sorted] = _reorder_block(cols_sorted)

    return out


"""
feature Eng
"""


def process_dates(df):
    date_cols = [f"date{i}" for i in range(1, 6)]

    # Define the expected date format
    date_format = "%d-%m-%Y"

    # 1. Convert date columns to datetime
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format=date_format, errors="coerce")

    # 2. Compute delta features between consecutive dates (in days)
    for i in range(1, len(date_cols)):
        delta_col = f"delta{i}"
        df[delta_col] = (df[f"date{i+1}"] - df[f"date{i}"]).dt.days

    return df


def add_advanced_color_features(df, include_color_features=True):
    if not include_color_features:
        return df

    epsilon = 1e-8
    dates = range(1, 6)

    for d in dates:

        # -------------------------
        # Basic aggregates
        # -------------------------
        r = df[f"img_red_mean_date{d}"]
        g = df[f"img_green_mean_date{d}"]
        b = df[f"img_blue_mean_date{d}"]

        r_std = df[f"img_red_std_date{d}"]
        g_std = df[f"img_green_std_date{d}"]
        b_std = df[f"img_blue_std_date{d}"]

        total = r + g + b
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)

        # -------------------------
        # Vegetation & color indices
        # -------------------------
        df[f"ndgr_date{d}"] = (g - r) / (g + r + epsilon)
        df[f"green_dominance_date{d}"] = g / (r + b + epsilon)
        df[f"norm_green_date{d}"] = g / (total + epsilon)
        df[f"excess_green_date{d}"] = 2 * g - r - b

        # -------------------------
        # Color structure
        # -------------------------
        df[f"contrast_date{d}"] = max_rgb - min_rgb
        df[f"saturation_date{d}"] = (max_rgb - min_rgb) / (max_rgb + epsilon)
        df[f"luminosity_date{d}"] = total / 3.0

        # -------------------------
        # Variances & texture
        # -------------------------
        df[f"var_red_date{d}"] = r_std**2
        df[f"var_green_date{d}"] = g_std**2
        df[f"var_blue_date{d}"] = b_std**2

        df[f"variance_total_date{d}"] = (
            df[f"var_red_date{d}"] + df[f"var_green_date{d}"] + df[f"var_blue_date{d}"]
        ) / 3.0

        df[f"green_variability_date{d}"] = g_std / (g + epsilon)

        df[f"std_ratio_red_green_date{d}"] = r_std / (g_std + epsilon)
        df[f"std_ratio_blue_green_date{d}"] = b_std / (g_std + epsilon)

        df[f"color_dispersion_date{d}"] = df[f"variance_total_date{d}"] / (
            total + epsilon
        )

    # =========================================================
    # TEMPORAL FEATURES (CRUCIAL FOR CHANGE DETECTION)
    # =========================================================

    green_means = [f"img_green_mean_date{i}" for i in dates]
    green_stds = [f"img_green_std_date{i}" for i in dates]
    exg_cols = [f"excess_green_date{i}" for i in dates]

    # Range across dates
    df["green_range"] = df[green_means].max(axis=1) - df[green_means].min(axis=1)
    df["green_std_range"] = df[green_stds].max(axis=1) - df[green_stds].min(axis=1)

    df["exg_range"] = df[exg_cols].max(axis=1) - df[exg_cols].min(axis=1)

    # Mean & std across time
    df["green_mean_over_time"] = df[green_means].mean(axis=1)
    df["green_std_over_time"] = df[green_means].std(axis=1)

    # Simple temporal trend (difference first-last)
    df["green_trend"] = df["img_green_mean_date5"] - df["img_green_mean_date1"]

    df["exg_trend"] = df["excess_green_date5"] - df["excess_green_date1"]

    # Consecutive deltas
    for d in range(1, 5):
        df[f"delta_green_mean_{d}_{d+1}"] = (
            df[f"img_green_mean_date{d+1}"] - df[f"img_green_mean_date{d}"]
        )

        df[f"delta_exg_{d}_{d+1}"] = (
            df[f"excess_green_date{d+1}"] - df[f"excess_green_date{d}"]
        )

        df[f"rel_change_green_{d}_{d+1}"] = df[f"delta_green_mean_{d}_{d+1}"] / (
            df[f"img_green_mean_date{d}"] + epsilon
        )

    # =========================================================
    # INTERACTION FEATURES
    # =========================================================

    df["green_intensity_x_variability"] = df["green_range"] * df["green_std_range"]

    df["volatility_score"] = df["green_std_over_time"] * df["green_range"]

    df["structural_change_score"] = df["exg_range"] * df["green_std_over_time"]

    return df


def add_geometry_features(df):

    if "geometry" not in df.columns:
        return df

    def extract_features(geom):

        if geom is None:
            return pd.Series(
                {
                    "bbox_area": np.nan,
                    "bbox_perimeter": np.nan,
                    "bbox_length_width_ratio": np.nan,
                    "bbox_min_side": np.nan,
                    "base_area": np.nan,
                    "base_perimeter": np.nan,
                    "num_vertices": np.nan,
                    "elongation_rotated": np.nan,
                    "perimeter_area_ratio": np.nan,
                    "thinness": np.nan,
                }
            )

        xmin, ymin, xmax, ymax = geom.bounds
        width = xmax - xmin
        height = ymax - ymin

        bbox_area = width * height
        bbox_perimeter = 2 * (width + height)

        longer = max(width, height)
        shorter = min(width, height) if min(width, height) > 0 else np.nan

        bbox_length_width_ratio = (
            longer / shorter if shorter and not np.isnan(shorter) else np.nan
        )

        base_area = geom.area
        base_perimeter = geom.length

        # Minimum side (useful for roads: small width)
        bbox_min_side = shorter

        # Number of vertices
        try:
            num_vertices = len(geom.exterior.coords)
        except:
            num_vertices = np.nan

        # ---------
        # Correct rotated elongation
        # ---------
        try:
            mrr = geom.minimum_rotated_rectangle
            coords = list(mrr.exterior.coords)

            edge_lengths = [
                np.linalg.norm(np.array(coords[i]) - np.array(coords[i + 1]))
                for i in range(4)
            ]

            edge_lengths = sorted(edge_lengths)
            width_r = edge_lengths[0]
            height_r = edge_lengths[2]

            elongation_rotated = height_r / width_r if width_r > 0 else np.nan

        except:
            elongation_rotated = np.nan

        # ---------
        # NEW STRONG STRUCTURAL FEATURES
        # ---------

        # High for thin shapes (roads)
        perimeter_area_ratio = (
            (base_perimeter**2) / (base_area + 1e-9) if base_area > 0 else np.nan
        )

        # Thinness: small for roads
        estimated_length = base_perimeter / 2
        thinness = (
            base_area / (estimated_length**2 + 1e-9) if estimated_length > 0 else np.nan
        )

        return pd.Series(
            {
                "bbox_area": bbox_area,
                "bbox_perimeter": bbox_perimeter,
                "bbox_length_width_ratio": bbox_length_width_ratio,
                "bbox_min_side": bbox_min_side,
                "base_area": base_area,
                "base_perimeter": base_perimeter,
                "num_vertices": num_vertices,
                "elongation_rotated": elongation_rotated,
                "perimeter_area_ratio": perimeter_area_ratio,
                "thinness": thinness,
            }
        )

    geom_features = df["geometry"].apply(extract_features)
    df = pd.concat([df, geom_features], axis=1)

    # ---------
    # Global vectorized features
    # ---------

    # Area percentile (mega projects obvious)
    df["area_rank"] = df["base_area"].rank(pct=True)

    # Explicit mega flag
    df["is_mega_area"] = (df["area_rank"] > 0.95).astype(int)

    # Explicit very elongated flag
    df["is_very_elongated"] = (df["bbox_length_width_ratio"] > 6).astype(int)

    df = df.drop(columns=["geometry"])

    return df


def add_adv_feature(df):
    status_cols = [f"change_status_date{i}" for i in range(1, 6)]

    status_map = {
        "Greenland": 1,
        "Land Cleared": 2,
        "Construction Started": 3,
        "Construction Midway": 4,
        "Construction Done": 5,
        "Operational": 5,
        "Prior Construction": 5,
        "Excavation": 1,
        "Materials Dumped": 3,
        "Materials Introduced": 3,
        None: 0,
    }

    # Vérification sécurité
    missing_cols = [col for col in status_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes : {missing_cols}")

    def adv(row):

        a = status_map.get(row[status_cols[0]], 0)  # first date
        b = status_map.get(row[status_cols[-1]], 0)  # last date

        # Cas spécial
        if row[status_cols[-1]] == "Materials Dumped":
            return 3

        # Si données manquantes
        if row[status_cols].isna().any():
            return 0

        max_status = 0
        min_status = 5

        for s in status_cols:
            st = status_map.get(row[s], 0)
            max_status = max(max_status, st)
            min_status = min(min_status, st)

        if max_status == 5 and (max_status - b) >= 2:
            return -1

        elif (
            (min_status >= 3 and a - b < 2)
            or (b >= 3 and max_status == b and min_status < 3)
            or ((b - min_status) > 2)
            or (max_status == b and min_status == a)
        ):
            return 1

        elif b <= 2 or (a >= 3 and a - min_status > 1):
            return -1

        else:
            return 0

    df = df.copy()
    df["adv"] = df.apply(adv, axis=1)

    return df


def add_construction_date(df):

    status_map = {
        "Greenland": 1,
        "Land Cleared": 2,
        "Construction Started": 3,
        "Construction Midway": 4,
        "Construction Done": 5,
        "Operational": 5,
        "Prior Construction": 5,
        "Excavation": 1,
        "Materials Dumped": 3,
        "Materials Introduced": 3,
        None: 0,
    }

    status_cols = [
        "change_status_date1",
        "change_status_date2",
        "change_status_date3",
        "change_status_date4",
        "change_status_date5",
    ]

    date_cols = ["date1", "date2", "date3", "date4", "date5"]

    # Safety check
    for col in status_cols + date_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    def duree(row):

        steps = [status_map.get(row[s], 0) for s in status_cols]

        m = 0  # starting index

        for i in range(1, len(steps)):

            # Same stage → move starting reference
            if steps[i] == steps[i - 1] and m == i - 1:
                m = i

            # Progression
            elif steps[i] - steps[i - 1] >= 0:

                if steps[i] == 5 or (i == len(steps) - 1 and steps[i] >= 3):

                    start_date = row[date_cols[m]]
                    end_date = row[date_cols[i]]

                    if pd.notna(start_date) and pd.notna(end_date):
                        return (end_date - start_date).days
                    else:
                        return 0

            # Regression
            else:
                return 0

        return 0

    df = df.copy()
    df["duration"] = df.apply(duree, axis=1)

    return df


def remove_date_col(df):
    date_cols = [f"date{i}" for i in range(1, 6)]
    df = df.drop(columns=date_cols)

    return df


def one_hot_encode_change_status(train_df, test_df, status_cols=None):
    """
    One-hot encodes the change status columns in train_df and test_df.

    The encoder is fit on the 'change_status_date1' column from train_df and
    applied to all change status columns (assumed to have the same set of unique values).

    Parameters:
      train_df (pd.DataFrame): Training DataFrame.
      test_df (pd.DataFrame): Test DataFrame.
      status_cols (list): List of change status column names. If None, defaults to
                          ['change_status_date1', 'change_status_date2',
                           'change_status_date3', 'change_status_date4',
                           'change_status_date5'].

    Returns:
      train_df (pd.DataFrame): Modified training DataFrame with one-hot encoded columns.
      test_df (pd.DataFrame): Modified test DataFrame with one-hot encoded columns.
    """
    if status_cols is None:
        status_cols = [f"change_status_date{i}" for i in range(1, 6)]

    # Initialize and fit the OneHotEncoder on change_status_date1 from training data
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_df[[status_cols[0]]].values)

    # Retrieve the list of unique categories learned by the encoder
    categories = encoder.categories_[0]

    # Process each status column in both train and test DataFrames
    for col in status_cols:
        # Transform the column for train and test using the same encoder
        train_encoded = encoder.transform(train_df[[col]].values)
        test_encoded = encoder.transform(test_df[[col]].values)

        # Create dummy column names (e.g., change_status_date1_Greenland, etc.)
        dummy_cols = [f"{col}_{cat}" for cat in categories]

        # Create DataFrames for the one-hot encoded features while preserving the original index
        train_dummies = pd.DataFrame(
            train_encoded, columns=dummy_cols, index=train_df.index
        )
        test_dummies = pd.DataFrame(
            test_encoded, columns=dummy_cols, index=test_df.index
        )

        # Drop the original column from both DataFrames
        train_df.drop(columns=[col], inplace=True)
        test_df.drop(columns=[col], inplace=True)

        # Concatenate the dummy DataFrames back to the original DataFrames
        train_df = pd.concat([train_df, train_dummies], axis=1)
        test_df = pd.concat([test_df, test_dummies], axis=1)

    return train_df, test_df
