from __future__ import annotations

import re
from io import BytesIO, StringIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
# SciPy / scikit-learn ship compiled DLLs. Some Windows machines (Smart App Control, WDAC, AppLocker)
# block them. The app still starts without them; only the statistics / regression / ML parts are disabled.
try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception as _scipy_err:  # noqa: BLE001
    stats, HAVE_SCIPY = None, False
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                                 mean_squared_error, precision_score, r2_score, recall_score,
                                 roc_auc_score)
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    HAVE_SKLEARN = True
except Exception as _sk_err:  # noqa: BLE001
    HAVE_SKLEARN = False

st.set_page_config(page_title="Social Media Analyzer", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); padding: 12px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = Path("best_social_media_model.joblib")
TEXT_COLS = ["content", "hashtags", "post_id", "created_at"]


def library_ok(ok: bool, what: str) -> bool:
    if not ok:
        st.error(f"{what} could not be loaded, so this section is disabled. Windows is probably blocking the "
                 "SciPy/scikit-learn DLLs (\"Application Control policy has blocked this file\"). "
                 "Try: PowerShell → `Get-ChildItem -Recurse \"$((python -c \"import site;print(site.getsitepackages()[-1])\"))\\scipy\" "
                 "| Unblock-File`, or use Python 3.12 in a fresh virtual environment, or ask your administrator.")
    return ok

# =============================== Shared helpers ==============================


def show_df(frame, **kw):
    try:
        st.dataframe(frame, width="stretch", **kw)
    except TypeError:  # older Streamlit
        st.dataframe(frame, use_container_width=True, **kw)


def wide_button(label, **kw):
    try:
        return st.button(label, width="stretch", **kw)
    except TypeError:  # older Streamlit
        return st.button(label, use_container_width=True, **kw)


def show_fig(fig):
    fig.tight_layout()
    try:
        st.pyplot(fig, width="stretch")
    except TypeError:  # older Streamlit
        st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def strength_label(r: float) -> str:
    a = abs(r)
    s = ("very weak/none" if a < 0.2 else "weak" if a < 0.4 else "moderate" if a < 0.6
         else "strong" if a < 0.8 else "very strong")
    return f"{s} {'positive' if r > 0 else 'negative' if r < 0 else ''}".strip()


def heatmap(m: pd.DataFrame, title: str, annot: bool = True, centered: bool = True):
    n = m.shape[1]
    fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(4, n * 0.6)))
    im = ax.imshow(m, aspect="auto", cmap="coolwarm", vmin=-1 if centered else None, vmax=1 if centered else None)
    ax.set_xticks(range(n), labels=m.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n), labels=m.index, fontsize=8)
    if annot and n <= 12:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{m.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    show_fig(fig)


def show_bar(series: pd.Series, title: str, x_label: str = "", y_label: str = "Count"):
    fig, ax = plt.subplots(figsize=(9, 4))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title); ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    ax.tick_params(axis="x", labelrotation=35)
    show_fig(fig)


def show_hist(series: pd.Series, title: str, x_label: str):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        st.info(f"No usable numeric values for {x_label}.")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(vals, bins=30, edgecolor="black", linewidth=0.35)
    ax.set_title(title); ax.set_xlabel(x_label); ax.set_ylabel("Frequency")
    show_fig(fig)


def numeric_cols(data: pd.DataFrame) -> list[str]:
    num = data.select_dtypes(include="number").dropna(axis=1, how="all")
    return num.columns[num.nunique() > 1].tolist()


# =============================== Data loading / cleaning =====================

ALIASES = {
    "post_id": ["post_id", "postid", "id", "tweet_id", "status_id"],
    "platform": ["platform", "social_platform", "network", "channel", "source"],
    "author": ["author", "username", "user", "account", "creator", "handle"],
    "topic": ["topic", "category", "content_category", "subject"],
    "content": ["content", "text", "caption", "message", "post_text", "tweet", "description", "title"],
    "hashtags": ["hashtags", "hashtag", "tags", "hash_tags"],
    "created_at": ["created_at", "created_date", "date", "datetime", "timestamp", "published_at", "post_date", "time"],
    "views": ["views", "view_count", "impressions", "reach"],
    "likes": ["likes", "like_count", "favorites", "favourites", "favorite_count"],
    "comments": ["comments", "comment_count", "replies", "reply_count"],
    "shares": ["shares", "share_count", "retweets", "reposts"],
    "sentiment": ["sentiment", "emotion", "polarity"],
    "viral": ["viral", "is_viral", "viral_status", "trending"],
}
METRIC_COLS = ["views", "likes", "comments", "shares"]


def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    original = list(frame.columns)
    normalized = {c: normalize_name(c) for c in original}
    rename_map, used = {}, set()
    for canonical, aliases in ALIASES.items():
        alias_set = {normalize_name(a) for a in aliases}
        for col in original:
            if col not in used and normalized[col] in alias_set:
                rename_map[col] = canonical
                used.add(col)
                break
    return frame.rename(columns=rename_map)


def find_csvs(folder: str) -> list[Path]:
    return sorted(Path(folder).rglob("*.csv"), key=lambda p: (len(p.name), p.name.lower()))


@st.cache_data(show_spinner=False)
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes), low_memory=False)


@st.cache_data(show_spinner=False)
def read_csv_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def clean_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    data = standardize_columns(raw).drop_duplicates().copy()
    for col in data.select_dtypes(include=["object", "string"]).columns:
        data[col] = data[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        data[col] = data[col].replace("", pd.NA)
    for col in METRIC_COLS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
            data[col] = data[col].clip(lower=0)
    if "created_at" in data.columns:
        data["created_at"] = pd.to_datetime(data["created_at"], errors="coerce", utc=True)
    if "sentiment" in data.columns:
        data["sentiment"] = data["sentiment"].astype("string").str.strip().str.lower()
    if "hashtags" not in data.columns and "content" in data.columns:
        data["hashtags"] = data["content"].astype("string").fillna("").map(
            lambda t: " ".join(re.findall(r"#[A-Za-z0-9_]+", t)))
    eng_cols = [c for c in ["likes", "comments", "shares"] if c in data.columns]
    if eng_cols:
        data["engagement"] = data[eng_cols].fillna(0).sum(axis=1)
    if "views" in data.columns and eng_cols:
        data["engagement_rate_pct"] = (data["engagement"] / data["views"].replace(0, np.nan) * 100).round(2)
    if "viral" in data.columns:
        data["viral_label"] = data["viral"].astype("string").str.lower().map({
            "1": "Viral", "true": "Viral", "yes": "Viral", "viral": "Viral",
            "0": "Not viral", "false": "Not viral", "no": "Not viral", "not viral": "Not viral",
        }).fillna(data["viral"].astype("string"))
    elif eng_cols and data["engagement"].notna().any():
        thr = data["engagement"].quantile(0.90)
        data["viral_label"] = data["engagement"].ge(thr).map({True: "Viral", False: "Not viral"})
    return data


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].astype("string")
    return out.to_csv(index=False).encode("utf-8")


def extract_hashtags(frame: pd.DataFrame) -> pd.Series:
    found = []
    if "hashtags" in frame.columns:
        for chunk in frame["hashtags"].dropna().astype(str):
            tags = re.findall(r"#[A-Za-z0-9_]+", chunk.lower())
            if not tags:
                tags = ["#" + t for t in re.split(r"[,;\s]+", chunk.lower()) if re.fullmatch(r"[a-z0-9_]+", t or "")]
            found.extend(tags)
    elif "content" in frame.columns:
        for chunk in frame["content"].dropna().astype(str):
            found.extend(re.findall(r"#[A-Za-z0-9_]+", chunk.lower()))
    return pd.Series(found, dtype="string").dropna()


# =============================== Statistics (Parts B & C) ====================


def relationship_section(data: pd.DataFrame, key: str):
    """Part B: covariance + correlation. Part C: Pearson / Spearman hypothesis tests."""
    names = numeric_cols(data)
    if len(names) < 2:
        st.info("At least two numeric columns are required.")
        return
    cols = st.multiselect("Numeric variables", names, default=names[:12], key=f"{key}_cols")
    if len(cols) < 2:
        st.info("Select at least two variables.")
        return
    sub = data[cols]
    method = st.radio("Correlation method", ["pearson", "spearman", "kendall"], horizontal=True, key=f"{key}_m")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Covariance matrix")
        cov = sub.cov()
        show_df(cov.round(3))
        heatmap(cov, "Covariance heatmap", annot=False, centered=False)
    with c2:
        st.subheader("Correlation matrix")
        corr = sub.corr(method=method)
        show_df(corr.round(3))
        heatmap(corr, "Correlation heatmap")
    st.caption("Covariance gives the direction of a relationship but its size depends on units. Correlation is "
               "scaled to -1..+1: |r| < 0.2 very weak, < 0.4 weak, < 0.6 moderate, < 0.8 strong, else very strong.")

    st.subheader("Hypothesis test for correlation (Pearson & Spearman)")
    if not library_ok(HAVE_SCIPY, "SciPy"):
        return
    a, b, ac = st.columns([2, 2, 1])
    x = a.selectbox("Variable X", cols, index=0, key=f"{key}_x")
    y = b.selectbox("Variable Y", cols, index=min(1, len(cols) - 1), key=f"{key}_y")
    alpha = ac.selectbox("Significance level α", [0.01, 0.05, 0.10], index=1, key=f"{key}_a")
    pair = data[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if x == y or len(pair) < 3:
        st.info("Choose two different variables with at least 3 complete observations.")
        return
    pr, pp = stats.pearsonr(pair[x], pair[y])
    sr, sp = stats.spearmanr(pair[x], pair[y])
    res = pd.DataFrame({
        "Test": ["Pearson", "Spearman"], "Coefficient": [pr, sr], "p-value": [pp, sp],
        "Strength": [strength_label(pr), strength_label(sr)],
        "Decision": [("Reject H0" if p < alpha else "Fail to reject H0") for p in (pp, sp)]})
    show_df(res.style.format({"Coefficient": "{:.4f}", "p-value": "{:.3e}"}))
    st.markdown(f"**H0:** no correlation between `{x}` and `{y}` (ρ = 0) · **H1:** ρ ≠ 0 (two-tailed)  \n"
                f"**n = {len(pair):,}** · **covariance = {pair[x].cov(pair[y]):.4f}**")
    for name, r, p in (("Pearson", pr, pp), ("Spearman", sr, sp)):
        verdict = "statistically significant" if p < alpha else "not statistically significant"
        st.write(f"• {name}: {strength_label(r)} relationship (r = {r:.3f}), {verdict} at α = {alpha}.")
    st.caption("Pearson measures linear association; Spearman is rank-based and captures monotonic "
               "relationships, so it is more robust to outliers and skewed social-media metrics.")
    fig, ax = plt.subplots(figsize=(7, 4))
    samp = pair.sample(min(len(pair), 3000), random_state=1)
    ax.scatter(samp[x], samp[y], s=10, alpha=0.5)
    if samp[x].nunique() > 1:
        m, c = np.polyfit(samp[x], samp[y], 1)
        xs = np.linspace(samp[x].min(), samp[x].max(), 100)
        ax.plot(xs, m * xs + c, color="red")
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"{x} vs {y}")
    show_fig(fig)


def chi_square_section(data: pd.DataFrame, key: str):
    if not library_ok(HAVE_SCIPY, "SciPy"):
        return
    st.caption("Chi-square test of independence between two categorical variables (numeric columns with many "
               "values are binned into quartiles first).")
    cand = [c for c in data.columns if data[c].nunique(dropna=True) >= 2 and
            (pd.api.types.is_numeric_dtype(data[c]) or data[c].nunique() <= 30)]
    if len(cand) < 2:
        st.info("Not enough suitable columns for a chi-square test.")
        return
    a, b, ac = st.columns([2, 2, 1])
    default1 = cand.index("platform") if "platform" in cand else 0
    default2 = cand.index("sentiment") if "sentiment" in cand else (1 if default1 == 0 else 0)
    v1 = a.selectbox("Variable 1", cand, index=default1, key=f"{key}_c1")
    v2 = b.selectbox("Variable 2", cand, index=default2, key=f"{key}_c2")
    alpha = ac.selectbox("α", [0.01, 0.05, 0.10], index=1, key=f"{key}_ca")
    if v1 == v2:
        st.info("Choose two different variables.")
        return

    def as_cat(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
            return pd.qcut(s, 4, duplicates="drop").astype(str)
        return s.astype(str)

    pair = data[[v1, v2]].dropna()
    table = pd.crosstab(as_cat(pair[v1]), as_cat(pair[v2]))
    if table.shape[0] < 2 or table.shape[1] < 2:
        st.info("Contingency table is too small.")
        return
    chi2, p, dof, expected = stats.chi2_contingency(table)
    cramers = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))
    st.write("Observed frequencies"); show_df(table)
    st.write("Expected frequencies")
    show_df(pd.DataFrame(expected, index=table.index, columns=table.columns).round(2))
    m = st.columns(4)
    m[0].metric("χ² statistic", f"{chi2:,.3f}"); m[1].metric("Degrees of freedom", f"{dof}")
    m[2].metric("p-value", f"{p:.3e}"); m[3].metric("Cramér's V", f"{cramers:.3f}")
    st.markdown(f"**H0:** `{v1}` and `{v2}` are independent · **H1:** they are associated.")
    if p < alpha:
        st.success(f"Reject H0 (p < {alpha}): the variables are significantly associated.")
    else:
        st.info(f"Fail to reject H0 (p ≥ {alpha}): no significant association found.")
    if (expected < 5).any():
        st.warning("Some expected counts are below 5, so the chi-square approximation may be unreliable.")


def group_test_section(data: pd.DataFrame, key: str):
    if not library_ok(HAVE_SCIPY, "SciPy"):
        return
    st.caption("Compare a numeric metric across groups: t-test / Mann-Whitney (2 groups) or ANOVA / Kruskal-Wallis (3+).")
    nums = numeric_cols(data)
    cats = [c for c in data.columns if c not in TEXT_COLS and 2 <= data[c].nunique(dropna=True) <= 10
            and not pd.api.types.is_numeric_dtype(data[c])]
    if not nums or not cats:
        st.info("Needs one numeric metric and one categorical column with 2-10 groups.")
        return
    a, b = st.columns(2)
    num = a.selectbox("Numeric metric", nums, index=nums.index("engagement") if "engagement" in nums else 0, key=f"{key}_gn")
    cat = b.selectbox("Grouping variable", cats, index=cats.index("platform") if "platform" in cats else 0, key=f"{key}_gc")
    groups = {str(k): v[num].dropna().values for k, v in data.groupby(cat) if v[num].notna().sum() >= 2}
    if len(groups) < 2:
        st.info("Need at least two groups with 2+ observations.")
        return
    vals = list(groups.values())
    if len(vals) == 2:
        t, p1 = stats.ttest_ind(*vals, equal_var=False)
        u, p2 = stats.mannwhitneyu(*vals)
        out = pd.DataFrame({"Test": ["Welch t-test", "Mann-Whitney U"], "Statistic": [t, u], "p-value": [p1, p2]})
    else:
        f, p1 = stats.f_oneway(*vals)
        h, p2 = stats.kruskal(*vals)
        out = pd.DataFrame({"Test": ["One-way ANOVA", "Kruskal-Wallis"], "Statistic": [f, h], "p-value": [p1, p2]})
    out["Decision (α=0.05)"] = np.where(out["p-value"] < 0.05, "Reject H0 (groups differ)", "Fail to reject H0")
    show_df(out.style.format({"Statistic": "{:.4f}", "p-value": "{:.3e}"}))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(vals, showfliers=False)
    ax.set_xticks(range(1, len(groups) + 1), labels=list(groups), rotation=30)
    ax.set_title(f"{num} by {cat}")
    show_fig(fig)


# =============================== Regression models ===========================


def regression_section(data: pd.DataFrame, key: str):
    if not library_ok(HAVE_SKLEARN, "scikit-learn"):
        return
    names = numeric_cols(data)
    if len(names) < 2:
        st.info("At least two numeric columns are required for regression.")
        return
    target = st.selectbox("Target (to predict)", names, key=f"{key}_t",
                          index=names.index("engagement") if "engagement" in names else len(names) - 1)
    leak = {"engagement", "engagement_rate_pct", "likes", "comments", "shares"} if target == "engagement" else set()
    pool = [c for c in names if c != target]
    feats = st.multiselect("Features", pool, default=([c for c in pool if c not in leak][:6] or pool[:6]), key=f"{key}_f")
    if target == "engagement":
        st.caption("Engagement = likes + comments + shares, so those columns are left out by default (data leakage).")
    if not feats:
        return
    test_size = st.slider("Test size", 0.1, 0.5, 0.25, 0.05, key=f"{key}_ts")
    work = data[feats + [target]].dropna()
    if len(work) < 20:
        st.warning("Need at least 20 complete rows.")
        return
    Xtr, Xte, ytr, yte = train_test_split(work[feats], work[target], test_size=test_size, random_state=42)
    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Lasso Regression": make_pipeline(StandardScaler(), Lasso(alpha=0.1, max_iter=10000)),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }
    rows, preds = [], {}
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        p = mdl.predict(Xte)
        preds[name] = p
        rows.append({"Model": name, "R²": r2_score(yte, p), "MAE": mean_absolute_error(yte, p),
                     "RMSE": float(np.sqrt(mean_squared_error(yte, p)))})
    res = pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)
    show_df(res.style.format({"R²": "{:.4f}", "MAE": "{:,.3f}", "RMSE": "{:,.3f}"}))
    best = res.loc[0, "Model"]
    st.success(f"Best regression model by R²: {best}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(yte, preds[best], s=10, alpha=0.5)
    lims = [min(yte.min(), preds[best].min()), max(yte.max(), preds[best].max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title(f"{best}: actual vs predicted")
    show_fig(fig)
    coef = pd.Series(models["Linear Regression"].steps[-1][1].coef_, index=feats).sort_values(key=abs, ascending=False)
    st.write("Linear regression coefficients (standardised features)")
    show_df(coef.rename("Coefficient").reset_index().rename(columns={"index": "Feature"}))


# =============================== EDA (steps 1-2) =============================


def dataset_section(raw: pd.DataFrame, df: pd.DataFrame):
    st.subheader("1. Dataset")
    st.write("**First 5 rows**"); show_df(raw.head())
    st.write("**Last 5 rows**"); show_df(raw.tail())
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{raw.shape[0]:,}"); c2.metric("Columns", f"{raw.shape[1]:,}")
    c3.metric("Duplicate records", f"{int(raw.duplicated().sum()):,}")
    st.write("**Column names**"); st.write(list(raw.columns))
    t1, t2, t3 = st.tabs(["Data types", "Dataset information", "Missing values"])
    with t1:
        show_df(raw.dtypes.astype(str).rename("dtype").reset_index().rename(columns={"index": "column"}))
    with t2:
        buf = StringIO(); raw.info(buf=buf); st.text(buf.getvalue())
    with t3:
        show_df(pd.DataFrame({"Missing values": raw.isna().sum(),
                              "Missing %": (raw.isna().mean() * 100).round(2)}).sort_values("Missing %", ascending=False))
    st.caption("Cleaning (applied to the analysis copy): exact duplicates removed, whitespace trimmed, blanks → "
               "missing, metrics converted to numbers and negatives clipped to zero.")


def eda_section(df: pd.DataFrame):
    st.subheader("2. Exploratory data analysis")
    nums = numeric_cols(df)
    cats = [c for c in df.columns if c not in nums and c not in ("created_at",) and not pd.api.types.is_datetime64_any_dtype(df[c])]
    small_cats = [c for c in cats if 2 <= df[c].nunique(dropna=True) <= 30]
    st.write(f"**Numerical variables ({len(nums)}):** {nums}")
    st.write(f"**Categorical / text variables ({len(cats)}):** {cats}")
    if nums:
        st.write("**Distribution plot**")
        c1, c2 = st.columns(2)
        n = c1.selectbox("Numeric variable", nums, key="eda_n")
        with c1:
            show_hist(df[n], f"Distribution of {n}", n)
        if small_cats:
            cc = c2.selectbox("Categorical variable", small_cats, key="eda_c")
            with c2:
                show_bar(df[cc].value_counts().head(15), f"Count of {cc}", cc, "Count")
            st.write("**Box plot** (numeric variable by category)")
            top = df[cc].value_counts().head(8).index
            groups = [df.loc[df[cc] == g, n].dropna() for g in top]
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.boxplot(groups, showfliers=False)
            ax.set_xticks(range(1, len(top) + 1), labels=[str(g) for g in top], rotation=30)
            ax.set_title(f"{n} by {cc}")
            show_fig(fig)
    if len(nums) >= 2:
        st.write("**Scatter plot**")
        s1, s2, s3 = st.columns(3)
        x = s1.selectbox("X", nums, key="eda_sx")
        y = s2.selectbox("Y", nums, index=1, key="eda_sy")
        colour = s3.selectbox("Colour by", ["(none)"] + small_cats, key="eda_sc")
        samp = df[[x, y] + ([] if colour == "(none)" else [colour])].dropna()
        samp = samp.sample(min(len(samp), 3000), random_state=1)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if colour == "(none)":
            ax.scatter(samp[x], samp[y], s=10, alpha=0.5)
        else:
            for g in samp[colour].value_counts().head(6).index:
                sub = samp[samp[colour] == g]
                ax.scatter(sub[x], sub[y], s=10, alpha=0.6, label=str(g))
            ax.legend(title=colour)
        ax.set_xlabel(x); ax.set_ylabel(y)
        show_fig(fig)


# =============================== ML pipeline (steps 3-14) ====================


def build_xy(data: pd.DataFrame, target: str, feats: list[str], max_rows: int):
    log = []
    d = data[feats + [target]].copy()
    d = d.dropna(subset=[target])
    dups = int(d.duplicated().sum())
    d = d.drop_duplicates()
    log.append(f"Removed {dups} duplicate records.")
    num = [c for c in feats if pd.api.types.is_numeric_dtype(d[c])]
    cat = [c for c in feats if c not in num]
    log.append(f"Numerical variables: {num}")
    log.append(f"Categorical variables: {cat}")
    miss = int(d[feats].isna().sum().sum())
    if num:
        d[num] = d[num].replace([np.inf, -np.inf], np.nan)
        d[num] = d[num].fillna(d[num].median()).fillna(0)
    for c in cat:
        d[c] = d[c].astype("object").where(d[c].notna(), "Unknown").astype(str)
    log.append(f"Missing values handled: {miss} cells (numeric → median, categorical → 'Unknown')." if miss
               else "No missing values in the selected features.")
    if len(d) > max_rows:
        d = d.sample(max_rows, random_state=42)
        log.append(f"Sampled {max_rows:,} rows to keep training fast.")
    classes = sorted(d[target].astype(str).unique())
    y = d[target].astype(str).map({c: i for i, c in enumerate(classes)}).astype(int)
    X = pd.get_dummies(d[feats], columns=cat, drop_first=True, dtype=int) if cat else d[feats].copy()
    if cat:
        log.append(f"Categorical → numerical: one-hot encoded {cat} ({X.shape[1]} model features).")
    log.append(f"Target `{target}` encoded: " + ", ".join(f"{c} = {i}" for i, c in enumerate(classes)))
    spec_num = {c: (float(d[c].median()), float(d[c].min()), float(d[c].max())) for c in num}
    spec_cat = {c: sorted(d[c].unique().tolist())[:50] for c in cat}
    return X.reset_index(drop=True), y.reset_index(drop=True), classes, spec_num, spec_cat, log


def ml_pipeline(data: pd.DataFrame):
    if not library_ok(HAVE_SKLEARN, "scikit-learn"):
        return
    st.caption("Predict a social-media outcome (default: **viral vs not viral**) from post features.")
    cands = [c for c in data.columns if c not in TEXT_COLS and 2 <= data[c].nunique(dropna=True) <= 10]
    if not cands:
        st.info("No suitable target column found (needs 2-10 distinct values, e.g. viral_label or sentiment).")
        return
    target = st.selectbox("Target variable", cands, index=cands.index("viral_label") if "viral_label" in cands else 0, key="ml_target")
    derived = target == "viral_label" and "viral" not in data.columns
    leak = {"engagement", "engagement_rate_pct", "likes", "comments", "shares", "viral", "viral_label"} if derived \
        else {"viral", "viral_label"} & {target}
    leak |= {target}
    pool = [c for c in data.columns if c not in TEXT_COLS and c != target and data[c].nunique(dropna=True) >= 2
            and (pd.api.types.is_numeric_dtype(data[c]) or data[c].nunique() <= 30)
            and not (target in ("viral", "viral_label") and c in ("viral", "viral_label"))]
    feats = st.multiselect("Features", pool, default=[c for c in pool if c not in leak][:8], key="ml_feats")
    if derived:
        st.warning("`viral_label` was derived from engagement (top 10%), so likes/comments/shares/engagement are "
                   "excluded by default to avoid data leakage. Add them only if you accept near-perfect scores.")
    if not feats:
        st.info("Select at least one feature.")
        return
    c1, c2, c3 = st.columns(3)
    test_size = c1.slider("Test size", 0.1, 0.4, 0.2, 0.05, key="ml_ts")
    seed = int(c2.number_input("Random state", 0, 9999, 42, key="ml_seed"))
    max_rows = int(c3.number_input("Max rows for training", 500, 50000, 5000, 500, key="ml_max"))

    try:
        X, y, classes, spec_num, spec_cat, log = build_xy(data, target, feats, max_rows)
    except Exception as exc:
        st.error(f"Pre-processing failed: {exc}")
        return
    if len(classes) < 2 or y.value_counts().min() < 2:
        st.warning("Each class needs at least 2 rows. Pick another target or loosen filters.")
        return

    st.subheader("3. Data pre-processing")
    for line in log:
        st.write("• " + line)
    show_df(X.head())

    full = X.copy(); full["target"] = y
    corr_full = full.corr().fillna(0)
    top = corr_full["target"].abs().sort_values(ascending=False).index[:12].tolist()
    st.subheader("4. Correlation analysis")
    heatmap(corr_full.loc[top, top], "Correlation heatmap (top features vs target)")
    st.subheader("5. Covariance analysis")
    heatmap(full[top].cov(), "Covariance heatmap (top features vs target)", annot=False, centered=False)

    st.subheader("6. Train-test split")
    strat = y if y.value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
    st.write(f"Train: **{len(Xtr):,}** rows · Test: **{len(Xte):,}** rows · stratified by `{target}`")

    st.subheader("7-11. Logistic Regression, Decision Tree, Random Forest, KNN, SVM")
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42)),
    }
    sig = (target, tuple(feats), test_size, seed, max_rows, len(data))
    binary = len(classes) == 2
    avg = "binary" if binary else "weighted"
    if st.button("Train all models", type="primary", key="ml_train"):
        results, fitted = [], {}
        cv = min(5, int(y.value_counts().min()))
        with st.spinner("Training models..."):
            for name, mdl in models.items():
                mdl.fit(Xtr, ytr)
                pred = mdl.predict(Xte)
                proba = mdl.predict_proba(Xte)
                try:
                    auc = roc_auc_score(yte, proba[:, 1]) if binary else roc_auc_score(yte, proba, multi_class="ovr", average="weighted")
                except Exception:
                    auc = np.nan
                results.append({
                    "Model": name, "Accuracy": accuracy_score(yte, pred),
                    "Precision": precision_score(yte, pred, average=avg, zero_division=0),
                    "Recall": recall_score(yte, pred, average=avg, zero_division=0),
                    "F1": f1_score(yte, pred, average=avg, zero_division=0), "ROC AUC": auc,
                    "CV Accuracy": cross_val_score(mdl, X, y, cv=cv).mean() if cv >= 2 else np.nan})
                fitted[name] = mdl
        st.session_state.update(ml_results=pd.DataFrame(results), ml_models=fitted, ml_test=(Xte, yte), ml_sig=sig)
    ss = st.session_state
    if ss.get("ml_sig") != sig:
        st.info("Click **Train all models** (settings changed or models not trained yet).")
        return
    if binary:
        st.caption(f"Positive class for precision/recall/F1: **{classes[1]}**")
    fmt = {c: "{:.4f}" for c in ss.ml_results.columns if c != "Model"}
    show_df(ss.ml_results.style.format(fmt))
    pick = st.selectbox("Confusion matrix for", list(ss.ml_models), key="ml_cm")
    Xt, yt = ss.ml_test
    cm = confusion_matrix(yt, ss.ml_models[pick].predict(Xt), labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)
    ax.set_xticks(range(len(classes)), classes, rotation=30); ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(pick)
    show_fig(fig)

    st.subheader("12. Model selection")
    metric = st.selectbox("Select best model by", ["F1", "Accuracy", "Recall", "Precision", "ROC AUC", "CV Accuracy"], key="ml_metric")
    ranked = ss.ml_results.sort_values(metric, ascending=False).reset_index(drop=True)
    show_df(ranked.style.format(fmt))
    best = ranked.loc[0, "Model"]
    st.success(f"Best model by {metric}: **{best}**")
    show_bar(ranked.set_index("Model")[metric], f"Model comparison – {metric}", "Model", metric)

    st.subheader("13. Save the best model (Joblib)")
    if st.button("💾 Save best model", key="ml_save"):
        joblib.dump({"model": ss.ml_models[best], "name": best, "target": target, "classes": classes,
                     "features": X.columns.tolist(), "spec_num": spec_num, "spec_cat": spec_cat}, MODEL_PATH)
        st.success(f"Saved {best} to {MODEL_PATH.resolve()}")
    prediction_interface()


def prediction_interface():
    st.subheader("14. Prediction interface")
    if not MODEL_PATH.exists():
        st.info("Save a model above to enable predictions.")
        return
    b = joblib.load(MODEL_PATH)
    st.download_button("Download saved model", MODEL_PATH.read_bytes(), MODEL_PATH.name, key="ml_dl")
    st.write(f"Model: **{b['name']}** · predicts `{b['target']}` → {b['classes']}")
    cols = st.columns(3)
    vals, i = {}, 0
    for c, (med, lo, hi) in b["spec_num"].items():
        span = max(hi - lo, 1.0)
        vals[c] = cols[i % 3].number_input(c, min_value=float(lo - span), max_value=float(hi + span * 2),
                                           value=float(med), key=f"pi_{c}")
        i += 1
    for c, options in b["spec_cat"].items():
        vals[c] = cols[i % 3].selectbox(c, options, key=f"pi_{c}")
        i += 1
    if st.button("Predict", type="primary", key="ml_pred"):
        row = {f: 0 for f in b["features"]}
        for c in b["spec_num"]:
            row[c] = vals[c]
        for c in b["spec_cat"]:
            key = f"{c}_{vals[c]}"
            if key in row:
                row[key] = 1
        frame = pd.DataFrame([row])[b["features"]]
        pred = int(b["model"].predict(frame)[0])
        st.success(f"Prediction: **{b['classes'][pred]}**")
        proba = b["model"].predict_proba(frame)[0]
        show_df(pd.DataFrame({"Class": b["classes"], "Probability": proba}).style.format({"Probability": "{:.1%}"}))


# =============================== Dashboard sections ==========================


def dashboard(filtered: pd.DataFrame):
    st.header("Key performance indicators")
    tv = filtered["views"].sum(min_count=1) if "views" in filtered.columns else None
    te = filtered["engagement"].sum(min_count=1) if "engagement" in filtered.columns else None
    ae = filtered["engagement"].mean() if "engagement" in filtered.columns else None
    ar = filtered["engagement_rate_pct"].mean() if "engagement_rate_pct" in filtered.columns else None
    k = st.columns(5)
    k[0].metric("Posts", f"{len(filtered):,}")
    k[1].metric("Total views", f"{tv:,.0f}" if pd.notna(tv) else "N/A")
    k[2].metric("Total engagement", f"{te:,.0f}" if pd.notna(te) else "N/A")
    k[3].metric("Avg. engagement/post", f"{ae:,.2f}" if pd.notna(ae) else "N/A")
    k[4].metric("Avg. engagement rate", f"{ar:.2f}%" if pd.notna(ar) else "N/A")

    st.header("Engagement analysis")
    left, right = st.columns(2)
    with left:
        metrics = [c for c in METRIC_COLS if c in filtered.columns]
        if metrics:
            m = st.selectbox("Choose metric", metrics)
            show_hist(filtered[m], f"Distribution of {m}", m)
        else:
            st.info("No views/likes/comments/shares columns were detected.")
    with right:
        if "engagement" in filtered.columns and filtered["engagement"].notna().any():
            st.subheader("Top posts by engagement")
            sort_cols = ["engagement"] + (["views"] if "views" in filtered.columns else [])
            disp = [c for c in ["post_id", "platform", "author", "topic", "content", "views", "likes", "comments",
                                "shares", "engagement", "engagement_rate_pct"] if c in filtered.columns]
            show_df(filtered.sort_values(sort_cols, ascending=False).head(10)[disp], height=350)
        else:
            st.info("Engagement ranking requires a likes, comments or shares column.")

    if "platform" in filtered.columns:
        st.subheader("Platform performance")
        agg = {("post_id" if "post_id" in filtered.columns else filtered.columns[0]): "count"}
        for m in ["views", "likes", "comments", "shares", "engagement", "engagement_rate_pct"]:
            if m in filtered.columns:
                agg[m] = "mean"
        if len(agg) > 1:
            ps = filtered.groupby("platform", dropna=False).agg(agg).rename(columns={"post_id": "post_count"})
            show_df(ps.sort_values("engagement" if "engagement" in ps.columns else ps.columns[0], ascending=False))
            opts = [c for c in ["views", "likes", "comments", "shares", "engagement"] if c in filtered.columns]
            cm = st.selectbox("Platform chart metric", opts)
            show_bar(filtered.groupby("platform")[cm].sum(min_count=1).sort_values(ascending=False),
                     f"Total {cm} by platform", "Platform", f"Total {cm}")

    st.header("Hashtag analysis")
    hashtags = extract_hashtags(filtered)
    if not hashtags.empty:
        hc = hashtags.value_counts().head(25)
        h1, h2 = st.columns(2)
        with h1:
            show_df(hc.rename_axis("Hashtag").reset_index(name="Frequency"))
        with h2:
            show_bar(hc.head(15).sort_values(), "Top hashtags", "Hashtag", "Frequency")
    else:
        st.info("No hashtags detected.")

    st.header("Sentiment analysis")
    if "sentiment" in filtered.columns and filtered["sentiment"].notna().any():
        sc = filtered["sentiment"].value_counts()
        s1, s2 = st.columns(2)
        with s1:
            show_df(sc.rename_axis("Sentiment").reset_index(name="Posts"))
        with s2:
            show_bar(sc.sort_values(), "Posts by sentiment", "Sentiment", "Posts")
        if "engagement" in filtered.columns:
            st.subheader("Average engagement by sentiment")
            show_df(filtered.groupby("sentiment")["engagement"].mean().sort_values(ascending=False)
                    .rename("Average engagement").reset_index())
    else:
        st.info("No sentiment column was detected.")

    st.header("Posting and engagement trends")
    if "created_at" in filtered.columns and filtered["created_at"].notna().any():
        freq = st.selectbox("Trend grouping", ["Daily", "Weekly", "Monthly"])
        code = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[freq]
        dated = filtered.dropna(subset=["created_at"]).set_index("created_at").sort_index()
        if "engagement" in dated.columns and dated["engagement"].notna().any():
            st.subheader("Engagement over time")
            st.line_chart(dated["engagement"].resample(code).sum())
        st.subheader("Number of posts over time")
        st.line_chart(dated.resample(code).size())
    else:
        st.info("A date/time column is needed for time-series trends.")

    st.header("Viral post insights")
    if "viral_label" in filtered.columns:
        vc = filtered["viral_label"].value_counts(dropna=False)
        v1, v2 = st.columns(2)
        with v1:
            show_df(vc.rename_axis("Viral status").reset_index(name="Posts"))
        with v2:
            show_bar(vc, "Viral vs non-viral posts", "Status", "Posts")
        if "engagement" in filtered.columns:
            show_df(filtered.groupby("viral_label", dropna=False).agg(
                posts=("engagement", "size"), average_engagement=("engagement", "mean"),
                maximum_engagement=("engagement", "max")))
    else:
        st.info("Viral classification is not available.")

    st.header("Content and author insights")
    c1, c2 = st.columns(2)
    with c1:
        if "author" in filtered.columns:
            st.subheader("Most active authors")
            show_df(filtered["author"].value_counts().head(15).rename_axis("Author").reset_index(name="Posts"))
    with c2:
        if "topic" in filtered.columns:
            st.subheader("Posts by topic")
            show_df(filtered["topic"].value_counts().head(15).rename_axis("Topic").reset_index(name="Posts"))
    if "content" in filtered.columns:
        st.subheader("Most frequent words")
        stop = set("the a an and or but if then to of in on for with at by from is are was were be been being it "
                   "this that these those i you we they he she them our your my as not do does did have has had".split())
        blob = " ".join(filtered["content"].dropna().astype(str)).lower()
        words = [w for w in re.findall(r"\b[a-z][a-z0-9_]{2,}\b", blob) if w not in stop and not w.startswith("http")]
        if words:
            show_df(pd.Series(words).value_counts().head(25).rename_axis("Word").reset_index(name="Frequency"))

    st.header("Descriptive statistics")
    num = filtered.select_dtypes(include="number")
    if not num.empty:
        show_df(num.describe().T)
        st.download_button("Download descriptive statistics", num.describe().T.to_csv().encode("utf-8"),
                           "descriptive_statistics.csv", "text/csv")


def integrations(filtered: pd.DataFrame):
    st.header("Optional PySpark analysis")
    with st.expander("Run local Spark analysis"):
        st.write("Starts Spark in local mode; a Spark cluster is not required.")
        if st.button("Run PySpark summary and word count"):
            try:
                from pyspark.sql import SparkSession, functions as F
                spark = SparkSession.builder.master("local[*]").appName("SocialMediaAnalyzer").getOrCreate()
                spark.sparkContext.setLogLevel("ERROR")
                sp = filtered.copy()
                for col in sp.columns:
                    if pd.api.types.is_datetime64_any_dtype(sp[col]):
                        sp[col] = sp[col].astype("string")
                sp = sp.astype(object).where(pd.notna(sp), None)
                sdf = spark.createDataFrame(sp)
                st.code(sdf._jdf.schema().treeString(), language="text")
                st.metric("Spark row count", f"{sdf.count():,}")
                sn = [c for c in ["views", "likes", "comments", "shares", "engagement"] if c in sdf.columns]
                if sn:
                    exprs = []
                    for col in sn:
                        exprs += [F.avg(col).alias(f"avg_{col}"), F.max(col).alias(f"max_{col}"), F.sum(col).alias(f"sum_{col}")]
                    show_df(sdf.agg(*exprs).toPandas())
                if "content" in sdf.columns:
                    wd = (sdf.select(F.explode(F.split(F.lower(F.col("content").cast("string")), r"[^a-zA-Z0-9_]+")).alias("word"))
                          .where(F.length("word") >= 3).groupBy("word").count().orderBy(F.desc("count")).limit(25))
                    st.subheader("Top words using Spark"); show_df(wd.toPandas())
                if "platform" in sdf.columns and "engagement" in sdf.columns:
                    st.subheader("Spark engagement by platform")
                    show_df(sdf.groupBy("platform").agg(
                        F.count("*").alias("post_count"), F.avg("engagement").alias("avg_engagement"),
                        F.sum("engagement").alias("total_engagement")).orderBy(F.desc("total_engagement")).toPandas())
                spark.stop()
            except Exception as exc:
                st.error(f"PySpark analysis failed: {exc}")
                st.info("Check Java/PySpark compatibility and installed dependencies.")

    st.header("Optional Cassandra storage")
    with st.expander("Save filtered records to Cassandra"):
        st.warning("Use only a Cassandra instance you control. Connection details are not saved by this app.")
        host = st.text_input("Cassandra host", value="127.0.0.1")
        keyspace = st.text_input("Keyspace", value="social_media_db")
        table = st.text_input("Table", value="social_posts")
        if st.button("Connect and save records to Cassandra"):
            try:
                if not (re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", keyspace) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table)):
                    raise ValueError("Keyspace and table names may only contain letters, digits and underscores.")
                from cassandra.cluster import Cluster
                from uuid import uuid4
                cluster = Cluster([host])
                session = cluster.connect()
                session.execute(f"CREATE KEYSPACE IF NOT EXISTS {keyspace} WITH replication = "
                                f"{{'class':'SimpleStrategy','replication_factor':1}}")
                session.set_keyspace(keyspace)
                session.execute(f"""CREATE TABLE IF NOT EXISTS {table} (
                    record_id uuid PRIMARY KEY, platform text, author text, content text, created_at text,
                    views bigint, likes bigint, comments bigint, shares bigint, engagement double, sentiment text)""")
                insert = session.prepare(f"""INSERT INTO {table}
                    (record_id, platform, author, content, created_at, views, likes, comments, shares, engagement, sentiment)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""")

                def s_int(row, col):
                    v = row.get(col)
                    try:
                        return int(v) if pd.notna(v) else None
                    except (ValueError, TypeError, OverflowError):
                        return None

                def s_str(row, col):
                    v = row.get(col)
                    return str(v) if pd.notna(v) else None

                for _, row in filtered.iterrows():
                    session.execute(insert, [
                        uuid4(), s_str(row, "platform"), s_str(row, "author"), s_str(row, "content"),
                        s_str(row, "created_at"), s_int(row, "views"), s_int(row, "likes"), s_int(row, "comments"),
                        s_int(row, "shares"),
                        float(row["engagement"]) if "engagement" in row and pd.notna(row["engagement"]) else None,
                        s_str(row, "sentiment")])
                cluster.shutdown()
                st.success(f"Saved {len(filtered):,} records to {keyspace}.{table}.")
            except Exception as exc:
                st.error(f"Cassandra save failed: {exc}")
                st.info("Install cassandra-driver and ensure Cassandra is running.")

    st.header("Export results")
    st.download_button("Download filtered and cleaned dataset (CSV)", to_csv_bytes(filtered),
                       "social_media_cleaned_analysis.csv", "text/csv", type="primary")
    st.caption("Column detection supports common aliases. For unusual column names, update ALIASES near the top of app.py.")


# =============================== Main app ====================================

st.title("📊 Social Media Analytics Dashboard")
st.write("Explore posts, engagement, sentiment, hashtags and trends, then test relationships statistically "
         "and build regression and classification models, all on your social media data.")
st.caption("Local dashboard · Kaggle dataset import · Streamlit · Optional PySpark and Cassandra integrations")

ss = st.session_state
ss.setdefault("raw_data", None)
ss.setdefault("source_label", "")

with st.sidebar:
    st.header("📥 Data source")
    source_mode = st.radio("Choose how to load data", ["Kaggle dataset", "Upload CSV", "Local CSV path"])
    kaggle_slug, uploaded, local_path = "", None, ""
    download_kaggle = load_local = False
    if source_mode == "Kaggle dataset":
        kaggle_slug = st.text_input("Kaggle dataset slug", placeholder="owner/dataset-name",
                                    help="Copy the owner/dataset-name portion from the Kaggle dataset URL.")
        download_kaggle = wide_button("Download from Kaggle", type="primary")
    elif source_mode == "Upload CSV":
        uploaded = st.file_uploader("Select CSV file", type=["csv"])
    else:
        local_path = st.text_input("CSV file path", placeholder=r"C:\data\social_media_posts.csv")
        load_local = wide_button("Load local CSV", type="primary")
    st.divider()
    st.caption("Kaggle authentication: configure kaggle.json or the supported Kaggle environment credentials.")

if source_mode == "Kaggle dataset" and download_kaggle:
    if not kaggle_slug.strip() or "/" not in kaggle_slug.strip():
        st.error("Enter a valid Kaggle slug in owner/dataset-name format.")
    else:
        try:
            with st.spinner("Downloading dataset from Kaggle..."):
                import kagglehub
                folder = kagglehub.dataset_download(kaggle_slug.strip())
                candidates = find_csvs(folder)
                if not candidates:
                    st.error("The downloaded dataset contains no CSV files. Upload a CSV instead.")
                elif len(candidates) == 1:
                    ss.raw_data = read_csv_path(str(candidates[0]))
                    ss.source_label = f"Kaggle · {kaggle_slug.strip()} · {candidates[0].name}"
                    ss.csv_options = None
                    st.success(f"Loaded {candidates[0].name}")
                else:
                    ss.csv_options = [str(p) for p in candidates]
                    ss.source_label = f"Kaggle · {kaggle_slug.strip()}"
        except Exception as exc:
            st.error(f"Kaggle download failed: {exc}")
            st.info("Check the slug, internet connection, Kaggle credentials and dataset permissions.")

if source_mode == "Kaggle dataset" and ss.get("csv_options"):
    chosen = st.selectbox("Choose CSV from downloaded dataset", ss.csv_options, format_func=lambda p: Path(p).name)
    if st.button("Load selected CSV"):
        ss.raw_data = read_csv_path(chosen)
        ss.source_label = f"{ss.source_label} · {Path(chosen).name}"

if source_mode == "Upload CSV" and uploaded is not None:
    try:
        ss.raw_data = read_csv_bytes(uploaded.getvalue())
        ss.source_label = f"Uploaded file · {uploaded.name}"
    except Exception as exc:
        st.error(f"Unable to read CSV: {exc}")

if source_mode == "Local CSV path" and load_local and local_path.strip():
    try:
        ss.raw_data = read_csv_path(local_path.strip())
        ss.source_label = f"Local file · {local_path.strip()}"
    except Exception as exc:
        st.error(f"Unable to load the path: {exc}")

if ss.raw_data is None:
    st.info("Choose a source in the sidebar to begin: download a Kaggle dataset, upload a CSV, or provide a local CSV path.")
    st.markdown("""
### Setup
1. `pip install streamlit pandas matplotlib scipy scikit-learn joblib kagglehub`
   (add `pyspark` and `cassandra-driver` only if you use those sections)
2. For Kaggle downloads, create an API token in Kaggle Account Settings and place `kaggle.json` in
   `%USERPROFILE%\\.kaggle\\kaggle.json`.
3. Run `streamlit run app.py`, then enter a dataset slug such as `owner/dataset-name`, upload a CSV, or give a local path.

**Privacy:** do not share or commit your Kaggle API token.
""")
    st.stop()

raw_df = ss.raw_data.copy()
df = clean_dataset(raw_df)
st.success(f"Dataset loaded: {ss.source_label}")

st.sidebar.subheader("Dataset filters")
filtered = df.copy()
for col, label in [("platform", "Platform"), ("sentiment", "Sentiment"), ("topic", "Topic/category")]:
    if col in filtered.columns:
        opts = sorted(filtered[col].dropna().astype(str).unique().tolist())
        sel = st.sidebar.multiselect(label, opts, default=opts)
        if sel:
            filtered = filtered[filtered[col].astype(str).isin(sel)]

tab_dash, tab_eda, tab_stats, tab_reg, tab_ml, tab_more = st.tabs([
    "📈 Dashboard", "🔎 Dataset & EDA", "📐 Correlation, Covariance & Hypothesis Tests",
    "📉 Regression Models", "🤖 Classification (ML)", "🧩 Spark · Cassandra · Export"])

with tab_dash:
    dashboard(filtered)
with tab_eda:
    dataset_section(raw_df, df)
    eda_section(filtered)
with tab_stats:
    st.header("Part B: Relationship analysis (covariance & correlation) and Part C: hypothesis testing")
    relationship_section(filtered, "stat")
    st.header("Chi-square test of independence")
    chi_square_section(filtered.drop(columns=[c for c in TEXT_COLS if c in filtered.columns]), "stat")
    st.header("Group comparison tests")
    group_test_section(filtered, "stat")
with tab_reg:
    st.header("Regression models")
    regression_section(filtered, "reg")
with tab_ml:
    st.header("Classification system for social media posts")
    ml_pipeline(filtered)
with tab_more:
    integrations(filtered)
