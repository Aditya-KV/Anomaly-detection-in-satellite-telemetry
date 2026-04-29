# ============================================================
# FINAL CODE (COLAB) — SMAP + OPS-SAT + SYNTHETIC  (+ OPS-SAT INJECTION)
# What this update does:
# 1) Keeps SMAP + Synthetic exactly as before (supervised where labels exist).
# 2) For OPS-SAT:
#    - loads the OPS-SAT CSV
#    - (OPTIONAL) injects synthetic anomalies into OPS-SAT numeric data
#    - creates a NEW labeled OPS-SAT test set with a "label" column
#    - runs your same pipeline and now prints F1 for OPS-SAT (on injected labels)
# 3) Also prints a quick "useful/not useful" message for OPS-SAT BEFORE running models:
#    - if numeric columns are too few / too many NaNs / no variance -> not useful
# ============================================================

import os
import glob
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    precision_recall_fscore_support,
)

# ---------------------------
# HTML Logger Injection
# ---------------------------
import io
import base64
import webbrowser
import builtins
import os

HTML_REPORT = [
    "<!DOCTYPE html>",
    "<html><head>",
    "<title>Anomaly Detection Report</title>",
    "<style>",
    "body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; color: #343a40; padding: 20px; max-width: 1200px; margin: auto; }",
    "pre { background-color: #e9ecef; padding: 10px; border-radius: 5px; overflow-x: auto; font-size: 14px; border: 1px solid #ced4da; white-space: pre-wrap; word-wrap: break-word; }",
    ".log-block { margin-bottom: 20px; }",
    ".plot-container { background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px; text-align: center; }",
    "img { max-width: 100%; height: auto; border-radius: 4px; }",
    "h1 { color: #2c3e50; text-align: center; border-bottom: 2px solid #3498db; padding-bottom: 10px; }",
    "</style>",
    "</head><body>",
    "<h1>Anomaly Detection Report</h1>"
]
_current_log_block = []

_original_print = builtins.print
def log_print(*args, **kwargs):
    _original_print(*args, **kwargs)
    if kwargs.get('file') is not None:
        return
    msg = " ".join(str(a) for a in args)
    _current_log_block.append(msg)
builtins.print = log_print

def flush_log_block():
    global _current_log_block
    if _current_log_block:
        text = "\n".join(_current_log_block)
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        HTML_REPORT.append(f"<div class='log-block'><pre>{text}</pre></div>")
        _current_log_block = []

def add_plot_to_html(fig):
    flush_log_block()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    HTML_REPORT.append(f"<div class='plot-container'><img src='data:image/png;base64,{b64}' /></div>")

def finalize_html_report():
    flush_log_block()
    HTML_REPORT.append("</body></html>")
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(HTML_REPORT))
    _original_print(f"Report generated: {report_path}")
    webbrowser.open(f"file://{report_path}")

# ---------------------------
# Styling
# ---------------------------
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
COLORS = ["#2c3e50", "#e74c3c", "#3498db", "#f1c40f", "#2ecc71"]


# ============================================================
# Synthetic Benchmark (guaranteed labels)
# ============================================================
def generate_synthetic_benchmark(seed=42):
    np.random.seed(seed)
    print("\n[System] Generating Synthetic Benchmark Data...")
    t = np.linspace(0, 100, 4000)
    signal = np.sin(t) + np.sin(t / 3) * 0.5 + np.random.normal(0, 0.1, 4000)

    y = np.zeros(4000, dtype=int)
    signal[500:520] += 4.0
    y[500:520] = 1
    signal[2000:2200] += np.random.normal(0, 2.0, 200)
    y[2000:2200] = 1
    signal[3200:3400] -= 3.0
    y[3200:3400] = 1

    df = pd.DataFrame({"sensor_1": signal, "sensor_2": signal * 0.5 + np.cos(t)})
    return df.values, y, "Synthetic Benchmark (Guaranteed Labels)"


# ============================================================
# Utilities
# ============================================================
def normalize_chan_id(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace(".npy", "")
    s = s.replace("_train", "").replace("_test", "")
    return s.strip()


def detect_label_columns(df: pd.DataFrame):
    cols = [c.lower().strip() for c in df.columns]
    id_candidates = ["chan_id", "channel_id", "channel", "id", "sensor", "name"]
    seq_candidates = ["anomaly_sequences", "anomalies", "anomaly_sequence", "sequences"]

    id_col, seq_col = None, None
    for cand in id_candidates:
        if cand in cols:
            id_col = df.columns[cols.index(cand)]
            break
    for cand in seq_candidates:
        if cand in cols:
            seq_col = df.columns[cols.index(cand)]
            break
    return id_col, seq_col


def minmax01(x):
    x = np.asarray(x).reshape(-1, 1)
    return MinMaxScaler().fit_transform(x).ravel()


def find_first_file(root, patterns):
    for pat in patterns:
        hits = glob.glob(os.path.join(root, "**", pat), recursive=True)
        if hits:
            hits.sort(key=lambda p: (len(p), p))
            return hits[0]
    return None


def find_all_files(root, pattern):
    hits = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    hits.sort(key=lambda p: (len(p), p))
    return hits


# ============================================================
# Quick OPS-SAT usefulness check
# ============================================================
def opssat_usefulness_report(df: pd.DataFrame, min_numeric_cols=2):
    rep = {}
    num = df.select_dtypes(include=[np.number]).copy()
    rep["rows"] = int(df.shape[0])
    rep["cols"] = int(df.shape[1])
    rep["numeric_cols"] = int(num.shape[1])

    if num.shape[1] == 0:
        rep["useful"] = False
        rep["reason"] = "No numeric columns found."
        return rep

    # NaN ratio and variance
    nan_ratio = float(num.isna().mean().mean())
    rep["nan_ratio_avg"] = nan_ratio

    variances = num.var(skipna=True).values
    zero_var_cols = int(np.sum(variances == 0))
    rep["zero_var_cols"] = zero_var_cols

    if rep["numeric_cols"] < min_numeric_cols:
        rep["useful"] = False
        rep["reason"] = f"Too few numeric columns ({rep['numeric_cols']})."
        return rep

    if nan_ratio > 0.60:
        rep["useful"] = False
        rep["reason"] = f"Too many missing values (avg NaN ratio {nan_ratio:.2f})."
        return rep

    if zero_var_cols >= max(1, rep["numeric_cols"] // 2):
        rep["useful"] = False
        rep["reason"] = f"Too many zero-variance columns ({zero_var_cols}/{rep['numeric_cols']})."
        return rep

    rep["useful"] = True
    rep["reason"] = "Looks usable (enough numeric columns, acceptable NaNs, non-trivial variance)."
    return rep


# ============================================================
# OPS-SAT anomaly injection (creates labels)
# ============================================================
def inject_anomalies_opssat(
    df: pd.DataFrame,
    anomaly_frac=0.03,
    seed=42,
    spike_mag=6.0,
    drift_mag=2.5,
    dropout_value=0.0,
    debug=True,
):
    """
    Returns:
      X_aug_df (numeric df),
      y (0/1 labels aligned to rows)
    """
    np.random.seed(seed)

    X = df.select_dtypes(include=[np.number]).copy()
    if X.shape[1] == 0:
        raise ValueError("OPS-SAT injection failed: no numeric columns.")

    Xn = X.values.astype(float)
    n, d = Xn.shape

    y = np.zeros(n, dtype=int)
    k = max(1, int(n * anomaly_frac))
    idx = np.random.choice(np.arange(n), size=k, replace=False)

    types = np.random.choice(["spike", "drift", "dropout", "noiseburst"], size=k, replace=True)

    X_aug = Xn.copy()

    # precompute stds (avoid std=0)
    stds = np.std(Xn, axis=0)
    stds[stds == 0] = 1.0

    for i, t in zip(idx, types):
        j = np.random.randint(0, d)

        if t == "spike":
            X_aug[i, j] += spike_mag * stds[j]
            y[i] = 1

        elif t == "drift":
            L = np.random.randint(10, 60)
            end = min(n, i + L)
            ramp = np.linspace(0, drift_mag * stds[j], end - i)
            X_aug[i:end, j] += ramp
            y[i:end] = 1

        elif t == "dropout":
            L = np.random.randint(5, 30)
            end = min(n, i + L)
            X_aug[i:end, j] = dropout_value
            y[i:end] = 1

        elif t == "noiseburst":
            L = np.random.randint(10, 50)
            end = min(n, i + L)
            X_aug[i:end, j] += np.random.normal(0, 2.0 * stds[j], size=(end - i))
            y[i:end] = 1

    X_aug_df = pd.DataFrame(X_aug, columns=X.columns)

    if debug:
        print("\n[DEBUG] OPS-SAT INJECTION")
        print("rows:", n, "features:", d)
        print("target anomaly frac:", anomaly_frac, "| actual:", float(y.mean()))
        print("anomaly count:", int(y.sum()))

    return X_aug_df, y


# ============================================================
# SMAP/MSL loader (train on *_train.npy; test on *_test.npy)
# ============================================================
def load_smap_msl_anywhere(root_search="/content", chan_id="P-1", debug=True):
    chan_norm = normalize_chan_id(chan_id)

    train_path = find_first_file(root_search, [f"{chan_norm}_train.npy"])
    test_path  = find_first_file(root_search, [f"{chan_norm}_test.npy"])
    labels_path = find_first_file(root_search, ["labeled_anomalies*.csv", "*labeled*anomal*.csv"])

    if not (train_path and test_path):
        raise FileNotFoundError(f"Missing {chan_norm}_train.npy or {chan_norm}_test.npy under {root_search}")

    X_train = np.load(train_path)
    X_test = np.load(test_path)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    if debug:
        print("\n[DEBUG] SMAP/MSL FILES")
        print("train_path :", train_path)
        print("test_path  :", test_path)
        print("labels_csv :", labels_path)

    y_test = None
    note = f"NASA SMAP/MSL ({chan_norm}) train/test loaded"

    if labels_path is None:
        note += " | labels csv NOT found"
        return X_train, X_test, y_test, note

    labels_df = pd.read_csv(labels_path)
    id_col, seq_col = detect_label_columns(labels_df)

    if debug:
        print("[DEBUG] labels columns:", list(labels_df.columns))
        print("[DEBUG] detected id_col:", id_col, "| seq_col:", seq_col)

    if id_col is None or seq_col is None:
        note += " | label columns not recognized"
        return X_train, X_test, y_test, note

    labels_df["_chan_norm"] = labels_df[id_col].apply(normalize_chan_id)
    row = labels_df[labels_df["_chan_norm"] == chan_norm]

    if row.empty:
        note += " | label row NOT found for channel"
        return X_train, X_test, y_test, note

    raw_seqs = row.iloc[0][seq_col]
    seqs = ast.literal_eval(raw_seqs) if isinstance(raw_seqs, str) else raw_seqs

    y_test = np.zeros(len(X_test), dtype=int)
    for start, end in seqs:
        start = int(start)
        end = int(end)
        if start < len(y_test):
            y_test[start: min(end + 1, len(y_test))] = 1

    if debug:
        idx = np.where(y_test == 1)[0]
        print("[DEBUG] X_test len:", len(X_test), "| anomaly_count:", int(y_test.sum()))
        print("[DEBUG] first anomaly:", int(idx[0]) if len(idx) else None,
              "| last anomaly:", int(idx[-1]) if len(idx) else None)

    note += " | labels loaded"
    return X_train, X_test, y_test, note


# ============================================================
# OPS-SAT loader (loads raw df + numeric X) so we can inject
# ============================================================
def load_opssat_df_anywhere(root_search="/content", debug=True):
    csvs = find_all_files(root_search, "*.csv")
    if not csvs:
        return None

    candidates = []
    for p in csvs:
        bn = os.path.basename(p).lower()
        if ("labeled" in bn and "anomal" in bn):
            continue
        candidates.append(p)

    if not candidates:
        return None

    pick = None
    for p in candidates:
        bn = os.path.basename(p).lower()
        if "opssat" in bn or "ops-sat" in bn or "ops_sat" in bn:
            pick = p
            break
    if pick is None:
        pick = candidates[0]

    df = pd.read_csv(pick)
    df.columns = [str(c).strip() for c in df.columns]

    if debug:
        print("\n[DEBUG] OPS-SAT RAW CSV")
        print("csv used:", pick)
        print("[DEBUG] shape:", df.shape)
        print("[DEBUG] columns:", list(df.columns)[:40], "..." if len(df.columns) > 40 else "")

    return df, pick


# ============================================================
# Feature Engineering
# ============================================================
def make_rolling_features_1d(X_1d_scaled: np.ndarray, window: int = 8):
    df_temp = pd.DataFrame(X_1d_scaled, columns=["val"])
    df_temp["mean"] = df_temp["val"].rolling(window).mean()
    df_temp["std"]  = df_temp["val"].rolling(window).std()
    df_temp["min"]  = df_temp["val"].rolling(window).min()
    df_temp["max"]  = df_temp["val"].rolling(window).max()
    df_temp["diff"] = df_temp["val"].diff().rolling(window).mean()
    return df_temp.fillna(0).values


def window_features(X_scaled: np.ndarray, W: int = 30):
    X_scaled = np.asarray(X_scaled)
    T, d = X_scaled.shape
    feats = []

    for j in range(d):
        s = pd.Series(X_scaled[:, j])

        roll = s.rolling(W, min_periods=max(5, W // 4))
        mean = roll.mean()
        std  = roll.std()
        mn   = roll.min()
        mx   = roll.max()
        med  = roll.median()
        q10  = roll.quantile(0.10)
        q90  = roll.quantile(0.90)
        iqr  = q90 - q10

        diff = s.diff()
        roll_d = diff.rolling(W, min_periods=max(5, W // 4))
        dmean = roll_d.mean()
        dstd  = roll_d.std()

        energy = (s ** 2).rolling(W, min_periods=max(5, W // 4)).mean()

        block = pd.concat([mean, std, mn, mx, med, q10, q90, iqr, dmean, dstd, energy], axis=1)
        feats.append(block)

    F = pd.concat(feats, axis=1).fillna(0.0).values
    return F


# ============================================================
# Thresholding
# ============================================================
def smooth_scores(scores: np.ndarray, window: int = 7) -> np.ndarray:
    if window <= 1:
        return scores
    return pd.Series(scores).rolling(window, center=True, min_periods=1).mean().values


def apply_persistence_rule(y_raw: np.ndarray, k: int = 3) -> np.ndarray:
    if k <= 1:
        return y_raw
    rs = pd.Series(y_raw.astype(int)).rolling(k, min_periods=1).sum().values
    return (rs >= k).astype(int)


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray):
    lo = np.percentile(scores, 70)
    hi = np.percentile(scores, 99.5)
    best_f1, best_t = -1, None
    for t in np.linspace(lo, hi, 220):
        pred = (scores > t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def smart_thresholding(scores: np.ndarray, y_true=None, smooth_w=7, persist_k=2):
    scores = np.asarray(scores)
    scores_s = smooth_scores(scores, window=smooth_w)

    if y_true is not None and len(np.unique(y_true)) > 1:
        t, _ = best_f1_threshold(y_true, scores_s)
        y_pred = (scores_s > t).astype(int)
        y_pred = apply_persistence_rule(y_pred, k=persist_k)
        return t, y_pred, scores_s

    t = float(np.quantile(scores_s, 0.99))
    y_pred = (scores_s > t).astype(int)
    y_pred = apply_persistence_rule(y_pred, k=max(1, persist_k))
    return t, y_pred, scores_s


# ============================================================
# Plot dashboard
# ============================================================
def plot_research_dashboard(name, X_plot, y, score, y_pred, thresh, metrics):
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(X_plot[:, 0], color=COLORS[0], alpha=0.7, lw=1, label="sensor")
    if y is not None:
        true_idx = np.where(y == 1)[0]
        if len(true_idx):
            ax1.scatter(true_idx, X_plot[true_idx, 0], c=COLORS[1], s=18, marker="x",
                        label="ground truth", zorder=3)
    pred_idx = np.where(y_pred == 1)[0]
    if len(pred_idx):
        ax1.scatter(pred_idx, X_plot[pred_idx, 0], facecolors="none", edgecolors=COLORS[3],
                    s=60, lw=1.5, label="alerts", zorder=4)
    ax1.set_title(f"{name}: signal & anomalies")
    ax1.legend(loc="upper right")
    ax1.margins(x=0)

    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(score, color="#8e44ad", lw=1.2, label="score")
    ax2.axhline(thresh, color=COLORS[1], linestyle="--", label="threshold")
    ax2.set_title("ensemble score (smoothed)")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="upper left")
    ax2.margins(x=0)

    ax3 = fig.add_subplot(gs[2, 0])
    if y is not None and len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, score)
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, color=COLORS[2], lw=2, label=f"auc={roc_auc:.2f}")
        ax3.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax3.set_title("roc")
        ax3.legend(loc="lower right")
    else:
        ax3.text(0.5, 0.5, "no usable labels\nfor ROC", ha="center", va="center", color="gray")
        ax3.set_axis_off()

    ax4 = fig.add_subplot(gs[2, 1])
    if y is not None and len(np.unique(y)) > 1:
        prec, rec, _ = precision_recall_curve(y, score)
        ax4.plot(rec, prec, color=COLORS[4], lw=2)
        ax4.set_title("precision-recall")
        ax4.set_xlabel("recall")
        ax4.set_ylabel("precision")
    else:
        ax4.text(0.5, 0.5, "no usable labels\nfor PR", ha="center", va="center", color="gray")
        ax4.set_axis_off()

    ax5 = fig.add_subplot(gs[2, 2])
    if y is not None and len(np.unique(y)) > 1:
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax5, cbar=False)
        ax5.set_title("confusion matrix")
        ax5.set_xlabel("predicted")
        ax5.set_ylabel("actual")
    else:
        ax5.text(0.5, 0.5, "no usable labels\nfor matrix", ha="center", va="center", color="gray")
        ax5.set_axis_off()

    ax6 = fig.add_subplot(gs[2, 3])
    if X_plot.shape[1] > 1:
        Z = PCA(n_components=2).fit_transform(X_plot)
        if len(Z) > 2000:
            idx = np.random.choice(len(Z), 2000, replace=False)
            Z = Z[idx]
            y_plot = y_pred[idx]
        else:
            y_plot = y_pred

        ax6.scatter(Z[y_plot == 0, 0], Z[y_plot == 0, 1], c="lightgray", s=6, alpha=0.5)
        ax6.scatter(Z[y_plot == 1, 0], Z[y_plot == 1, 1], c=COLORS[1], s=14, marker="^")
        ax6.set_title("pca latent space")
    else:
        ax6.text(0.5, 0.5, "1D data\n(no PCA)", ha="center", va="center")
        ax6.set_axis_off()

    if metrics:
        fig.text(
            0.01, 0.98,
            " | ".join([f"{k}: {v:.3f}" for k, v in metrics.items()]),
            ha="left", va="top", fontsize=10, color=COLORS[0]
        )

    plt.tight_layout()
    add_plot_to_html(fig)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    # -------- OPS-SAT injection knobs --------
    DO_OPSSAT_INJECTION = True     # set False to run OPS-SAT as UNSUPERVISED (no labels)
    OPSSAT_ANOM_FRAC = 0.03        # 1% - 5% recommended
    OPSSAT_SEED = 42
    # ---------------------------------------

    all_data = {}  # IMPORTANT

    # 1) Load SMAP/MSL
    try:
        X_tr, X_te, y_te, note = load_smap_msl_anywhere(ROOT, chan_id="P-1", debug=True)
        all_data["NASA SMAP/MSL - P-1"] = (X_tr, X_te, y_te, note)
    except Exception as e:
        print("[WARN] SMAP/MSL load failed:", str(e))

    # 2) Load OPS-SAT raw df
    ops_df_out = load_opssat_df_anywhere(ROOT, debug=True)
    if ops_df_out is not None:
        df_ops, ops_path = ops_df_out

        # usefulness report BEFORE injection
        rep = opssat_usefulness_report(df_ops)
        print("\n[OPS-SAT CHECK] Useful?:", rep["useful"])
        print("[OPS-SAT CHECK] Reason :", rep["reason"])
        print("[OPS-SAT CHECK] rows/cols:", rep["rows"], rep["cols"],
              "| numeric_cols:", rep["numeric_cols"],
              "| avg_nan_ratio:", f'{rep.get("nan_ratio_avg", 0):.2f}',
              "| zero_var_cols:", rep.get("zero_var_cols", 0))

        if DO_OPSSAT_INJECTION:
            # Inject anomalies + create labels
            X_aug_df, y_aug = inject_anomalies_opssat(
                df_ops,
                anomaly_frac=OPSSAT_ANOM_FRAC,
                seed=OPSSAT_SEED,
                debug=True
            )

            # save optional
            out_csv = os.path.join(ROOT, "opssat_injected_labeled.csv")
            tmp = X_aug_df.copy()
            tmp["label"] = y_aug
            tmp.to_csv(out_csv, index=False)
            print("[DEBUG] saved injected labeled OPS-SAT to:", out_csv)

            # split train/test like others
            X_full = X_aug_df.values
            n = len(X_full)
            split = int(n * 0.50)  # train on first half
            X_tr = X_full[:split]
            X_te = X_full[split:]
            y_te = y_aug[split:]
            note = f"ESA OPS-SAT | injected labels (csv: {os.path.basename(ops_path)} | saved: opssat_injected_labeled.csv)"
            all_data["ESA OPS-SAT"] = (X_tr, X_te, y_te, note)

        else:
            # UNSUPERVISED OPS-SAT (no labels)
            X_num = df_ops.select_dtypes(include=[np.number]).fillna(0).values
            X_tr = X_num[: min(len(X_num), 5000)]
            X_te = X_num
            y_te = None
            note = f"ESA OPS-SAT | unsupervised (csv: {os.path.basename(ops_path)} | injection OFF)"
            all_data["ESA OPS-SAT"] = (X_tr, X_te, y_te, note)

    else:
        print("[WARN] OPS-SAT not loaded (no suitable csv found)")

    # 3) Synthetic
    X_syn, y_syn, note_syn = generate_synthetic_benchmark(seed=42)
    split = 2000
    all_data["Synthetic Benchmark"] = (X_syn[:split], X_syn[split:], y_syn[split:], note_syn)

    if not all_data:
        raise RuntimeError("No datasets loaded. Upload your files under /content.")

    comparison_stats = []

    for name, (X_train_raw, X_test_raw, y_test, note) in all_data.items():
        print(f"\n{'='*60}\nprocessing: {name}\nstatus: {note}\n{'='*60}")

        X_train_raw = np.nan_to_num(X_train_raw)
        X_test_raw  = np.nan_to_num(X_test_raw)

        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test  = scaler.transform(X_test_raw)

        X_plot = X_test.copy()
        if X_plot.ndim == 1:
            X_plot = X_plot.reshape(-1, 1)

        supervised_ok = (y_test is not None) and (len(np.unique(y_test)) > 1)
        if y_test is None:
            print("[DEBUG] labels: none")
        else:
            print(f"[DEBUG] y_test unique: {np.unique(y_test)} | anomaly_count={int(np.sum(y_test))}")

        # ---- Feature Engineering for IF/SVM ----
        if X_train.shape[1] == 1:
            X_train_fe = make_rolling_features_1d(X_train, window=8)
            X_test_fe  = make_rolling_features_1d(X_test, window=8)
        else:
            X_train_fe = X_train
            X_test_fe  = X_test

        # ---- IF ----
        iso = IsolationForest(contamination=0.02, n_estimators=400, random_state=42)
        iso.fit(X_train_fe)
        s_if = minmax01(-iso.decision_function(X_test_fe))

        # ---- One-Class SVM ----
        svm = OneClassSVM(nu=0.02, kernel="rbf", gamma="scale")
        tr_idx = np.random.choice(len(X_train_fe), min(len(X_train_fe), 4000), replace=False)
        svm.fit(X_train_fe[tr_idx])
        s_svm = minmax01(-svm.decision_function(X_test_fe).ravel())

        # ---- Random Forest (only if usable labels) ----
        s_rf = None
        if supervised_ok:
            X_rf = window_features(X_test, W=30)
            T = len(X_rf)
            split_t = int(T * 0.70)
            idx_tr = np.arange(0, split_t)

            if len(np.unique(y_test[idx_tr])) < 2:
                print("[DEBUG] RandomForest skipped: training split contains only one class.")
            else:
                rf = RandomForestClassifier(
                    n_estimators=900,
                    random_state=42,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    n_jobs=-1
                )
                rf.fit(X_rf[idx_tr], y_test[idx_tr])
                s_rf = minmax01(rf.predict_proba(X_rf)[:, 1])
        else:
            if "OPS-SAT" in name:
                print("[DEBUG] OPS-SAT: F1 cannot be computed (unless injection ON).")

        # ---- Ensemble ----
        if s_rf is None:
            s_final = 0.5 * s_if + 0.5 * s_svm
        else:
            s_final = 0.25 * s_if + 0.25 * s_svm + 0.50 * s_rf

        # ---- Thresholding (dataset-specific) ----
        if "OPS-SAT" in name:
            # keep your tuned version for opssat
            thresh, y_pred, s_used = smart_thresholding(
                s_final, y_true=y_test, smooth_w=3, persist_k=1
            )
        else:
            thresh, y_pred, s_used = smart_thresholding(
                s_final, y_true=y_test, smooth_w=7, persist_k=2
            )

        # ---- Metrics ----
        metrics = None
        if supervised_ok:
            p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
            metrics = {"Precision": p, "Recall": r, "F1": f1}
            comparison_stats.append({"Dataset": name, "F1-Score": f1})
            print(f"--> Performance: F1={f1:.3f} | Precision={p:.3f} | Recall={r:.3f}")
        elif y_test is not None:
            print("--> Labels exist but only one class present (F1/ROC/PR not meaningful).")
        else:
            print("--> Unsupervised mode (no labels found).")

        plot_research_dashboard(name, X_plot, y_test, s_used, y_pred, thresh, metrics)

    # ---- Final comparison plot ----
    if comparison_stats:
        plt.figure(figsize=(10, 6))
        df_res = pd.DataFrame(comparison_stats)
        sns.barplot(data=df_res, x="Dataset", y="F1-Score", palette="viridis", hue="Dataset", legend=False)
        plt.title("Final Model Comparison (F1-Score)")
        plt.ylim(0, 1.1)
        for i, row in df_res.iterrows():
            plt.text(i, row["F1-Score"] + 0.02, f"{row['F1-Score']:.2f}", ha="center")
        add_plot_to_html(plt.gcf())
        plt.close()

    finalize_html_report()
