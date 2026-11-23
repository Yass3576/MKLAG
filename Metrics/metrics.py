import os
from collections import defaultdict
from typing import List, Dict, Union, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


#########################################################
# Helpers
#########################################################

CLASS_NAMES = {
    2: ['NT', 'T'],
    5: ['NT', 'ADB', 'SSB', 'HB', 'EHB'],
}

_CLS = np.array([0, 0, 0, 0, 1, 2, 3, 4])


def _preds(df, kpred, nb_class):
    if nb_class is None or nb_class <= 5:
        return df, nb_class
    df = df.copy()
    df[kpred] = [_CLS[_as_sequence(v).astype(int)] for v in df[kpred]]
    return df, 5


#########################################################
# Classification metrics
#########################################################

def accuracy(df: pd.DataFrame,
             klabel: str,
             kpred: str,
             save_csv_path: str = "report",
             nb_class: Optional[int] = None) -> dict:
    """
    @brief Compute classification accuracy and per-class precision / recall / F1 / IoU.

    @param df (pd.DataFrame): DataFrame indexed by video, with columns klabel and kpred.

    @param save_csv_path (str): path of the CSV report.

    @return (dict): accuracy, per-class metrics.
    """
    count = 0
    df, nb_class = _preds(df, kpred, nb_class)
    total_samples = 0
    correct = 0
    class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'support': 0})

    for video in df.index:
        count += 1
        print(video)
        labels = _as_sequence(df.loc[video][klabel])
        preds = _as_sequence(df.loc[video][kpred])
        print(labels, preds)

        for label, pred in zip(labels, preds):
            class_stats[label]['support'] += 1

            if pred == label:
                correct += 1
                class_stats[label]['tp'] += 1
            else:
                class_stats[label]['fn'] += 1
                class_stats[pred]['fp'] += 1
            total_samples += 1

    acc = correct / total_samples if total_samples > 0 else 0

    total_tp = 0 
    total_fp = 0
    total_fn = 0
    rows = []
    metrics_per_class = {}

    for cls in sorted(class_stats):
        tp, fp, fn = class_stats[cls]['tp'], class_stats[cls]['fp'], class_stats[cls]['fn']
        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        
        metrics_per_class[cls] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': class_stats[cls]['support']}
        rows.append({'class': cls, **metrics_per_class[cls]})

    print(f"\nGlobal Accuracy: {acc:.4f}")

    rows.append({'class': 'GLOBAL_ACC', 'precision': acc, 'recall': '', 'f1': '', 'support': total_samples})

    if save_csv_path:
        pd.DataFrame(rows).to_csv(save_csv_path + ".csv", index=False)
    print("count ", count)
    return {
        'accuracy_global': acc,
        'metrics_per_class': metrics_per_class,
    }


def compute_cm(df: pd.DataFrame, klabel: str, kpred: str, nb_class: int) -> np.ndarray:
    """
    @brief Confusion matrix (rows = predicted class, columns = true class).
    """
    cm = np.zeros((nb_class, nb_class), dtype=np.int32)

    for video in df.index:
        labels = _as_sequence(df.loc[video][klabel])
        preds = _as_sequence(df.loc[video][kpred])

        for label, pred in zip(labels, preds):
            cm[pred, label] += 1

    return cm


def display_cm(df: pd.DataFrame, klabel: str, kpred: str, nb_class: int, matrix_name: str,
               title: str = "Confusion matrix", figsize: tuple = (10, 8)) -> None:
    """
    @brief Plot and save the confusion matrix heatmap (`matrix_name`.png).
    """
    df, nb_class = _preds(df, kpred, nb_class)
    cm = compute_cm(df, klabel, kpred, nb_class)
    classes = _class_names(nb_class)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)

    plt.figure(figsize=figsize)
    ax = sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True, annot_kws={"size": 12}, linewidths=.5)

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('True Class', fontsize=12)
    ax.set_ylabel('Predicted Label', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    plt.tight_layout()
    plt.savefig(f"{matrix_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved as {matrix_name}.png")


#########################################################
# Mean confusion matrices over several runs
#########################################################

def compute_mean_cm(df_list: List[pd.DataFrame], klabel: str, kpred: str, nb_class: int,
                    weights: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
    """
    @brief Mean confusion matrix in frequencies (normalised per true class = column).

    Returns {'mean', 'std', 'all' (per-run frequency CM), 'sum' (absolute counts), 'n_runs'}
    """
    freq_matrices, abs_matrices = [], []
    df_list = [_preds(df, kpred, nb_class)[0] for df in df_list]
    nb_class = 5 if nb_class > 5 else nb_class

    for df in df_list:
        cm_abs = compute_cm(df, klabel, kpred, nb_class)
        abs_matrices.append(cm_abs)
        freq_matrices.append(cm_abs / (cm_abs.sum(axis=0, keepdims=True) + 1e-10))

    freq_array = np.stack(freq_matrices, axis=0)
    abs_array = np.stack(abs_matrices, axis=0)

    if weights is not None:
        mean_cm = np.average(freq_array, axis=0, weights=np.asarray(weights))
    else:
        mean_cm = np.mean(freq_array, axis=0)

    return {
        "mean": mean_cm,
        "std": np.std(freq_array, axis=0),
        "all": freq_array,
        "sum": np.sum(abs_array, axis=0),
        "n_runs": len(df_list),
    }


def display_mean_cm(df_list: Union[List[pd.DataFrame], Dict[str, np.ndarray]], klabel: str, kpred: str,
                    nb_class: int, matrix_name: str, title: str = "Mean confusion matrix",
                    figsize: tuple = (12, 10), show_std: bool = True,
                    classes: Optional[List[str]] = None, std_format: str = "±{:.1%}") -> Dict[str, np.ndarray]:
    """
    @brief Display the mean confusion matrix in frequencies (mean ± std, rows = predicted class, columns = true class).
    """
    stats = df_list if isinstance(df_list, dict) else compute_mean_cm(
        df_list, klabel, kpred, nb_class)

    mean_cm, std_cm = stats["mean"], stats["std"]
    nb_class = mean_cm.shape[0]   
    if classes is None:
        classes = _class_names(nb_class)

    cm_df = pd.DataFrame(mean_cm, index=classes, columns=classes)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_df, cmap="Blues", cbar=True, linewidths=0.5, annot=False, ax=ax)

    for i in range(nb_class):
        for j in range(nb_class):
            mean_val, std_val = mean_cm[i, j], std_cm[i, j]
            txt = f"{mean_val:.1%}\n{std_format.format(std_val)}" if show_std else f"{mean_val:.1%}"
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=11,
                    color="white" if mean_val > 0.5 else "black")

    ax.set_title(f"{title}\nMean ± Std over {stats['n_runs']} runs", fontsize=14, pad=20)
    ax.set_xlabel("True class", fontsize=12)
    ax.set_ylabel("Predicted label", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.collections[0].colorbar.set_label("Frequency", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(f"{matrix_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Mean confusion matrix saved to {matrix_name}.png")
    print(f"Number of runs: {stats['n_runs']}")

    _create_detailed_matrix(stats, matrix_name, classes)
    return stats


def _create_detailed_matrix(stats: Dict[str, np.ndarray], matrix_name: str, classes: List[str]) -> None:
    """Save detailed per-class statistics from the mean confusion matrix."""
    mean_cm, std_cm = stats["mean"], stats["std"]

    with open(f"{matrix_name}_detailed.txt", "w") as f:
        f.write(f"Mean confusion matrix statistics (n={stats['n_runs']})\n")
        f.write("=" * 60 + "\n\n")

        for i in range(mean_cm.shape[0]):
            for j in range(mean_cm.shape[1]):
                f.write(f"True {classes[j]} → Pred {classes[i]}: {mean_cm[i, j]:.2%} ± {std_cm[i, j]:.2%}\n")
            f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("Per-class metrics (from mean CM)\n")

        for i, cls in enumerate(classes):
            tp = mean_cm[i, i]
            fn = mean_cm[:, i].sum() - tp
            fp = mean_cm[i, :].sum() - tp

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)

            f.write(f"\nClass {cls}:\n")
            f.write(f"  Precision: {precision:.2%}\n")
            f.write(f"  Recall:    {recall:.2%}\n")
            f.write(f"  F1-score:  {f1:.2%}\n")

    print(f"Detailed statistics saved to {matrix_name}_detailed.txt")



#########################################################
# LAG temporal localisation
#########################################################

def iou(s_pred, e_pred, s_gt, e_gt) -> float:
    intersection = max(0.0, min(e_pred, e_gt) - max(s_pred, s_gt))
    union = (e_pred - s_pred) + (e_gt - s_gt) - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


def merge_segments(segments):
    """@brief Merge overlapping segments (start, end)"""
    if len(segments) == 0:
        return []
    segments = sorted(segments, key=lambda x: x[0])
    merged = [segments[0]]
    for current in segments[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]:
            merged[-1] = (prev[0], max(prev[1], current[1]))
        else:
            merged.append(current)
    return merged


def gt_segment(row, i):
    """GT segment i (1-based) of a prediction row, or None if absent."""
    if not bool(_to_int(row[f"has_seg{i}"])):
        return None
    s = float(row[f"lag_s{i}"])
    e = float(row[f"lag_e{i}"]) #float(np.clip(float(row[f"lag_e{i}"]), 0.0, 1.0))
    return s, e


def scores(tp, fp, fn, tn, ious):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total > 0 else 0
        return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "Precision": precision, "Recall": recall, "F1": f1, "Accuracy": acc,
                "Mean_IoU": np.mean(ious) if len(ious) > 0 else 0}


def evaluate_temporal_detection(df, max_segments=3, conf_threshold=0.5, output_dir: str = None):
    """
     @brief segment-wise evaluation: prediction k is compared to GT segment k.
      - presence: TP / FP / FN / TN with P(n_segment >= k) > conf_threshold
      - localisation: IoU on the TP segments

    pred_lag = [z_k, s_k, d_k] x 3 with sigmoid(z_k) = P(n_segment >= k)
    """
    tp = fp = fn = tn = 0
    iou_list = []
    segment_stats = {i: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "iou": []} for i in range(1, max_segments + 1)}

    for idx in df.index:
        row = df.loc[idx]

        for i in range(1, max_segments + 1):
            p, s_pred, e_pred = decode_pred_lag(row["pred_lag"], i)
            has_pred = p > conf_threshold
            gt = gt_segment(row, i)
            has_gt = gt is not None

            if not has_gt and not has_pred:
                tn += 1
                segment_stats[i]["tn"] += 1
            elif not has_gt and has_pred:
                fp += 1
                segment_stats[i]["fp"] += 1
            elif has_gt and not has_pred:
                fn += 1
                segment_stats[i]["fn"] += 1
            else:
                seg_iou = iou(s_pred, e_pred, gt[0], gt[1])
                iou_list.append(seg_iou)
                segment_stats[i]["iou"].append(seg_iou)
                segment_stats[i]["tp"] += 1
                tp += 1

    global_results = pd.DataFrame([scores(tp, fp, fn, tn, iou_list)])
    per_segment_df = pd.DataFrame([
        {"segment": i, **scores(st["tp"], st["fp"], st["fn"], st["tn"], st["iou"])}
        for i, st in segment_stats.items()
    ])

    print("\n===== GLOBAL RESULTS =====")
    print(global_results)
    print("\n===== PER SEGMENT RESULTS =====")
    print(per_segment_df)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        global_results.to_csv(os.path.join(output_dir, "global_temporal_metrics.csv"), index=False)
        per_segment_df.to_csv(os.path.join(output_dir, "per_segment_temporal_metrics.csv"), index=False)

    return global_results, per_segment_df


def analyze_temporal_localisation_errors(df, max_segments=3, conf_threshold=0.5, output_dir=None):
    """
    @brief LAG localisation errors on the GT segments (a missed GT segment counts with IoU = 0, duration 0).
    @return per-segment DataFrame, summary DataFrame (+ CSVs and pred_vs_gt_duration.png in output_dir)
    """
    records = []
    for idx in df.index:
        row = df.loc[idx]
        for i in range(1, max_segments + 1):
            gt = gt_segment(row, i)
            if gt is None:
                continue
            p, s_pred, e_pred = decode_pred_lag(row["pred_lag"], i)
            detected = p > conf_threshold
            d_gt = gt[1] - gt[0]
            d_pred = e_pred - s_pred if detected else 0.0
            records.append({"Segment": i, "GT_Duration": d_gt, "Pred_Duration": d_pred,
                            "IoU": iou(s_pred, e_pred, gt[0], gt[1]) if detected else 0.0,
                            "Duration_Error": abs(d_pred - d_gt)})

    analysis_df = pd.DataFrame(records, columns=["Segment", "GT_Duration", "Pred_Duration", "IoU", "Duration_Error"])
    summary_df = pd.DataFrame([{
        "Mean_IoU": analysis_df["IoU"].mean(),
        #"Median_IoU": analysis_df["IoU"].median(),
        "Mean_Duration_Error": analysis_df["Duration_Error"].mean(),
        #"Median_Duration_Error": analysis_df["Duration_Error"].median(),
    }])
    print("\n===== TEMPORAL LOCALISATION ANALYSIS =====")
    print(summary_df)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        analysis_df.to_csv(os.path.join(output_dir, "temporal_localisation_analysis.csv"), index=False)
        summary_df.to_csv(os.path.join(output_dir, "temporal_localisation_summary.csv"), index=False)

        if len(analysis_df) > 0:   # predicted vs GT duration, coloured by IoU
            plt.figure(figsize=(6, 6))
            sns.scatterplot(data=analysis_df, x="GT_Duration", y="Pred_Duration", hue="IoU", palette="viridis", s=60)
            plt.plot([0, 1], [0, 1], "--", color="grey")
            plt.xlim(0, 1), plt.ylim(0, 1)
            plt.xlabel("GT duration (normalised)"), plt.ylabel("Predicted duration (normalised)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "pred_vs_gt_duration.png"), dpi=300)
            plt.close()

    return analysis_df, summary_df


def _to_int(x) -> int:
    return int(x.item()) if hasattr(x, "item") else int(x)

def _class_names(nb_class: int) -> List[str]:
    return CLASS_NAMES.get(nb_class, [str(i) for i in range(nb_class)])


def _as_sequence(x) -> np.ndarray:
    """Tensor / scalar / list -> 1D numpy array."""
    x = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return np.atleast_1d(x)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def decode_pred_lag(pred_lag, i: int):
    """
    Decode a `pred_lag` vector [z_k, s_k, d_k] x 3.
    sigmoid(z_k) = P(n_segment >= k), e_k = s_k + d_k.

    @return (presence probability, start, end clipped to [0, 1]) x 3
    """
    pred = _as_sequence(pred_lag).reshape(-1)
    p = _sigmoid(pred[(i - 1) * 3])
    s = float(pred[(i - 1) * 3 + 1])
    e = float(np.clip(s + pred[(i - 1) * 3 + 2], 0.0, 1.0))
    return p, s, e