#!/usr/bin/env python3
"""
@brief compute metrics from the prediction pickle(s) produced by inference.py.
"""
import argparse

import pandas as pd

from Metrics.metrics import (accuracy, display_cm, display_mean_cm,
                             evaluate_temporal_detection, analyze_temporal_localisation_errors)


parser = argparse.ArgumentParser()
parser.add_argument("--path_to_pickle", type=str, default=None)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--tasks", nargs="+", default=["cat"], choices=["cat", "dev", "lag"])
parser.add_argument("--mean_cm", nargs="*", default=None, help="pickles of several runs for a mean confusion matrix")
parser.add_argument('--n_actions', type=int, default=11)
parser.add_argument('--n_tclasses', type=int, default=5)
parser.add_argument('--lag_segments', type=int, default=3)
args = parser.parse_args()


if args.mean_cm:
    dfs = [pd.read_pickle(p) for p in args.mean_cm]
    display_mean_cm(dfs, klabel="ycat", kpred="pred_cat", nb_class=args.n_tclasses,
                    matrix_name=args.out + "_morphokinetic")

if args.path_to_pickle:
    df = pd.read_pickle(args.path_to_pickle)

    # Morphokinetic Category
    if "cat" in args.tasks:
        accuracy(df, klabel="ycat", kpred="pred_cat", save_csv_path=args.out + "_cat", nb_class=args.n_tclasses)
        display_cm(df, klabel="ycat", kpred="pred_cat", nb_class=args.n_tclasses,
                   matrix_name=args.out + "_cat")

    # Developmental stages
    if "dev" in args.tasks:
        accuracy(df, klabel="ydev", kpred="pred_dev", save_csv_path=args.out + "_dev")
        display_cm(df, klabel="ydev", kpred="pred_dev", nb_class=args.n_actions,
                   matrix_name=args.out + "_dev")

    # LAG localisation
    if "lag" in args.tasks:
        evaluate_temporal_detection(df, max_segments=args.lag_segments, output_dir=args.out + "_lag")
        analyze_temporal_localisation_errors(df, max_segments=args.lag_segments, output_dir=args.out + "_lag")
