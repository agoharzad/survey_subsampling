#!/usr/bin/env python
"""Survey subsampling functions and runscript."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import PathLike, makedirs
from os import path as op
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    class_likelihood_ratios,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold

from survey_subsampling import sorting
from survey_subsampling.core import constants
from survey_subsampling.core.learner import Learner


def load_data(
    infile: PathLike, threshold: int = 50, verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Grabs a parquet file, extracts the columns we want, removes NaNs, and returns."""
    # Grabs data file from disk, sets diagnostic labels to 0 or 1.
    df_full = pd.read_parquet(infile)
    df_full[constants.Dx_labels_all] = df_full[constants.Dx_labels_all].replace(
        {2.0: 1, 0.0: 0}
    )

    # Define column selector utility to iteratively use to subsample the dataset.
    def _column_selector(
        df: pd.DataFrame, columns: np.ndarray, drop: bool = False
    ) -> pd.DataFrame:
        return df[columns].dropna(axis=0, how="any") if drop else df[columns]

    def _get_prevalance(df: pd.DataFrame, diagnoses: np.ndarray) -> pd.DataFrame:
        tmp = []
        for dx in diagnoses:
            vc = df[dx].value_counts()
            tmp += [{"Dx": dx, "HC": vc.loc[0.0], "Pt": vc.loc[1.0]}]
        return pd.DataFrame.from_dict(tmp).set_index("Dx").sort_values(by="Pt")

    # Initialize the dataset cleaning
    # Subset table based on all diagnoses, compute prevalance
    df = _column_selector(
        df_full,
        np.append(constants.CBCLABCL_items, constants.Dx_labels_all),
        drop=False,
    )
    df_prev = _get_prevalance(df, diagnoses=constants.Dx_labels_all)

    # Drop low-prevalance diagnoses right away from sparse dataset
    #        full list of diagnoses  -  all diagnoses with low prevalance
    Dx_labels_subset = np.array(
        list(set(df_prev.index) - set(df_prev[df_prev["Pt"] < threshold].index))
    )

    # Prepare to iteratively repeat the process as the total N for each Dx may
    # change as we prune missing data and change the included column lists
    low_N = True
    while low_N:
        # Repeat dataset table subsetting (densely this time) and drop low N
        df = _column_selector(
            df_full, np.append(constants.CBCLABCL_items, Dx_labels_subset), drop=True
        )
        df_prev = _get_prevalance(df, diagnoses=Dx_labels_subset)

        # Grab a dataframe of the low-prevalance diagnoses...
        low_N_df = df_prev[df_prev["Pt"] < threshold]
        if low_N := (len(low_N_df.index) > 0):
            # ... and remove them from the set of consideration, then go again
            Dx_labels_subset = np.array(
                list(set(Dx_labels_subset) - set(low_N_df.index))
            )

    # Report on prevalance table and overall dataset length
    if verbose:
        print(df_prev)
        print("Original Dataset Length:", len(df_full))
        print("Pruned Dataset Length:", len(df))

    return df, df_prev, Dx_labels_subset


def fit_models(
    df: pd.DataFrame, x_ids: np.ndarray, y_ids: np.ndarray, verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """General purpose function for fitting callibrated classifiers."""
    # Establish Models & Cross-Validation Strategy
    #   Base clf: Random Forest. Rationale: non-parametric, has feature importance
    clf_rf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
    #   CV: Stratified K-fold. Rationale: shuffle data, balance classes across folds
    cv = StratifiedKFold(shuffle=True, random_state=42)
    #   Top-level clf: Calibrated CV classifier.
    #   Rationale: prioritizes maintaining class balances
    clf_calib = CalibratedClassifierCV(estimator=clf_rf, cv=cv)
    np.random.seed(42)

    # Create X (feature) matrix: grab relevant survey columns x rows from dataframe
    X = df[x_ids].values

    # Create empty list of learners
    learners = []
    summaries = []
    # For every Dx that we want to predict...
    for y_name in y_ids:
        # Create the y (target) matrix/vector: grab relevant Dx rows from dataframe
        y = df[y_name].values.astype(int)

        # Get Pt and HC counts from dataframe, and initialize the learner object
        _, uc = np.unique(y, return_counts=True)

        current_learner = Learner(dx=y_name, hc_n=uc[0], dx_n=uc[1], x_ids=x_ids)
        # Set-up a CV loop (note, we use the same CV strategy both within the
        #  Calibrated CLF and here, resulting in nested-stratified-k-fold-CV)
        for idx_train, idx_test in cv.split(X, y):
            # Split the dataset into train and test sets
            X_tr = X[idx_train, :]
            y_tr = y[idx_train]

            X_te = X[idx_test, :]
            y_te = y[idx_test]

            # Fit the callibrated classifier on the training data
            clf_calib.fit(X_tr, y_tr)

            # Extract/Generate relevant data from (all internal folds of) the clf...
            # Make predictions on the test set
            y_pred = clf_calib.predict(X_te)

            # Grab training and validation perf using the integrated scoring (accuracy)
            y_pred_tr = clf_calib.predict(X_tr)
            current_learner.acc_train = np.append(
                current_learner.acc_train, accuracy_score(y_tr, y_pred_tr)
            )
            current_learner.acc_valid = np.append(
                current_learner.acc_valid, accuracy_score(y_te, y_pred)
            )

            # Grab feature importance scores
            fis = [
                _.estimator.feature_importances_
                for _ in clf_calib.calibrated_classifiers_
            ]
            current_learner.fi.append(fis)

            # Grab the prediction probabilities
            current_learner.proba = np.append(
                current_learner.proba, clf_calib.predict_proba(X_te)
            )
            current_learner.label = np.append(current_learner.label, y_te)

            # Grab the prediction labels
            current_learner.f1 = np.append(current_learner.f1, f1_score(y_te, y_pred))

            # Grab the sensitivity and specificity (i.e. recall of each of Dx and HC)
            report_dict = classification_report(y_te, y_pred, output_dict=True)
            current_learner.sen = np.append(
                current_learner.sen, report_dict["1"]["recall"]
            )
            current_learner.spe = np.append(
                current_learner.spe, report_dict["0"]["recall"]
            )

            # Grab the positive/negative likelihood ratios
            lrp, lrn = class_likelihood_ratios(y_te, y_pred)
            current_learner.LRp = np.append(current_learner.LRp, lrp)
            current_learner.LRn = np.append(current_learner.LRn, lrn)

        # Summarize current learner performance, save it, and get ready to go again!
        summaries += [current_learner.summary()]

        tmp_learner = {"Dx": current_learner.dx}

        means = np.mean(np.vstack(current_learner.fi), axis=0)  # type: ignore[call-overload]
        for assessment, importance in zip(x_ids, means):
            tmp_learner[assessment] = importance
        learners += [tmp_learner]

        del current_learner

    # Improve formatting of summaries and complete learners
    summaries = pd.concat(summaries).set_index("Dx")
    learners = pd.DataFrame.from_dict(learners).set_index("Dx")

    return learners, summaries


def calculate_feature_importance(
    learners: pd.DataFrame,
    x_ids: np.ndarray,
    outdir: PathLike,
    number_of_questions: int = 20,
    plot: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sorts values using two strategies: aggregate and topN."""
    _, x_ids_sorted_by_aggregate = sorting.aggregate_sort(learners, x_ids)
    _, x_ids_sorted_by_topn, _ = sorting.topn_sort(learners, x_ids)

    # Report the sorted list using both approaches
    sorted_importance_agg = x_ids_sorted_by_aggregate[0:number_of_questions]
    print(
        f"The {number_of_questions} cumulatively most predictive features are:",
        ", ".join(sorted_importance_agg),
    )

    sorted_importance_topn = x_ids_sorted_by_topn[0:number_of_questions]
    print(
        f"The {number_of_questions} most commonly useful features are:",
        ", ".join(sorted_importance_topn),
    )

    # Evaluate sorting consistency
    all_qs = list(set(list(sorted_importance_agg) + list(sorted_importance_topn)))
    diff = np.abs(number_of_questions - len(all_qs))
    frac = 1.0 * diff / number_of_questions * 100
    print(f"The two lists differ by {diff} / {number_of_questions} items ({frac:.2f}%)")

    # Compute the average position across the two methods
    avg_rank = np.mean(
        np.where(x_ids_sorted_by_aggregate[:, None] == x_ids_sorted_by_topn), axis=0
    )
    x_ids_sorted_average = x_ids_sorted_by_aggregate[np.argsort(avg_rank)]

    return x_ids_sorted_by_aggregate, x_ids_sorted_by_topn, x_ids_sorted_average


def degrading_fit(
    df: pd.DataFrame,
    sorted_x_ids: np.ndarray,
    y_ids: np.ndarray,
    threads: int = 4,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Wrapper for fit_models that degrades the performance by reducing the x-set."""
    # Initiatlize some storage containers
    degraded_tuple = []
    futures = []

    with ProcessPoolExecutor(max_workers=threads) as pool:
        # For each number of questions (from the length of sorted_x_ids down to 1)...
        for n_questions in range(len(sorted_x_ids))[::-1]:
            # Grab the first n_questions from the sorted list
            xi = sorted_x_ids[0 : n_questions + 1]

            # Add the diagnostic prediction models to the queue using this reduced x-set
            #   Equiv command: degraded_tuple = fit_models(df, xi, y_ids, verbose=False)
            futures.append(pool.submit(fit_models, df, xi, y_ids, verbose=False))

    # Store the results as they come in
    for future in as_completed(futures):
        degraded_tuple.append(future.result())

    # Separate the learners and respective summaries
    degraded_learners = [dt[0] for dt in degraded_tuple]
    degraded_summaries = [dt[1] for dt in degraded_tuple]

    # Concatenate and return the learners and summaries
    degraded_summaries = pd.concat(degraded_summaries)
    degraded_learners = pd.concat(degraded_learners)
    return degraded_learners, degraded_summaries


def run() -> None:
    """CLI runscript for subsampling."""
    # TODO: improve docstrings, helptext, and the like
    parser = ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("outdir")
    parser.add_argument("-n", "--number_of_questions", default=20, type=int)
    parser.add_argument("--random_state", default=42, type=int)
    parser.add_argument("-t", "--dx_threshold", default=150, type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-w", "--warnings", action="store_true")
    parser.add_argument("--n_threads", default=4, type=int)
    args = parser.parse_args()

    # Grab input values
    infile = args.infile
    outdir = args.outdir
    threshold = args.dx_threshold
    verbose = args.verbose
    NQ = args.number_of_questions
    nt = args.n_threads
    np.random.seed(args.random_state)

    # Prepare output directory
    if not op.isdir(f"{outdir}"):
        makedirs(f"{outdir}")

    # Suppress the warnings that come up when training degrading learners by default
    if not args.warnings:
        import warnings

        warnings.filterwarnings("ignore")

    # Load the dataset and remove incomplete subjects and under-represented diagnoses
    df, df_prev, Dx_labels_subset = load_data(
        infile, threshold=threshold, verbose=verbose
    )

    # Establish baseline prediction, and further remove diagnostic labels for which the
    #  models fail to make reasonable predictions (read as: make predictions to both
    #  classes; identifiable as diagnoses with a NaN for LR+)
    learners, summaries = fit_models(
        df, constants.CBCLABCL_items, Dx_labels_subset, verbose=verbose
    )
    Dx_labels_subset = np.array(
        list(set(Dx_labels_subset) - set(summaries[summaries["LR+"].isna()].index))
    )
    learners = learners.loc[Dx_labels_subset]
    learners.to_parquet(f"{outdir}/learners.parquet")

    summaries = summaries.loc[Dx_labels_subset]
    summaries.to_parquet(f"{outdir}/summaries.parquet")
    if verbose:
        print(summaries)

    # Compute and plot feature importance, and then save results in a CSV file
    sorted_agg, sorted_topn, sorted_avg = calculate_feature_importance(
        learners, constants.CBCLABCL_items, outdir, number_of_questions=NQ
    )
    importance = pd.DataFrame(
        {"Aggregate": sorted_agg, "Top-N": sorted_topn, "Average": sorted_avg}
    )
    importance.to_parquet(f"{outdir}/feature_importance.parquet")

    # Redo the learning process with a degrading set of data
    learners_deg, summaries_deg = degrading_fit(
        df, sorted_avg, Dx_labels_subset, threads=nt, verbose=verbose
    )
    summaries_deg.to_parquet(f"{outdir}/summaries_degraded.parquet")
    learners_deg.to_parquet(f"{outdir}/learners_degraded.parquet")


if __name__ == "__main__":
    run()
