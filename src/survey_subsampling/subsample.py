#!/usr/bin/env python

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from argparse import ArgumentParser
import os
from os import PathLike
from typing import List

from sklearn.metrics import accuracy_score, f1_score, classification_report, class_likelihood_ratios
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import plotly.express as px
import pandas as pd
import numpy as np
from joblib import dump, load



CBCL_ABCL_cannot_be_harmonized = ["cl1", "cl2", "cl6", "cl15", "cl23", "cl30", "cl38", "cl44",
                                  "cl47", "cl49", "cl53", "cl55", "cl56h", "cl59", "cl60",
                                  "cl64", "cl67", "cl72", "cl73", "cl76", "cl78", "cl81",
                                  "cl83", "cl89", "cl92", "cl96", "cl98", "cl99", "cl106",
                                  "cl107", "cl108", "cl109", "cl110", "cl113"]

CBCL_items = ['cl1', 'cl2', 'cl3', 'cl4', 'cl5', 'cl6', 'cl7', 'cl8', 'cl9', 'cl10', 'cl11',
              'cl12', 'cl13', 'cl14', 'cl15', 'cl16', 'cl17', 'cl18', 'cl19', 'cl20', 'cl21',
              'cl22', 'cl23', 'cl24', 'cl25', 'cl26', 'cl27', 'cl28', 'cl29', 'cl30', 'cl31',
              'cl32', 'cl33', 'cl34', 'cl35', 'cl36', 'cl37', 'cl38', 'cl39', 'cl40', 'cl41',
              'cl42', 'cl43', 'cl44', 'cl45', 'cl46', 'cl47', 'cl48', 'cl49', 'cl50', 'cl51',
              'cl52', 'cl53', 'cl54', 'cl55', 'cl56', 'cl56a', 'cl56b', 'cl56c', 'cl56d',
              'cl56e', 'cl56f', 'cl56g', 'cl56h', 'cl57', 'cl58', 'cl59', 'cl60', 'cl61',
              'cl62', 'cl63', 'cl64', 'cl65', 'cl66', 'cl67', 'cl68', 'cl69', 'cl70', 'cl71',
              'cl72', 'cl73', 'cl74', 'cl75', 'cl76', 'cl77', 'cl78', 'cl79', 'cl80', 'cl81',
              'cl82', 'cl83', 'cl84', 'cl85', 'cl86', 'cl87', 'cl88', 'cl89', 'cl90', 'cl91',
              'cl92', 'cl93', 'cl94', 'cl95', 'cl96', 'cl97', 'cl98', 'cl99', 'cl100', 'cl101',
              'cl102', 'cl103', 'cl104', 'cl105', 'cl106', 'cl107', 'cl108', 'cl109', 'cl110',
              'cl111', 'cl112', 'cl113']

ABCL_items = ['al1', 'al2', 'al3', 'al4', 'al5', 'al6', 'al7', 'al8', 'al9', 'al10', 'al11',
              'al12', 'al13', 'al14', 'al15', 'al16', 'al17', 'al18', 'al19', 'al20', 'al21',
              'al22', 'al23', 'al24', 'al25', 'al26', 'al27', 'al28', 'al29', 'al30', 'al31',
              'al32', 'al33', 'al34', 'al35', 'al36', 'al37', 'al38', 'al39', 'al40', 'al41',
              'al42', 'al43', 'al44', 'al45', 'al46', 'al47', 'al48', 'al49', 'al50', 'al51',
              'al52', 'al53', 'al54', 'al55', 'al56', 'al56a', 'al56b', 'al56c', 'al56d',
              'al56e', 'al56f', 'al56g', 'al57', 'al58', 'al59', 'al60', 'al61', 'al62',
              'al63', 'al64', 'al65', 'al66', 'al67', 'al68', 'al69', 'al70', 'al71', 'al72',
              'al73', 'al74', 'al75', 'al76', 'al77', 'al78', 'al79', 'al80', 'al81', 'al82',
              'al83', 'al84', 'al85', 'al86', 'al87', 'al88', 'al89', 'al90', 'al91', 'al92',
              'al93', 'al94', 'al95', 'al96', 'al97', 'al98', 'al99', 'al100', 'al101', 'al102',
              'al103', 'al104', 'al105', 'al106', 'al107', 'al108', 'al109', 'al110', 'al111',
              'al112', 'al113', 'al114', 'al115', 'al116', 'al117', 'al118', 'al119', 'al120',
              'al121', 'al122', 'al123', 'al124', 'al125', 'al126']

CBCLABCL_items = list(set(CBCL_items) - set(CBCL_ABCL_cannot_be_harmonized))

Dx_labels_all = ["dcany", "dcanyanx", "dcanydep", "dcanyhk", "dcanycd", "dcsepa","dcspph", "dcsoph", "dcpanic", "dcagor", "dcptsd", "dcocd",
"dcgena", "dcdmdd", "dcmadep", "dcmania", "dcodd", "dccd"]

Dx_labels_valid = ["dcany", "dcanyanx", "dcanydep", "dcanyhk", "dcanycd",
                  "dcmadep", "dcodd"]


@dataclass
class Learner:
    """Data class to record and present statistics from trained & evaluated models"""
    dx: str  # Diagnosis
    hc_n: int  # Number of healthy controls
    dx_n: int  # Number of patients
    x_ids: List = field(default_factory=lambda: [])  # List of features used in the learner
    fi: List = field(default_factory=lambda: [])  # Feature importance lists
    f1: List = field(default_factory=lambda: [])  # F1 score lists
    sen: List = field(default_factory=lambda: [])  # Sensitivity score lists
    spe: List = field(default_factory=lambda: [])  # Specificity score lists
    LRp: List = field(default_factory=lambda: [])  # Positive likelihood ratio lists
    LRn: List = field(default_factory=lambda: [])  # Negative likelihood ratio lists
    acc_train: List = field(default_factory=lambda: [])  # Performance on the training set
    acc_valid: List = field(default_factory=lambda: [])  # Performance on the validation set

    proba: List = field(default_factory=lambda: [])  # Prediction probability/confidence
    label: List = field(default_factory=lambda: [])  # Prediction labels

    def summary(self, verbose=False):
        """Constructs a dataframe from the models and prints a summary report"""
        self._sanitize()
        
        tmpdict = [{
            'Dx': self.dx,
            'N_HC': self.hc_n,
            'N_Pt': self.dx_n,
            'N_xs': len(self.x_ids),
            'F1': np.mean(self.f1),
            'Sensitivity': np.mean(self.sen),
            'Specificity': np.mean(self.spe),
            'LR+': np.mean(self.LRp),
            'LR-': np.mean(self.LRn),
            'Accuracy (Train)': np.mean(self.acc_train),
            'Accuracy (Validation)': np.mean(self.acc_valid)
            
        }]
        tmpdf = pd.DataFrame.from_dict(tmpdict)

        if verbose:
            self._plot_probas()
            print(tmpdf)

        return tmpdf
    
    def _sanitize(self):
        """Make lists of lists a bit more palletable..."""
        self.proba = np.vstack(self.proba)
        self.label = np.hstack(self.label)
    
    def _plot_probas(self):
        """Show prediction confidence scores"""
        tmpdict = {
            'proba' : self.proba[:,1],            
            'label' : self.label
        }
        tmpdf = pd.DataFrame.from_dict(tmpdict)
        fig = px.histogram(tmpdf, color='label', x='proba', marginal='box',
                           barmode="overlay", opacity=0.7, category_orders={'label':[0,1]})
        fig.show()

@dataclass
class Learner_eval:
    """Data class to record and present statistics from previously trained models on new data"""
    dx: str  # Diagnosis
    hc_n: int  # Number of healthy controls
    dx_n: int  # Number of patients
    x_ids: List = field(default_factory=lambda: [])  # List of features used in the learner
    fi: List = field(default_factory=lambda: [])  # Feature importance lists
    f1: List = field(default_factory=lambda: [])  # F1 score lists
    sen: List = field(default_factory=lambda: [])  # Sensitivity score lists
    spe: List = field(default_factory=lambda: [])  # Specificity score lists
    LRp: List = field(default_factory=lambda: [])  # Positive likelihood ratio lists
    LRn: List = field(default_factory=lambda: [])  # Negative likelihood ratio lists
    acc_valid: List = field(default_factory=lambda: [])  # Performance on the validation set

    proba: List = field(default_factory=lambda: [])  # Prediction probability/confidence
    label: List = field(default_factory=lambda: [])  # Prediction labels

    def summary(model_fitted, verbose=False):
        """Constructs a dataframe from the models and prints a summary report"""
        model_fitted._sanitize()
        
        tmpdict = [{
            'Dx': model_fitted.dx,
            'N_HC': model_fitted.hc_n,
            'N_Pt': model_fitted.dx_n,
            'N_xs': len(model_fitted.x_ids),
            'F1': np.mean(model_fitted.f1),
            'Sensitivity': np.mean(model_fitted.sen),
            'Specificity': np.mean(model_fitted.spe),
            'LR+': np.mean(model_fitted.LRp),
            'LR-': np.mean(model_fitted.LRn),
            'Accuracy (Validation)': np.mean(model_fitted.acc_valid)
            
        }]
        tmpdf = pd.DataFrame.from_dict(tmpdict)

        if verbose:
            model_fitted._plot_probas()
            print(tmpdf)

        return tmpdf
    
    def _sanitize(model_fitted):
        """Make lists of lists a bit more palletable..."""
        model_fitted.proba = np.vstack(model_fitted.proba)
        model_fitted.label = np.hstack(model_fitted.label)
    
    def _plot_probas(model_fitted):
        """Show prediction confidence scores"""
        tmpdict = {
            'proba' : model_fitted.proba[:,1],            
            'label' : model_fitted.label
        }
        tmpdf = pd.DataFrame.from_dict(tmpdict)
        fig = px.histogram(tmpdf, color='label', x='proba', marginal='box',
                           barmode="overlay", opacity=0.7, category_orders={'label':[0,1]})
        fig.show()

def load_data(infile: PathLike,load_mod:bool,threshold: int=140, verbose: bool=True):
    """Grabs a parquet file, extracts the columns we want, removes NaNs, and returns"""
        
    # Grabs data file from disk, sets diagnostic labels to 0 or 1.
    df_full = pd.read_parquet(infile)
    df_full[Dx_labels_all] = df_full[Dx_labels_all].replace({2.0:1, 0.0:0})

    # Define column selector utility that we'll iteratively use to subsample the dataset.
    def _column_selector(df: pd.DataFrame, columns: list, drop: bool=False):        
        return df[columns].dropna(axis=0, how='any') if drop else df[columns]

    def _get_prevalance(df: pd.DataFrame, diagnoses: list):
        tmp = []
        for dx in diagnoses:
            vc = df[dx].value_counts()
            tmp += [{"Dx": dx, "HC": vc.loc[0.0], "Pt":vc.loc[1.0]}]
        return pd.DataFrame.from_dict(tmp).set_index('Dx').sort_values(by='Pt')

    # Initialize the dataset cleaning

    if load_mod==True:
        Dx_labels = Dx_labels_valid
    else:
        Dx_labels = Dx_labels_all

    # Subset table based on all diagnoses, compute prevalance
    df = _column_selector(df_full, CBCLABCL_items + Dx_labels, drop=False)
    df_prev = _get_prevalance(df, diagnoses=Dx_labels)

    # Drop low-prevalance diagnoses right away from sparse dataset
    #        full list of diagnoses  -  all diagnoses with low prevalance
    Dx_labels_subset = list(set(df_prev.index) - set(df_prev[df_prev['Pt'] < threshold].index))

    # Prepare to iteratively repeat the process, as the N of various conditions may change as
    # we prune missing data and subsequently change the included column lists
    low_N = True
    while low_N:
        # Repeat dataset table subsetting (densely this time), compute prevalance, and drop low N
        df = _column_selector(df_full, CBCLABCL_items + Dx_labels_subset, drop=True)
        df_prev = _get_prevalance(df, diagnoses=Dx_labels_subset)

        # Grab a dataframe of the low-prevalance diagnoses...
        low_N_df = df_prev[df_prev['Pt'] < threshold]
        if low_N := (len(low_N_df.index) > 0):
            # ... and remove them from the set of consideration, then go again
            Dx_labels_subset = list(set(Dx_labels_subset)-set(low_N_df.index))

    # Report on prevalance table and overall dataset length
    if verbose:
        print(df_prev)
        print("Original Dataset Length:", len(df_full))
        print("Pruned Dataset Length:", len(df))

    return df, df_prev, Dx_labels_subset

    
def load_data(infile: PathLike, threshold: int=50, verbose: bool=True):
    """Grabs a parquet file, extracts the columns we want, removes NaNs, and returns"""
    # Grabs data file from disk, sets diagnostic labels to 0 or 1.
    df_full = pd.read_parquet(infile)
    df_full[Dx_labels_all] = df_full[Dx_labels_all].replace({2.0:1, 0.0:0})

    # Define column selector utility that we'll iteratively use to subsample the dataset.
    def _column_selector(df: pd.DataFrame, columns: list, drop: bool=False):        
        return df[columns].dropna(axis=0, how='any') if drop else df[columns]

    def _get_prevalance(df: pd.DataFrame, diagnoses: list):
        tmp = []
        for dx in diagnoses:
            vc = df[dx].value_counts()
            tmp += [{"Dx": dx, "HC": vc.loc[0.0], "Pt":vc.loc[1.0]}]
        return pd.DataFrame.from_dict(tmp).set_index('Dx').sort_values(by='Pt')

    # Initialize the dataset cleaning
    # Subset table based on all diagnoses, compute prevalance
    df = _column_selector(df_full, CBCLABCL_items + Dx_labels_all, drop=False)
    df_prev = _get_prevalance(df, diagnoses=Dx_labels_all)

    # Drop low-prevalance diagnoses right away from sparse dataset
    #        full list of diagnoses  -  all diagnoses with low prevalance
    Dx_labels_subset = list(set(df_prev.index) - set(df_prev[df_prev['Pt'] < threshold].index))

    # Prepare to iteratively repeat the process, as the N of various conditions may change as
    # we prune missing data and subsequently change the included column lists
    low_N = True
    while low_N:
        # Repeat dataset table subsetting (densely this time), compute prevalance, and drop low N
        df = _column_selector(df_full, CBCLABCL_items + Dx_labels_subset, drop=True)
        df_prev = _get_prevalance(df, diagnoses=Dx_labels_subset)

        # Grab a dataframe of the low-prevalance diagnoses...
        low_N_df = df_prev[df_prev['Pt'] < threshold]
        if low_N := (len(low_N_df.index) > 0):
            # ... and remove them from the set of consideration, then go again
            Dx_labels_subset = list(set(Dx_labels_subset)-set(low_N_df.index))

    # Report on prevalance table and overall dataset length
    if verbose:
        print(df_prev)
        print("Original Dataset Length:", len(df_full))
        print("Pruned Dataset Length:", len(df))

    return df, df_prev, Dx_labels_subset

def load_model(export_mod_dir:PathLike, modelname, degr:bool):
    ## Load a trained model from disk
        trained_model_dir = f"{export_mod_dir}/degraded_models" if degr else f"{export_mod_dir}/full_models"

        model = load(trained_model_dir + '/model_' + modelname + '.joblib')
        
        return model

def fit_models(df: pd.DataFrame, x_ids: list, y_ids: list, outdir: PathLike, export_mod_dir: PathLike, 
degr: bool,verbose: bool=True):
    """General purpose function for fitting callibrated classifiers for Dx from Survey data"""

    # Establish Models & Cross-Validation Strategy
    #   Base classifier: Random Forest. Rationale: non-parametric, has feature importance
    clf_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    #   CV: Stratified K-fold. Rationale: shuffle data, balance classes across folds
    cv = StratifiedKFold(shuffle=True, random_state=42)
    #   Top-level classifier: Calibrated CV classifier. Rationale: prioritizes maintaining class balances
    clf_calib = CalibratedClassifierCV(estimator=clf_rf, cv=cv)
    np.random.seed(42)

    # Create X (feature) matrix: grab relevant survey columns x rows from dataframe
    X = df[x_ids].values

    # Create empty list of learners
    learners = []
    summaries = []
    models_fitted = dict({'Dx':[],'model_name':[],'fit_model':[],'degr':[],'nqs':[]})
    # For every Dx that we want to predict...
    for y_name in y_ids:
        # Create the y (target) matrix/vector: grab relevant Dx rows from dataframe
        y = df[y_name].values.astype(int)
        
        # Get Pt and HC counts from dataframe, and initialize the learner object
        _, uc = np.unique(y, return_counts=True)

        if verbose:
            print(f"Dx: {y_name} | HC: {uc[0]} | Pt: {uc[1]}")

        current_learner = Learner(dx=y_name, hc_n=uc[0], dx_n=uc[1], x_ids=x_ids)
        # Set-up a CV loop (note, we use the same CV strategy both within the Calibrated CLF and here,
        #   resulting in nested-stratified-k-fold-CV)
        for idx_train, idx_test in cv.split(X, y):
            # Split the dataset into train and test sets
            X_tr = X[idx_train, :]
            y_tr = y[idx_train]

            X_te = X[idx_test, :]
            y_te = y[idx_test]

            # Fit the callibrated classifier on the training data
            clf_calib.fit(X_tr, y_tr)

            # Extract/Generate relevant data from (all internal folds of) the classifier...
            # Make predictions on the test set
            y_pred = clf_calib.predict(X_te)
            
            # Grab training and validation performance using the integrated scoring function (accuracy)
            y_pred_tr = clf_calib.predict(X_tr)
            current_learner.acc_train.append(accuracy_score(y_tr, y_pred_tr))
            current_learner.acc_valid.append(accuracy_score(y_te, y_pred))

            # Grab feature importance scores
            fis = [_.estimator.feature_importances_ for _ in clf_calib.calibrated_classifiers_]
            current_learner.fi.append(fis)

            # Grab the prediction probabilities
            current_learner.proba.append(clf_calib.predict_proba(X_te))
            current_learner.label.append(y_te)

            # Grab the prediction labels
            current_learner.f1.append(f1_score(y_te, y_pred))

            # Grab the sensitivity and specificity (i.e. recall of each of Dx and HC classes)
            report_dict = classification_report(y_te, y_pred, output_dict=True)
            current_learner.sen.append(report_dict['1']['recall'])
            current_learner.spe.append(report_dict['0']['recall'])

            # Grab the positive/negative likelihood ratios
            lrp, lrn = class_likelihood_ratios(y_te, y_pred)
            current_learner.LRp.append(lrp)
            current_learner.LRn.append(lrn)

        # Summarize current learner performance, save it, and get ready to go again!
        summaries += [current_learner.summary()]

        tmp_learner = {'Dx': current_learner.dx}

        means = np.mean(np.vstack(current_learner.fi), axis=0)
        for assessment, importance in zip(x_ids, means):
            tmp_learner[assessment] = importance
        learners += [tmp_learner]

        model_name = f"{y_name}_{'full' if not degr else 'degraded'}"

        models_fitted['Dx'].append(y_name)
        models_fitted['model_name'].append(model_name)
        models_fitted['fit_model'].append(clf_calib)
        models_fitted['degr'].append(degr)
        models_fitted['nqs'].append(len(x_ids))
        models_fitted['questions'].append((x_ids))

        print(model_name, 'number of questions',len(x_ids))

        del current_learner
        
    # Improve formatting of summaries and complete learners
    summaries = pd.concat(summaries).set_index('Dx')
    learners = pd.DataFrame.from_dict(learners).set_index('Dx')
    models_fitted = pd.DataFrame.from_dict(models_fitted).set_index('Dx')
    
    print('fit almost done')


    return learners, summaries, models_fitted

def evaluate_trained_models(df: pd.DataFrame, x_ids: list, y_ids: list, outdir: PathLike, export_mod_dir: PathLike,degr:bool, verbose: bool=True):
    """General purpose function for predictions from pretrained model"""
    
    # Create X (feature) matrix: grab relevant survey columns x rows from dataframe
    
    X = df[x_ids].values

    # Create empty list of learners
    learners = []
    summaries = []
    # For every Dx that we want to predict...
    for y_name in y_ids:
        # Create the y (target) matrix/vector: grab relevant Dx rows from dataframe
        print(y_name)
        y = df[y_name].values.astype(int)
        
        # Get Pt and HC counts from dataframe, and initialize the learner object
        _, uc = np.unique(y, return_counts=True)

        if verbose:
            print(f"Dx: {y_name} | HC: {uc[0]} | Pt: {uc[1]}")

        current_learner = Learner_eval(dx=y_name, hc_n=uc[0], dx_n=uc[1], x_ids=x_ids)
        print(current_learner)

        X_new = X[:]
        y_new = y[:]

        if degr==False:
            model_name = y_name +'_full'
        else:
            model_name = y_name +'_degraded'

        trained_model = load_model(export_mod_dir,model_name,degr=degr)
        print(f"Loading {model_name}, from {trained_model}")
        print(trained_model)

        y_pred = trained_model.predict(X_new)
        
        # Grab validation performance using the integrated scoring function (accuracy)

        current_learner.acc_valid.append(accuracy_score(y_new, y_pred))

        # Grab feature importance scores
        fis = [_.estimator.feature_importances_ for _ in trained_model.calibrated_classifiers_]
        current_learner.fi.append(fis)

        # Grab the prediction probabilities
        current_learner.proba.append(trained_model.predict_proba(X_new))
        current_learner.label.append(y_new)

        # Grab the prediction labels
        current_learner.f1.append(f1_score(y_new, y_pred))

        # Grab the sensitivity and specificity (i.e. recall of each of Dx and HC classes)
        report_dict = classification_report(y_new, y_pred, output_dict=True)
        current_learner.sen.append(report_dict['1']['recall'])
        current_learner.spe.append(report_dict['0']['recall'])

        # Grab the positive/negative likelihood ratios
        lrp, lrn = class_likelihood_ratios(y_new, y_pred)
        current_learner.LRp.append(lrp)
        current_learner.LRn.append(lrn)

        # Summarize current learner performance, save it, and get ready to go again!
        
        summaries += [current_learner.summary()]

        tmp_learner = {'Dx': current_learner.dx}

        means = np.mean(np.vstack(current_learner.fi), axis=0)
        for assessment, importance in zip(x_ids, means):
            tmp_learner[assessment] = importance
        learners += [tmp_learner]

        del current_learner
    
    # Improve formatting of summaries and complete learners
    summaries = pd.concat(summaries).set_index('Dx')
    learners = pd.DataFrame.from_dict(learners).set_index('Dx')

    

    return learners, summaries


def calculate_feature_importance(learners: list, x_ids: list, outdir: PathLike, number_of_questions: int=20, plot=True):
    """Visualizes feature importance and sorts values using two strategies: aggregate and top-N"""
    # Melt the dataframe into a long format for plotting
    df_learners_melt = learners.reset_index()
    df_learners_melt = df_learners_melt.melt(id_vars=['Dx'], value_vars=x_ids, value_name='importance')

    # Sort based on aggregate feature importance 
    sort_agg = (df_learners_melt
                .groupby('variable')
                .sum()
                .reset_index()
                .sort_values('importance')['variable'].values[::-1])

    if plot:
        # Plot a stacked-bar of feature importance across diagnoses, sorted by variable importance
        fig = px.bar(df_learners_melt, x="variable", y="importance", color="Dx",
                    title="Relative feature importance of survey questions across diagnoses",
                    labels={"variable": "CBCL Question",
                            "importance": "Relative Feature Importance",
                            "target": "Diagnosis"},
                    template='plotly_white')

        # Update plot to reflect sorting, and target number of questions
        fig.update_xaxes(categoryorder='array', categoryarray=sort_agg)
        fig.update_layout(width=1620, height=900, legend={'orientation': 'v','y':0.97,'x':0.84})
        fig.add_vline(x=sort_agg[number_of_questions+1], line_width=1, line_dash="dash", line_color="gray")
        fig.write_image(f'{outdir}/feature_importance_agg.png')


    # Redo sorting and plotting with the top-N approach
    # Initialize an empty list of questions, to be populated iteratively.
    topN = len(x_ids)
    item_relevance = np.empty((topN, len(x_ids)))

    # For each threshold of "we can only include N questions..."
    for n in range(topN):
        # Record the proportion of diagnoses for which a given question belongs
        item_relevance[n, :] = (learners[x_ids]
                                .rank(axis=1, ascending=False)
                                .apply(lambda x: x <= n+1)
                                .sum(axis=0))

    # Sort the prevalance table we just built, and apply it to the questions
    idx_topn = np.lexsort(item_relevance[::-1,:])[::-1]
    sort_topn = np.array(x_ids)[idx_topn]
    n_diagnoses = len(learners)

    if plot:
        # Plot a heatmap of feature importance across top-N selection for each diagnosis
        fig = px.imshow(item_relevance[:, idx_topn]*1.0/n_diagnoses,
                        x=sort_topn, y=np.arange(topN)+1,
                        labels={'x':'CBCL Question',
                                'y':'Number of Questions (N)',
                                'color':'Top-N Fraction'},
                        title="Consistency of question usefulness across diagnoses",
                        template='plotly_white')

        # Update plot to reflect the target number of questions
        fig.update_layout(width=1620, height=900)
        fig.add_vline(x=sort_topn[number_of_questions+1], line_width=1, line_dash="dash", line_color="gray")
        fig.write_image(f'{outdir}/feature_importance_topn.png')

    # Report the sorted list using both approaches
    sorted_importance_agg = sort_agg[0:number_of_questions]
    print(f"The {number_of_questions} cumulatively most predictive survey questions overall are:", ", ".join(sorted_importance_agg))
    
    sorted_importance_topn = sort_topn[0:number_of_questions]
    print(f"The {number_of_questions} most commonly useful survey questions overall are:", ", ".join(sorted_importance_topn))

    # Evaluate sorting consistency
    all_qs = list(set(list(sorted_importance_agg) + list(sorted_importance_topn)))
    diff = np.abs(number_of_questions-len(all_qs))
    frac = 1.0*diff/number_of_questions*100
    print(f"The two lists differ by {diff} / {number_of_questions} items ({frac:.2f}%)")

    # Compute the average position across the two methods and produce a 3rd and final sorting
    avg_rank = np.mean(np.where(sort_agg[:, None] == sort_topn), axis=0)
    sort_avg = sort_agg[np.argsort(avg_rank)]

    return sort_agg, sort_topn, sort_avg


def degrading_fit(df: pd.DataFrame, sorted_x_ids: list, y_ids: list, outdir: PathLike, export_mod_dir: PathLike,NQ:int=20, threads: int=4, degr: bool=True, verbose: bool=True):
    """Routine that wraps the `fit_models` function, gradually degrading the performance by reducing the x-set"""
    # Initialize some storage containers
    degraded_tuple = []
    futures = []

    with ProcessPoolExecutor(max_workers=threads) as pool:
        # For each number of questions (from the length of sorted_x_ids down to 1)...
        # for n_questions in range(len(sorted_x_ids))[::-1]:
        for n_questions in range(30)[::-1]:
            # Grab the first n_questions from the sorted list
            xi = sorted_x_ids[0:n_questions+1]

            # Add the diagnostic prediction models to the queue using this reduced x-set
            futures.append(pool.submit(fit_models, df, xi, y_ids, verbose=False, outdir=outdir, export_mod_dir=export_mod_dir, degr=degr))

    # Store the results as they come in
    for future in as_completed(futures):
        degraded_tuple.append(future.result())

    # Separate the learners and respective summaries
    degraded_learners = [dt[0] for dt in degraded_tuple]
    degraded_summaries = [dt[1] for dt in degraded_tuple]
    degraded_models = [dt[2] for dt in degraded_tuple]

    # Concatenate and return the learners and summaries
    degraded_summaries = pd.concat(degraded_summaries)
    degraded_learners = pd.concat(degraded_learners)
    degraded_models = pd.concat(degraded_models)
    print('deg.head',degraded_models.head())

    # only export 20 question model
    degraded_models_q20 = degraded_models[(degraded_models['nqs']==NQ)]
    print('deg_20.head',degraded_models_q20.head())


    return degraded_learners, degraded_summaries,degraded_models,degraded_models_q20


def export_model(models_fitted,export_mod_dir:PathLike):
    total_mods = len(models_fitted['model_name'])
    for m in range(total_mods):
        export_dir = f"{export_mod_dir}/degraded_models" if models_fitted['degr'][m] else f"{export_mod_dir}/full_models"
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        print(f"Exporting model to {export_dir}/model_{models_fitted['model_name'][m]}.joblib")
        dump(models_fitted['fit_model'][m],f"{export_dir}/model_{models_fitted['model_name'][m]}.joblib")


def run():
    parser = ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("outdir")
    parser.add_argument("export_mod_dir")
    parser.add_argument("-l", "--load_model", default=False, action="store_true")    
    parser.add_argument("-nq", "--number_of_questions", default=20, type=int)
    parser.add_argument("--random_state", default=42, type=int)
    parser.add_argument("-t", "--dx_threshold", default=150, type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-w", "--warnings", action="store_true")
    parser.add_argument("--n_threads", default=4, type=int)

    args = parser.parse_args()

    # Grab input values
    # infile = '../../data/converted/harmonized_allwaves.parquet'
    # outdir = '../../data/figures/'
    infile = args.infile
    outdir = args.outdir
    export_mod_dir = args.export_mod_dir
    threshold = args.dx_threshold
    verbose = args.verbose
    NQ = args.number_of_questions
    nt = args.n_threads
    load_mod = args.load_model

    np.random.seed(args.random_state)

    # Suppress the many warnings that come up when training degrading learners by default
    if not args.warnings:
        import warnings
        warnings.filterwarnings("ignore")

    # Load the dataset and remove incomplete subjects and under-represented diagnoses
    df, df_prev, Dx_labels_subset = load_data(infile, threshold=threshold, verbose=verbose)

    if load_mod==False:
    # Establish baseline prediction, and further remove diagnostic labels for which the models
    #  fail to make reasonable predictions (read as: make predictions to both classes;
    #  identifiable as diagnoses with a NaN for LR+)
        learners, summaries, models_fitted = fit_models(df, CBCLABCL_items, Dx_labels_subset, verbose=verbose,outdir=outdir,
        export_mod_dir=export_mod_dir,degr=False)

        Dx_labels_subset = list(set(Dx_labels_subset) - set(summaries[summaries['LR+'].isna()].index))
        learners = learners.loc[Dx_labels_subset]
        learners.to_parquet(f'{outdir}/learners.parquet')

        summaries = summaries.loc[Dx_labels_subset]
        summaries.to_parquet(f'{outdir}/summaries.parquet')
        print(summaries)

        models_fitted= models_fitted.loc[Dx_labels_subset]
        export_model(models_fitted,export_mod_dir)

        # Compute and plot feature importance, and then save results in a CSV file
        sorted_agg, sorted_topn, sorted_avg = calculate_feature_importance(learners, CBCLABCL_items, outdir, number_of_questions=NQ)
        importance = pd.DataFrame({'Aggregate': sorted_agg, 'Top-N': sorted_topn, 'Average': sorted_avg})
        importance.to_parquet(f'{outdir}/feature_importance.parquet')

        # Finally, redo the learning process with a degrading set of data, iteratively removing questions
        learners_deg, summaries_deg, models_deg, models_deg_20 = degrading_fit(df, sorted_avg, Dx_labels_subset, threads=nt,outdir=outdir,
        export_mod_dir=export_mod_dir, NQ=NQ, degr=True, verbose=verbose)
        
        summaries_deg.to_parquet(f'{outdir}/summaries_degraded.parquet')
        learners_deg.to_parquet(f'{outdir}/learners_degraded.parquet')
        
        export_model(models_deg,export_mod_dir)
        export_model(models_deg_20,export_mod_dir20)

        print('Summary Degraded',summaries_deg)
        print('models_deg_fitted',models_deg)
        print('models_deg_20',models_deg_20)



    else:
        # Load already fitted models and evaluate on new dataset

        # full_models_dir= export_mod_dir +'/full_models'
        # degraded_models_dir= export_mod_dir +'/degraded_models'

        learners, summaries = evaluate_trained_models(df, CBCLABCL_items, Dx_labels_subset,outdir=outdir,export_mod_dir=export_mod_dir,degr=True,verbose=verbose)

        Dx_labels_subset = list(set(Dx_labels_subset) - set(summaries[summaries['LR+'].isna()].index))

        # learners = learners.loc[Dx_labels_subset]
        learners.to_parquet(f"{outdir}/learners_fitted_degr.parquet" if degr else f"{outdir}/learners_fitted_full.parquet")

        # summaries = summaries.loc[Dx_labels_subset]
        summaries.to_parquet(f"{outdir}/summaries_fitted_degr.parquet" if degr else f"{outdir}/summaries_fitted_full.parquet")

        print(summaries)
    



if __name__ == "__main__":
    run()
