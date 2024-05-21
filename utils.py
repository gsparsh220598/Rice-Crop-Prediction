import os
import pickle
import subprocess
import time
from typing import List
from scipy import stats
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
)

from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings("ignore")
NUM_TRIALS = 1000
scv = StratifiedKFold(n_splits=5)


def prep_s2_data(data):
    """
    Preprocess Sentinel-2 satellite data to compute vegetation indices.

    This function calculates NDVI, SAVI, and EVI for each time point and sample in the input data.
    It returns the processed data reshaped to the desired format.

    Parameters:
    - data: A numpy array of shape (samples, bands, time_points). Each sample contains the following bands:
      - Red (index 0)
      - Green (index 1)
      - Blue (index 2)
      - NIR (index 3)
    - time: A list or array of time points corresponding to the time dimension of the data.

    Returns:
    - A numpy array of shape (samples, 3, 10) containing the computed NDVI, SAVI, and EVI values for each sample.
    """
    times = np.array(
        [
            "2021-11-11T03:19:49.024000000",
            "2021-11-16T03:20:11.024000000",
            "2021-11-21T03:20:29.024000000",
            "2021-11-26T03:20:51.024000000",
            "2021-12-01T03:21:09.024000000",
            "2021-12-06T03:21:21.024000000",
            "2021-12-11T03:21:29.024000000",
            "2021-12-16T03:21:41.024000000",
            "2021-12-21T03:21:39.024000000",
            "2021-12-26T03:21:41.024000000",
            "2021-12-31T03:21:29.024000000",
            "2022-01-05T03:21:31.024000000",
            "2022-01-10T03:21:09.024000000",
            "2022-01-20T03:20:39.024000000",
            "2022-01-30T03:19:49.024000000",
            "2022-02-04T03:19:31.024000000",
            "2022-02-09T03:18:59.024000000",
            "2022-02-14T03:18:31.024000000",
            "2022-02-19T03:17:49.024000000",
            "2022-02-24T03:17:31.024000000",
            "2022-03-01T03:16:49.024000000",
            "2022-03-06T03:16:21.024000000",
            "2022-03-11T03:15:39.024000000",
            "2022-03-16T03:15:41.024000000",
            "2022-03-21T03:15:39.024000000",
            "2022-03-26T03:15:41.024000000",
            "2022-03-31T03:15:39.024000000",
            "2022-04-05T03:15:41.024000000",
            "2022-04-10T03:15:39.024000000",
            "2022-04-15T03:15:41.024000000",
            "2022-04-20T03:15:39.024000000",
            "2022-04-25T03:15:41.024000000",
            "2022-04-30T03:15:29.024000000",
            "2022-05-10T03:15:39.024000000",
            "2022-05-15T03:15:41.024000000",
            "2022-05-20T03:15:39.024000000",
            "2022-05-25T03:15:51.024000000",
            "2022-05-30T03:15:39.024000000",
            "2022-06-04T03:15:51.024000000",
            "2022-06-09T03:15:39.024000000",
            "2022-06-14T03:15:51.024000000",
            "2022-06-19T03:15:19.024000000",
            "2022-06-24T03:15:51.024000000",
            "2022-06-29T03:15:29.024000000",
            "2022-07-04T03:15:31.024000000",
            "2022-07-09T03:15:29.024000000",
            "2022-07-14T03:15:51.024000000",
            "2022-07-19T03:15:29.024000000",
            "2022-07-24T03:15:31.024000000",
            "2022-07-29T03:15:29.024000000",
            "2022-08-03T03:15:31.024000000",
            "2022-08-08T03:15:19.024000000",
            "2022-08-13T03:15:31.024000000",
            "2022-08-18T03:15:19.024000000",
            "2022-08-23T03:15:31.024000000",
        ]
    )
    time = [t.split("T")[0][:7] for t in times]
    L = 0.5
    prep_data = []
    for d in range(data.shape[0]):
        ds = pd.DataFrame()
        ds["time"] = time
        ds["red"] = data[d, 0, :]
        ds["green"] = data[d, 1, :]
        ds["blue"] = data[d, 2, :]
        ds["nir"] = data[d, 3, :]
        ds["ndvi"] = (ds.nir - ds.red) / (ds.nir + ds.red)
        ds["savi"] = (1 + L) * (ds.nir - ds.red) / (ds.nir + ds.red + L)
        ds["evi"] = 2.5 * (
            (ds.nir - ds.red) / (ds.nir + 6 * ds.red - 7.5 * ds.blue) + 1
        )
        sample = (
            ds.groupby("time")
            .mean(numeric_only=True)
            .reset_index()[["ndvi", "savi", "evi"]]
            .to_numpy()
            .reshape(3, 10)
        )  # CHANGE FEATURES HERE
        prep_data.append(sample)
    return np.array(prep_data)


def prep_s1_data(data):
    """
    Preprocess Sentinel-1 SAR data to compute Radar Vegetation Index (RVI) and NDVI-SAR.

    This function calculates RVI and NDVI-SAR for each sample in the input data. It returns the original
    data with the additional computed indices.

    Parameters:
    - data: A numpy array of shape (samples, bands, time_points). Each sample contains the following bands:
      - VV (index 0)
      - VH (index 1)

    Returns:
    - A numpy array with the original data and the computed RVI and NDVI-SAR values added as additional bands.
    """
    vv = data[:, 0, :]
    vh = data[:, 1, :]
    q = vh / vv
    n = q * (q + 3)
    d = (q + 1) ** 2
    rvi = n / d  # CALCULATION OF Radar Vegetation Index
    ndvi_sar = (vh - vv) / (vh + vv)
    if len(data.shape) == 3:
        rvi = rvi.reshape(data.shape[0], 1, data.shape[2])
        ndvi_sar = ndvi_sar.reshape(data.shape[0], 1, data.shape[2])
    else:
        rvi = rvi.reshape(data.shape[0], 1)
        ndvi_sar = ndvi_sar.reshape(data.shape[0], 1)
    rvi = np.nan_to_num(rvi, nan=0)
    new_data = np.concatenate((data, rvi, ndvi_sar), axis=1).copy()
    return new_data


def create_col_names(
    features_sar=["VV", "VH", "RVI", "NDVI_SAR"],
    features_o=["ndvi", "savi", "evi"],
    timesteps=[52, 10],
) -> dict:
    """
    Create column names for SAR and optical features over specified timesteps.

    This function generates a dictionary mapping column indices to their corresponding
    feature names, based on the given SAR and optical feature lists and their respective
    timesteps.

    Parameters:
    - features_sar (list): List of SAR feature names.
    - features_o (list): List of optical feature names.
    - timesteps (list): A list containing the number of timesteps for SAR and optical
                        features respectively.

    Returns:
    - dict: A dictionary where keys are column indices and values are the feature names.

    Example:
    >>> create_col_names()
    {0: 'VV_0', 1: 'VV_1', 2: 'VV_2', ..., 51: 'NDVI_SAR_51', 52: 'ndvi_0', ..., 81: 'evi_9'}
    """
    cols_sar = [f"{feat}_{t}" for feat in features_sar for t in range(0, timesteps[0])]
    cols_o = [f"{feat}_{t}" for feat in features_o for t in range(0, timesteps[1])]
    cols = cols_sar + cols_o
    dicts = {
        t: c
        for t, c in zip(
            range(0, len(features_sar) * timesteps[0] + len(features_o) * timesteps[1]),
            cols,
        )
    }
    return dicts


def proc_pipeline(s1data, s2data) -> pd.DataFrame:
    """Preprocesses data for a pipeline.

    Args:
    s1data (array-like or None): The input data for the first part of the pipeline.
    s2data (array-like or None): The input data for the second part of the pipeline.

    Returns:
    pandas.DataFrame: A DataFrame containing the preprocessed data from both s1data and s2data, if both are provided. If only s1data is provided, returns a DataFrame containing the preprocessed data from s1data. If only s2data is provided, returns a DataFrame containing the preprocessed data from s2data. If both s1data and s2data are None, returns None.
    """
    if s1data is not None:
        data_s1 = s1data[:, :, :52].copy()  # CHANGE HERE FOR TIMESTEPS
        data_s1 = prep_s1_data(data_s1)
        data_s1 = pd.DataFrame(
            data_s1.reshape(data_s1.shape[0], data_s1.shape[1] * data_s1.shape[2])
        )  # .dropna(axis=1)
        if s2data is not None:
            data_s2 = prep_s2_data(s2data)
            data_s2 = pd.DataFrame(
                data_s2.reshape(data_s2.shape[0], data_s2.shape[1] * data_s2.shape[2])
            )
            complete_df = pd.concat([data_s1, data_s2], axis=1, ignore_index=True)
            return complete_df
        else:
            return data_s1
    else:
        data_s2 = prep_s2_data(s2data)
        data_s2 = pd.DataFrame(
            data_s2.reshape(data_s2.shape[0], data_s2.shape[1] * data_s2.shape[2])
        )
        return data_s2


def correlation_plot(dataframe, run=None):
    """
    Generates a correlation heatmap plot using Seaborn.

    Parameters:
    - dataframe: pandas DataFrame containing numeric columns.

    Returns:
    - None (displays the plot).
    """
    import wandb

    corr_matrix = dataframe.corr()
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", linewidths=0.5, annot_kws={"size": 6}, ax=ax
    )
    ax.set_title("Correlation Heatmap")

    # Log the plot directly to W&B
    run.log({"correlation_heatmap": wandb.Plotly(fig)})


def make_violinplot(df, run=None):
    """
    Generate a violin plot for each feature in the DataFrame and log it to Weights & Biases.

    This function creates a violin plot for each feature variable grouped by the target
    class and logs the plot to a Weights & Biases run instance. Violin plots are useful
    for visualizing the distribution of the data across different categories.

    Parameters:
    - df: pandas DataFrame containing the feature variables and a 'Target' column.
          The 'Target' column should have the classes to group by.
    - run: Weights & Biases run instance. If provided, the plot will be logged to this
           run. Defaults to None.

    Returns:
    - None
    """
    import wandb

    traces = []

    for feature_name, feature_values in df.items():
        trace = go.Violin(
            y=feature_values,
            x=df["Target"],
            name=feature_name,
            box_visible=True,
            meanline_visible=True,
        )
        traces.append(trace)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Violin Plot of Feature Variables by Target Class",
        yaxis_title="Feature Value",
        xaxis_title="Rice or Non-Rice",
    )
    # fig.show()
    run.log({"violin_plot": wandb.Plotly(fig)})


def tsne_plot(dataframe, target_column, run):
    """
    Generates a t-SNE plot using Plotly with hover information to visualize high-dimensional data.

    Parameters:
    - dataframe: pandas DataFrame containing numeric columns.
    - target_column: Name of the target column in the DataFrame.
    """
    import wandb

    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    hover_text = ["Index: {}".format(index) for index in dataframe.index]
    fig = px.scatter(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        color=y,
        labels={"x": "t-SNE Component 1", "y": "t-SNE Component 2"},
        color_continuous_scale="viridis",
        title="t-SNE Plot",
        hover_name=hover_text,
    )
    fig.update_coloraxes(colorbar_title=target_column)
    fig.update_layout(showlegend=False)
    run.log({"tsne_plot": wandb.Plotly(fig)})


def rank_seeds(clf, X_train, y_train, topk=10) -> List[int]:
    """
    Rank random seeds based on cross-validation accuracy for a given classifier.

    This function performs multiple cross-validation runs using different random seeds
    and ranks the seeds based on the average cross-validation accuracy. The top-k seeds
    with the lowest accuracy are returned.

    Parameters:
    - clf: The classifier to be evaluated. It must implement the 'fit' and 'predict' methods.
    - X_train: Training data features as a pandas DataFrame or numpy array.
    - y_train: Training data labels as a pandas Series or numpy array.
    - topk: The number of top seeds to return. Defaults to 10.

    Returns:
    - seeds: A list of the top-k seeds with the lowest cross-validation accuracy scores.
    """
    sd = {}
    for seed in tqdm(range(0, NUM_TRIALS)):
        scv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        score = cross_val_score(
            clf, X_train, y_train, cv=scv.split(X_train, y_train), scoring="accuracy"
        ).mean()
        sd.update({seed: score})
    seeds = (
        pd.DataFrame(list(sd.items()), columns=["Seed", "Score"])
        .sort_values("Score", ascending=True)
        .index[0:topk]
        .values.tolist()
    )
    return seeds


def score_clf(clf, X_train, y_train, seeds2) -> int:
    """
    Evaluate a classifier using multiple random seeds for cross-validation.

    This function performs cross-validation using different random seeds to evaluate
    the performance of a given classifier. It returns the mean accuracy score and
    the standard deviation of the accuracy scores.

    Parameters:
    - clf: The classifier to be evaluated. It must implement the 'fit' and 'predict' methods.
    - X_train: Training data features as a pandas DataFrame or numpy array.
    - y_train: Training data labels as a pandas Series or numpy array.
    - seeds2: A list of random seeds to be used for cross-validation.

    Returns:
    - t_dist_lower_bound: The lower bound of the t-distribution for the mean accuracy score.
    """
    scores = []
    for seed in seeds2:
        scv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        score = cross_val_score(
            clf, X_train, y_train, cv=scv.split(X_train, y_train), scoring="accuracy"
        ).mean()
        scores.append(score * 100)
    return t_dist_lower_bound(scores)


def plot_fi(feat_imp, run=None):
    """
    Generates and logs a feature importance plot using Plotly.

    This function creates a bar plot showing the importance of features,
    sorts the features by their importance scores in descending order,
    and logs the plot to a Weights & Biases run instance if provided.

    Parameters:
    - feat_imp: Dictionary containing feature names as keys and their importance scores as values.
    - run: Weights & Biases run instance. If provided, the plot will be logged to this run. Defaults to None.

    Returns:
    - None
    """
    import wandb

    sorted_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
    features = [item[0] for item in sorted_features]
    importance_scores = [item[1] for item in sorted_features]

    fig = go.Figure(data=[go.Bar(x=features, y=importance_scores)])
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Feature",
        yaxis_title="Importance Score",
    )
    run.log({"feature_importance": wandb.Plotly(fig)})


# NESTED CV STRATEGY TO SCORE THE VALIDATION SET
def nested_cv(
    pipe,
    X,
    y,
    grid,
    splits,
    iters=5,
    seed=42,
    metrics="accuracy",
) -> dict:
    """
    Perform nested cross-validation to evaluate a pipeline with hyperparameter tuning.

    This function performs nested cross-validation, combining an inner cross-validation
    for hyperparameter tuning using RandomizedSearchCV and an outer cross-validation
    for model evaluation. The best estimators and their scores are returned.

    Parameters:
    - pipe: The machine learning pipeline to be evaluated.
    - X: Features dataset as a pandas DataFrame or numpy array.
    - y: Target labels as a pandas Series or numpy array.
    - grid: Dictionary with parameters names (str) as keys and lists of parameter
            settings to try as values, used in RandomizedSearchCV.
    - splits: Number of folds for cross-validation.
    - iters: Number of parameter settings that are sampled in RandomizedSearchCV. Defaults to 5.
    - seed: Random state seed for reproducibility. Defaults to 42.
    - metrics: Scoring metric to evaluate the model performance. Defaults to 'accuracy'.

    Returns:
    - A dictionary containing:
        - "model_params": List of best estimators from each outer fold.
        - "accuracy": List of accuracy scores from each outer fold.
    """
    inner_cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    outer_cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    clf = RandomizedSearchCV(
        estimator=pipe,
        n_iter=iters,
        param_distributions=grid,
        cv=inner_cv,
        scoring=metrics,
        n_jobs=-1,
        random_state=seed,
    )
    scores = cross_validate(
        clf, X=X, y=y, cv=outer_cv, scoring=metrics, return_estimator=True
    )
    model_params = [e.best_estimator_ for e in scores["estimator"]]
    return {
        "model_params": model_params,
        "accuracy": scores["test_score"],
    }


# FUNCTION TO RUN MULTIPLE NESTED CV TRIALS
def run_cvs(pipe, X, y, grid, splits=10, iters=5, metrics="accuracy") -> pd.DataFrame:
    """
    Run multiple nested cross-validation trials and store results.

    This function performs nested cross-validation multiple times, each with a different
    random seed. It returns a DataFrame containing the results of each trial, including
    the best model parameters and the accuracy scores.

    Parameters:
    - pipe: The machine learning pipeline to be evaluated.
    - X: Features dataset as a pandas DataFrame or numpy array.
    - y: Target labels as a pandas Series or numpy array.
    - grid: Dictionary with parameter names (str) as keys and lists of parameter
            settings to try as values, used in RandomizedSearchCV.
    - splits: Number of folds for cross-validation. Defaults to 10.
    - iters: Number of parameter settings sampled in RandomizedSearchCV. Defaults to 5.
    - metrics: Scoring metric to evaluate the model performance. Defaults to 'accuracy'.

    Returns:
    - cv_results: DataFrame containing the results of each trial.
    """
    cv_results = pd.DataFrame()
    row_res = {}

    for i in tqdm(range(NUM_TRIALS)):  # ITERATE THROUGH NUMBER OF TRIALS
        row_res["seed"] = i
        cv_res = nested_cv(
            pipe, X, y, grid=grid, splits=splits, iters=iters, seed=i, metrics=metrics
        )
        row_res.update(cv_res)
        temp_res = pd.DataFrame(row_res, columns=list(row_res.keys()))
        cv_results = pd.concat([cv_results, temp_res], axis=0, ignore_index=True)
        row_res = {}
    return cv_results


# FUNCTION TO FIND THE WORST PERFORMING TRIAL OUT OF ALL THE TRIALS
def find_worst_seeds(res, topk=5) -> List[int]:
    """
    Identify the worst performing seeds based on cross-validation results.

    This function sorts the seeds by their average accuracy scores in ascending order
    and returns the seeds corresponding to the lowest scores.

    Parameters:
    - res: DataFrame containing cross-validation results, including accuracy scores
           and seeds.
    - topk: Number of worst performing seeds to return. Defaults to 5.

    Returns:
    - seeds: List of the worst performing seeds.
    """
    seeds = []
    for seed in (
        res.groupby("seed")["accuracy"]
        .mean()
        .sort_values(ascending=True)
        .index[0:topk]
        .values
    ):
        seeds.append(seed)
    return seeds


# MAKING VOTING CLASSIFIER USING A LIST OF MODELS
def make_vc(search_list, name_list) -> VotingClassifier:
    """
    Create a voting classifier from a list of models.

    This function takes a list of models and their corresponding names and creates a
    VotingClassifier that uses soft voting.

    Parameters:
    - search_list: List of models to be included in the voting classifier.
    - name_list: List of names corresponding to the models in search_list.

    Returns:
    - VotingClassifier: A VotingClassifier using soft voting.
    """
    estimator_list = [(str(n), s) for n, s in zip(name_list, search_list)]
    return VotingClassifier(estimators=estimator_list, voting="soft")


def evaluate(clf, X_train, y_train, X_test, y_test, run=None):
    """
    Evaluate a classifier and log the confusion matrix to Weights & Biases.

    This function fits a classifier on the training data, makes predictions on the test
    data, and logs the confusion matrix to a Weights & Biases run.

    Parameters:
    - clf: The classifier to be evaluated.
    - run: The Weights & Biases run instance.
    - combo: Name of the model combination used (for logging purposes).
    - X_train: Training data features.
    - y_train: Training data labels.
    - X_test: Test data features.
    - y_test: Test data labels.

    Returns:
    - None
    """
    import wandb

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    cm = wandb.plot.confusion_matrix(
        y_true=y_test, preds=predictions, class_names=["RICE", "NON-RICE"]
    )
    run.log({f"confusion_matrix": cm})


def predict_submission(clf, proc_pipe, X, y, use_s2=False):
    """
    Predict labels for submission using a fitted classifier.

    This function loads test data, processes it using a provided pipeline, fits the
    classifier on the training data, and predicts labels and probabilities for the
    test data.

    Parameters:
    - clf: The classifier to be used for predictions.
    - proc_pipe: Preprocessing pipeline to transform the test data.
    - X: Training data features.
    - y: Training data labels.
    - use_s2: Boolean indicating whether to use additional data (sentinel2). Defaults to False.

    Returns:
    - submission_predictions: Predicted labels for the test data.
    - submission_probs: Predicted probabilities for the test data.
    """
    testdfs1 = np.load("sar_data_test.npy")
    if use_s2:
        testdfs2 = np.load("sentinel2_data_test.npy")
        proc_testdf = proc_pipeline(testdfs1, testdfs2)
    else:
        proc_testdf = proc_pipeline(testdfs1, None)
    X_sub = proc_pipe.transform(proc_testdf)
    clf.fit(X, y)
    submission_predictions = clf.predict(X_sub)
    submission_probs = clf.predict_proba(X_sub)
    return submission_predictions, submission_probs


# THE FUNCTION TO CREATE VOTING CLASSIFIER BASED ON PERFORMANCE ON WORST PERFORMING SEED
def score_worst_seeds(clf, params, X, y, seeds, iters=100):
    """
    Create and evaluate a voting classifier based on the worst performing seeds.

    This function performs nested cross-validation for each of the worst performing
    seeds, creates a VotingClassifier using the best models from each trial, and
    evaluates the ensemble.

    Parameters:
    - clf: The base classifier to be used in nested cross-validation.
    - params: Dictionary with parameter names (str) as keys and lists of parameter
              settings to try as values, used in RandomizedSearchCV.
    - X: Training data features.
    - y: Training data labels.
    - seeds: List of worst performing seeds.
    - iters: Number of parameter settings sampled in RandomizedSearchCV. Defaults to 100.

    Returns:
    - A tuple containing:
        - VotingClassifier: An ensemble classifier using the models from the worst performing seeds.
    """
    model_ls = []
    valid_scores = []

    for seed in tqdm(seeds):
        cv_res = nested_cv(
            clf,
            X,
            y,
            grid=params,
            splits=5,
            iters=iters,
            seed=seed,
            metrics="f1_weighted",
        )
        cv_models = cv_res["model_params"]
        model = make_vc(cv_models, list([i for i in range(0, 5)]))
        score = cv_res["accuracy"].mean()
        model_ls.append(model)
        valid_scores.append(score * 100)

    fs = t_dist_lower_bound(valid_scores, confidence_level=0.99)
    print(f"The mean accuracy for the {len(seeds)} worst seeds is {fs}")
    return make_vc(model_ls, seeds), fs


# use itertools to make N>2 combinations of rf, svm, xgb, lgbm, mlp
def score_ensemble(run, model_dict, X_train, y_train, X_test, y_test, seeds, N=2):
    """
    Generate combinations of models, evaluate them, and log results to Weights & Biases.

    Parameters:
    - run: Weights & Biases run instance.
    - model_dict: Dictionary with model names as keys and model instances as values.
    - X_train: Training data features.
    - y_train: Training data labels.
    - seeds: List of random seeds for reproducibility.
    - N: Number of models to combine. Default is 2.

    Returns:
    - None
    """
    from itertools import combinations

    # Generate all combinations of models with length N
    model_list = list(model_dict.keys())
    combos = [list(combo) for combo in combinations(model_list, N)]

    for combo in combos:
        search_combo = [model_dict[c] for c in combo]
        vclf = make_vc(search_combo, combo)  # Create a voting classifier from the combo
        score = score_clf(vclf, X_train, y_train, seeds)  # Score the voting classifier
        run.log({f"score_ensemble": score})  # Log the score to Weights & Biases
        evaluate(
            vclf, X_train, y_train, X_test, y_test, run
        )  # Evaluate and log confusion matrix


def sample_dict(original_dict, sample_size) -> dict:
    """
    Function to sample elements from lists in a dictionary.

    :param original_dict: Dictionary with lists as values
    :param sample_size: Number of elements to sample from each list
    :return: New dictionary with sampled elements
    """
    new_dict = {}
    for key, value_list in original_dict.items():
        # Ensure sample size does not exceed the length of the list
        sample_size = min(sample_size, len(value_list))
        new_dict[key] = random.sample(value_list, sample_size)[0]
    return new_dict


def load_cache(filename) -> set:
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return set()


def save_cache(cache, filename):
    with open(filename, "wb") as f:
        pickle.dump(cache, f)


def hashable_dict(d) -> tuple:
    """
    Converts a dictionary into a hashable tuple.

    Parameters:
    d (dict): The input dictionary.

    Returns:
    tuple: A hashable representation of the dictionary.
    """
    return tuple(sorted(d.items()))


def cache(func, cache_file="exp_cache.pkl"):
    """
    A decorator that caches the arguments of the function call.

    Parameters:
    func (function): The function to be cached.

    Returns:
    function: The wrapped function with caching.
    """
    cache_set = load_cache(cache_file)

    def cached_func(*args):
        # Convert any dictionary arguments to a hashable type
        hashable_args = tuple(
            hashable_dict(arg) if isinstance(arg, dict) else arg for arg in args
        )
        if hashable_args in cache_set:
            print(f"Skipping {args} as it is already cached.")
            return
        cache_set.add(hashable_args)
        save_cache(cache_set, cache_file)
        print(f"Caching {args}")
        return func(*args)

    return cached_func


def run_experiment(args_dict):
    # args_dict = sample_dict(experiment_space, 1)
    cmd = "python main.py"
    for k, v in args_dict.items():
        cmd += " " + str(v)
    subprocess.run(cmd, shell=True)
    time.sleep(30)


def t_dist_lower_bound(arr, confidence_level=0.99) -> int:
    """
    Calculate the lower bound of a t-distributed random variable.

    Parameters:
    arr (list or array): The scores of the seeds.
    confidence_level (float): The desired confidence level. Default is 0.99.

    Returns:
    float: The lower bound of the t-distributed random variable.
    """
    mean, std_dev, sample_size = np.mean(arr), np.std(arr), len(arr)
    if sample_size == 1:
        return mean
    df = sample_size - 1  # Degrees of freedom
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(alpha / 2, df)
    margin_of_error = t_critical * (std_dev / (sample_size**0.5))
    lower_bound = mean + margin_of_error
    return lower_bound


def get_keys_above_threshold(d, threshold=95.0):
    sorted_items = sorted(d.items(), key=lambda item: item[1], reverse=True)
    keys_above_threshold = [key for key, value in sorted_items if value > threshold]
    if len(keys_above_threshold) > 4:
        keys_above_threshold = keys_above_threshold[:4]
    return keys_above_threshold
