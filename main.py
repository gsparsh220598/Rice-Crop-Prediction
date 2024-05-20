import os
import pandas as pd
import numpy as np
import wandb
import argparse
from dotenv import load_dotenv

from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.decomposition import KernelPCA, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.neural_network import MLPClassifier


from sklearn.feature_selection import RFECV
from sklearn.preprocessing import SplineTransformer, LabelEncoder
from utils import (
    score_ensemble,
    proc_pipeline,
    create_col_names,
    make_violinplot,
    tsne_plot,
    correlation_plot,
    score_clf,
    rank_seeds,
    plot_fi,
    run_cvs,
    find_worst_seeds,
    score_worst_seeds,
)
from hyperparams import (
    lr_params,
    mlp_params,
    rf_params,
    svm_params,
    xgb_params,
    lgbm_params,
)

import warnings

warnings.filterwarnings("ignore")
load_dotenv()

PROJECT_NAME = os.getenv("PROJECT_NAME")
ENTITY = os.getenv("ENTITY")
NUM_TRIALS = os.getenv("NUM_TRIALS")
SEED = os.getenv("SEED")
scv = StratifiedKFold(n_splits=5)

wandb.login()

# create argparser arguments
parser = argparse.ArgumentParser()
parser.add_argument("pca", type=str, default="yes")
parser.add_argument("nfeats", type=int, default=50)
parser.add_argument("knots", type=int, default=8)
parser.add_argument("degree", type=int, default=4)
parser.add_argument("ncomps", type=int, default=3)
parser.add_argument("kernel", type=str, default="cosine")
parser.add_argument("extrap", type=int, default="periodic")
parser.add_argument("mlp_iters", type=int, default=10)
parser.add_argument("rf_iters", type=int, default=200)
parser.add_argument("xgb_iters", type=int, default=200)
parser.add_argument("lgbm_iters", type=int, default=200)
parser.add_argument("svm_iters", type=int, default=200)
parser.add_argument("Ncombs", type=int, default=2)
args = parser.parse_args()

# initialize wandb run
if args.log == "yes":
    run = wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY,
        job_type="experiment",
        notes=f"Using SAR dataset to classify a piece of land from Rice to Non-Rice",
        tags=[
            "SAR_dataset",
            f"pca{args.pca}",
            f"nfeats{args.nfeats}",
            f"knots{args.knots}",
            f"degree{args.degree}",
            f"ncomps{args.ncomps}",
            f"kernel{args.kernel}",
            f"extrap{args.extrap}",
            f"mlp_iters{args.mlp_iters}",
            f"rf_iters{args.rf_iters}",
            f"xgb_iters{args.xgb_iters}",
            f"lgbm_iters{args.lgbm_iters}",
            f"svm_iters{args.svm_iters}",
            f"Ncombs{args.Ncombs}",
        ],
    )


###IMPORT DATA####
df = pd.read_csv("Crop_Location_Data_20221201.csv")
s1_data = np.load("sar_data.npy")  # Sentinel-1 RTC Corrected Data
s2_data = np.load("sentinel2_data_train.npy")[:, :4, :55]
s2_data[s2_data == 0] = np.nan

###PRE-PROCESSING PIPELINE###
if args.pca == "yes":
    proc_pipe1 = Pipeline(
        [
            ("thresh", VarianceThreshold()),  # Remove columns with constant features
            (
                "spline",
                SplineTransformer(
                    n_knots=args.knots, degree=args.degree, extrapolation=args.extrap
                ),
            ),
            ("scale", StandardScaler()),  # Scale the data
            (
                "select_feats1",
                SelectKBest(f_classif, k=100),
            ),  # select top 100 features using f_classif strategy
            (
                "select_feats2",
                RFECV(
                    XGBClassifier(random_state=SEED),
                    step=0.05,
                    min_features_to_select=args.nfeats,
                ),
            ),
            (
                "tsne",
                KernelPCA(
                    n_components=args.ncomps, kernel=args.kernel, random_state=42
                ),
            ),
        ]
    )
else:
    proc_pipe1 = Pipeline(
        [
            ("thresh", VarianceThreshold()),  # Remove columns with constant features
            (
                "spline",
                SplineTransformer(
                    n_knots=args.knots, degree=args.degree, extrapolation=args.extrap
                ),
            ),
            ("scale", StandardScaler()),  # Scale the data
            (
                "select_feats1",
                SelectKBest(f_classif, k=100),
            ),  # select top 100 features using f_classif strategy
            (
                "select_feats2",
                RFECV(
                    XGBClassifier(random_state=SEED),
                    step=0.05,
                    min_features_to_select=args.nfeats,
                ),
            ),
        ]
    )

proc_pipe = proc_pipe1

###PRE-PROCESSING PIPELINE###
le = LabelEncoder()
y = le.fit_transform(df["Class of Land"])
complete_df = proc_pipeline(s1_data, None)
dicts = create_col_names()
complete_df.rename(columns=dicts, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(
    complete_df, y, test_size=0.45, shuffle=True, random_state=807410395
)
X_train = proc_pipe1.fit_transform(X_train, y_train)
X_test = proc_pipe1.transform(X_test)
sel_feats = proc_pipe1.get_feature_names_out().tolist()
df_ = pd.DataFrame(X_train, columns=sel_feats)
df_["Target"] = y_train

X = proc_pipe.fit_transform(complete_df, y)
###VIOLIN PLOT###
make_violinplot(df_, run)

###CORRELATION PLOT###
correlation_plot(df_, run)

###TSNE PLOT###
tsne_plot(df_, target_column="Target", run=run)

wandb_logs = {}
###RANK SEEDS###
clf = LogisticRegression(max_iter=200, random_state=SEED)
cv_seeds = rank_seeds(clf)
wandb_logs["cv_seeds"] = cv_seeds
# seeds2=[1462, 9692, 5749, 1776, 6032, 7865, 4058, 7713, 7119, 1147]
score = score_clf(clf, X_train, y_train, cv_seeds)
wandb_logs["cv_score"] = score

###FEATURE IMPORTANCES PLOT###
model = clf.fit(X_train, y_train)
coefs = model.coef_.tolist()[0]
feat_imp = {c: i for c, i in zip(sel_feats, [abs(c) for c in coefs])}
plot_fi(feat_imp, run)

# BLOCK TO FIND THE WORST PERFORMING SEED ON THE DATA
clf = LogisticRegression(max_iter=200, random_state=SEED)
res_ridge = run_cvs(
    clf, X_train, y_train, lr_params, splits=5, iters=1, metrics="f1_weighted"
)
seeds = find_worst_seeds(res_ridge, topk=10)
wandb_logs["ncv_seeds"] = seeds

###MLP CLASSIFIER###
clf = MLPClassifier(
    batch_size=32, solver="adam", random_state=42, max_iter=5000, n_iter_no_change=10
)
model_mlp, mlp_score, mlp_sd = score_worst_seeds(
    clf, mlp_params, X_train, y_train, seeds, iters=args.mlp_iters
)
wandb_logs["mlp_score"] = {
    "mean": mlp_score["mean"],
    "std": mlp_score["std"],
}

###RANDOM FOREST CLASSIFIER###
clf = RandomForestClassifier(random_state=420)
model_rf, rf_score, rf_sd = score_worst_seeds(
    clf, rf_params, X_train, y_train, seeds, iters=args.rf_iters
)
wandb_logs["rf_score"] = {
    "mean": rf_score["mean"],
    "std": rf_score["std"],
}


###SUPPORT VECTOR CLASSIFIER###
clf = SVC(random_state=420, probability=True)
model_svm, svm_score, svm_sd = score_worst_seeds(
    clf, svm_params, X_train, y_train, seeds, iters=args.svm_iters
)
wandb_logs["svm_score"] = {
    "mean": svm_score["mean"],
    "std": svm_score["std"],
}


###XGBOOST CLASSIFIER###
clf = XGBClassifier(random_state=420)
model_xgb, xgb_score, xgb_sd = score_worst_seeds(
    clf, xgb_params, X_train, y_train, seeds, iters=args.xgb_iters
)
wandb_logs["xgb_score"] = {
    "mean": xgb_score["mean"],
    "std": xgb_score["std"],
}

###LIGHTGBM CLASSIFIER###
clf = LGBMClassifier(random_state=420)
model_lgbm, lgbm_score, lgbm_sd = score_worst_seeds(
    clf, lgbm_params, X_train, y_train, seeds, iters=args.lgbm_iters
)
wandb_logs["lgbm_score"] = {
    "mean": lgbm_score["mean"],
    "std": lgbm_score["std"],
}

###ENSEMBLING###
model_dict = {
    "rf": model_rf,
    "svm": model_svm,
    "xgb": model_xgb,
    "lgbm": model_lgbm,
    "mlp": model_mlp,
}
score_ensemble(run, model_dict, X_train, y_train, cv_seeds, N=args.Ncombs)
run.log(wandb_logs)
run.finish()
