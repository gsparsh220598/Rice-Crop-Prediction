import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import wandb
import argparse
from dotenv import load_dotenv

from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.decomposition import KernelPCA, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
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
    ada_params,
    knn_params,
    gbm_params,
    bag_params,
    xt_params,
)

import warnings

warnings.filterwarnings("ignore")
load_dotenv()

# PROJECT_NAME = os.getenv("PROJECT_NAME")
# ENTITY = os.getenv("ENTITY")
# NUM_TRIALS = int(os.getenv("NUM_TRIALS"))
# SEED = int(os.getenv("SEED"))
PROJECT_NAME = "EY RICE CROP CLASSIFICATION"
ENTITY = None
SEED = 42420
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
parser.add_argument("extrap", type=str, default="periodic")
parser.add_argument("nseeds", type=int, default=3)
# parser.add_argument("mlp_iters", type=int, default=10)
parser.add_argument("rf_iters", type=int, default=200)
parser.add_argument("xgb_iters", type=int, default=200)
parser.add_argument("lgbm_iters", type=int, default=200)
parser.add_argument("svm_iters", type=int, default=200)
parser.add_argument("ada_iters", type=int, default=10)
parser.add_argument("bag_iters", type=int, default=200)
parser.add_argument("knn_iters", type=int, default=200)
parser.add_argument("gbm_iters", type=int, default=200)
parser.add_argument("xt_iters", type=int, default=200)
parser.add_argument("Ncombs", type=int, default=2)
args = parser.parse_args()

# initialize wandb run
run = wandb.init(
    project=PROJECT_NAME,
    # entity=ENTITY,
    job_type="modeling",
    notes=f"Using SAR dataset to classify a piece of land from Rice to Non-Rice",
    tags=[
        "SAR_dataset",
        f"pca{args.pca}",
        f"nfeats_min{args.nfeats}",
        f"knots{args.knots}",
        f"degree{args.degree}",
        f"ncomps{args.ncomps}",
        f"kernel{args.kernel}",
        f"extrap{args.extrap}",
        f"nseeds{args.nseeds}",
        # f"mlp_iters{args.mlp_iters}",
        f"rf_iters{args.rf_iters}",
        f"xgb_iters{args.xgb_iters}",
        f"lgbm_iters{args.lgbm_iters}",
        f"svm_iters{args.svm_iters}",
        f"ada_iters{args.ada_iters}",
        f"bag_iters{args.bag_iters}",
        f"knn_iters{args.knn_iters}",
        f"gbm_iters{args.gbm_iters}",
        f"xt_iters{args.xt_iters}",
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
# X = proc_pipe.fit_transform(complete_df, y)
###VIOLIN PLOT###
make_violinplot(df_, run)

###CORRELATION PLOT###
# correlation_plot(df_, run)

###TSNE PLOT###
tsne_plot(df_, target_column="Target", run=run)

wandb_logs = {}
###RANK SEEDS###
clf = LogisticRegression(max_iter=200, random_state=SEED)
cv_seeds = rank_seeds(clf, X_train, y_train)
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
seeds = find_worst_seeds(res_ridge, topk=args.nseeds)
wandb_logs["ncv_seeds"] = seeds

################MODELING#################

# ###MLP CLASSIFIER###
# clf = MLPClassifier(
#     hidden_layer_sizes=(16, 8, 4),
#     batch_size=32,
#     solver="adam",
#     random_state=SEED,
#     max_iter=5000,
#     n_iter_no_change=10,
# )
# model_mlp, mlp_score = score_worst_seeds(
#     clf, mlp_params, X_train, y_train, seeds, iters=args.mlp_iters
# )
# wandb_logs["mlp_score"] = mlp_score

###RANDOM FOREST CLASSIFIER###
clf = RandomForestClassifier(random_state=SEED, n_jobs=-1)
model_rf, rf_score = score_worst_seeds(
    clf, rf_params, X_train, y_train, seeds, iters=args.rf_iters
)
wandb_logs["rf_score"] = rf_score


###SUPPORT VECTOR CLASSIFIER###
clf = SVC(random_state=SEED, probability=True)
model_svm, svm_score = score_worst_seeds(
    clf, svm_params, X_train, y_train, seeds, iters=args.svm_iters
)
wandb_logs["svm_score"] = svm_score


###XGBOOST CLASSIFIER###
clf = XGBClassifier(random_state=SEED, n_jobs=-1)
model_xgb, xgb_score = score_worst_seeds(
    clf, xgb_params, X_train, y_train, seeds, iters=args.xgb_iters
)
wandb_logs["xgb_score"] = xgb_score

###LIGHTGBM CLASSIFIER###
clf = LGBMClassifier(random_state=SEED, n_jobs=-1)
model_lgbm, lgbm_score = score_worst_seeds(
    clf, lgbm_params, X_train, y_train, seeds, iters=args.lgbm_iters
)
wandb_logs["lgbm_score"] = lgbm_score

###ADABOOST CLASSIFIER###
clf = AdaBoostClassifier(random_state=SEED)
model_ada, ada_score = score_worst_seeds(
    clf, ada_params, X_train, y_train, seeds, iters=args.ada_iters
)
wandb_logs["ada_score"] = ada_score

###KNN CLASSIFIER###
clf = KNeighborsClassifier(n_jobs=-1)
model_knn, knn_score = score_worst_seeds(
    clf, knn_params, X_train, y_train, seeds, iters=args.knn_iters
)
wandb_logs["knn_score"] = knn_score

###GBM CLASSIFIER###
clf = GradientBoostingClassifier(random_state=SEED)
model_gbm, gbm_score = score_worst_seeds(
    clf, gbm_params, X_train, y_train, seeds, iters=args.gbm_iters
)
wandb_logs["gbm_score"] = gbm_score

###BAGGING CLASSIFIER###
clf = BaggingClassifier(random_state=SEED, n_jobs=-1)
model_bag, bag_score = score_worst_seeds(
    clf, bag_params, X_train, y_train, seeds, iters=args.bag_iters
)
wandb_logs["bag_score"] = bag_score

###EXTRATREES CLASSIFIER###
clf = ExtraTreesClassifier(random_state=SEED, n_jobs=-1)
model_xt, xt_score = score_worst_seeds(
    clf, xt_params, X_train, y_train, seeds, iters=args.xt_iters
)
wandb_logs["xt_score"] = xt_score

###ENSEMBLING###
model_dict = {
    "rf": model_rf,
    "svm": model_svm,
    "xgb": model_xgb,
    "lgbm": model_lgbm,
    "ada": model_ada,
    "knn": model_knn,
    "gbm": model_gbm,
    "bag": model_bag,
    "xt": model_xt,
    # "mlp": model_mlp,
}
score_ensemble(run, model_dict, X_train, y_train, X_test, y_test, cv_seeds, N=9)
run.log(wandb_logs)
run.finish()
