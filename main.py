import pandas as pd
import numpy as np
import wandb
import argparse

from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
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
from utils import score_combos, proc_pipeline, create_col_names, make_violinplot, tsne_plot, correlation_plot, score_clf, rank_seeds, plot_fi, run_cvs, find_worst_seeds, score_worst_seeds
from hyperparams import lr_params, mlp_params, rf_params, svm_params, xgb_params, lgbm_params

import warnings 
warnings.filterwarnings("ignore")
NUM_TRIALS = 1000
SEED = 42420
scv = StratifiedKFold(n_splits=5)

wandb.login()

# create argparser arguments
parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, default="lgbm")
parser.add_argument("embedding", type=str, default="no")
parser.add_argument("environment", type=str, default="local")
parser.add_argument("splits", type=int, default=3)
parser.add_argument("iterations", type=int, default=1)
parser.add_argument("log", type=str, default="no")
parser.add_argument("usalgo", type=str, default="tomek")
args = parser.parse_args()

# initialize wandb run
if args.log == "yes":
    run = wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY,
        job_type="modeling",
        notes=f"Modelling the ipl2022 dataset with {clfs[args.model]} (5 classes) with feature embeddings={args.embedding}",
        tags=[
            f"niter{args.iterations}",
            f"model{args.model}",
            f"undersample_{args.usalgo}",
            "ipl2022",
            "5_classes",
            "custom_metrics",
        ],
    )


###IMPORT DATA####
df = pd.read_csv('Crop_Location_Data_20221201.csv')
s1_data = np.load('sar_data.npy') #Sentinel-2 data
s2_data = np.load('sentinel2_data_train.npy')[:,:4,:55]
s2_data[s2_data == 0] = np.nan

###PRE-PROCESSING PIPELINE###
proc_pipe1 = Pipeline([
      ('thresh', VarianceThreshold()), #Remove colmns with constant features
      ('spline', SplineTransformer(n_knots=8, degree=4, extrapolation='periodic')),
      ('scale', StandardScaler()), #Scale the data
    #   ('fe', PolynomialFeatures(degree=3)),
      ('select_feats1', SelectKBest(f_classif, k=100)), #select top 50 features using f_classif strategy
    #   ('select_feats', SelectFromModel(XGBClassifier(random_state=40), max_features=50)),
      ('select_feats2', RFECV(XGBClassifier(random_state=40), step=0.05, min_features_to_select=10)),
    #   ('tsne', KernelPCA(n_components=3, kernel='cosine', random_state=42))
])

proc_pipe = proc_pipe1

le = LabelEncoder()
y = le.fit_transform(df['Class of Land'])
complete_df = proc_pipeline(s1_data, None)
dicts = create_col_names()
complete_df.rename(columns=dicts,inplace=True)
X_train, X_test, y_train, y_test=train_test_split(complete_df,y,test_size=0.45,shuffle=True, random_state=807410395)
X_train = proc_pipe1.fit_transform(X_train,y_train)
X_test = proc_pipe1.transform(X_test)
sel_feats = proc_pipe1.get_feature_names_out().tolist()
df_ = pd.DataFrame(X_train,columns=sel_feats)
df_['Target'] = y_train

X = proc_pipe.fit_transform(complete_df,y)
###VIOLIN PLOT###
vp = make_violinplot(df_)

###CORRELATION PLOT###
cp = correlation_plot(df_)

###TSNE PLOT###
fig, tsne = tsne_plot(df_, 'Target')

###RANK SEEDS###
clf = LogisticRegression(random_state=5000)
seeds2 = rank_seeds(clf)
score_clf(clf, X_train, y_train, seeds2)

###FEATURE IMPORTANCES PLOT###
model = clf.fit(X_train,y_train)
coefs = model.coef_.tolist()[0]
feat_imp = {c:i for c,i in zip(sel_feats,[abs(c) for c in coefs])}
plot_fi(feat_imp)

#BLOCK TO FIND THE WORST PERFORMING SEED ON THE DATA
clf = LogisticRegression(max_iter=200,random_state=420)
res_ridge = run_cvs(clf, X_train, y_train, lr_params, splits=5, iters=1, metrics='f1_weighted')
seeds = find_worst_seeds(res_ridge, topk=10)

###MLP CLASSIFIER###
clf = MLPClassifier(batch_size=32, solver='adam', random_state=42, max_iter=5000, n_iter_no_change=10)
model_mlp = score_worst_seeds(clf, mlp_params, X_train, y_train, seeds, iters=10)

###RANDOM FOREST CLASSIFIER###
clf = RandomForestClassifier(random_state=420)
model_rf = score_worst_seeds(clf, rf_params, X_train, y_train, seeds,iters=500)

###SUPPORT VECTOR CLASSIFIER###
clf = SVC(random_state=420, probability=True)
model_svm = score_worst_seeds(clf, svm_params, X_train, y_train, seeds, iters=1000)

###XGBOOST CLASSIFIER###
clf = XGBClassifier(random_state=420)
model_xgb = score_worst_seeds(clf, xgb_params, X_train, y_train, seeds, iters=500)

###LIGHTGBM CLASSIFIER###
clf = LGBMClassifier(random_state=420)
model_lgbm = score_worst_seeds(clf, lgbm_params, X_train, y_train, seeds, iters=500)

###ENSEMBLING###
model_dict = {'rf':model_rf, 'svm':model_svm, 'xgb':model_xgb, 'lgbm':model_lgbm, 'mlp':model_mlp}
score_combos(model_dict, N=2)


