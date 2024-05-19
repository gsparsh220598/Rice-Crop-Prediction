from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, cross_validate

from sklearn.manifold import TSNE
import warnings 
warnings.filterwarnings("ignore")
NUM_TRIALS = 10000
SEED = 42420
scv = StratifiedKFold(n_splits=5)

times = np.array(['2021-11-11T03:19:49.024000000', '2021-11-16T03:20:11.024000000',
    '2021-11-21T03:20:29.024000000', '2021-11-26T03:20:51.024000000',
    '2021-12-01T03:21:09.024000000', '2021-12-06T03:21:21.024000000',
    '2021-12-11T03:21:29.024000000', '2021-12-16T03:21:41.024000000',
    '2021-12-21T03:21:39.024000000', '2021-12-26T03:21:41.024000000',
    '2021-12-31T03:21:29.024000000', '2022-01-05T03:21:31.024000000',
    '2022-01-10T03:21:09.024000000', '2022-01-20T03:20:39.024000000',
    '2022-01-30T03:19:49.024000000', '2022-02-04T03:19:31.024000000',
    '2022-02-09T03:18:59.024000000', '2022-02-14T03:18:31.024000000',
    '2022-02-19T03:17:49.024000000', '2022-02-24T03:17:31.024000000',
    '2022-03-01T03:16:49.024000000', '2022-03-06T03:16:21.024000000',
    '2022-03-11T03:15:39.024000000', '2022-03-16T03:15:41.024000000',
    '2022-03-21T03:15:39.024000000', '2022-03-26T03:15:41.024000000',
    '2022-03-31T03:15:39.024000000', '2022-04-05T03:15:41.024000000',
    '2022-04-10T03:15:39.024000000', '2022-04-15T03:15:41.024000000',
    '2022-04-20T03:15:39.024000000', '2022-04-25T03:15:41.024000000',
    '2022-04-30T03:15:29.024000000', '2022-05-10T03:15:39.024000000',
    '2022-05-15T03:15:41.024000000', '2022-05-20T03:15:39.024000000',
    '2022-05-25T03:15:51.024000000', '2022-05-30T03:15:39.024000000',
    '2022-06-04T03:15:51.024000000', '2022-06-09T03:15:39.024000000',
    '2022-06-14T03:15:51.024000000', '2022-06-19T03:15:19.024000000',
    '2022-06-24T03:15:51.024000000', '2022-06-29T03:15:29.024000000',
    '2022-07-04T03:15:31.024000000', '2022-07-09T03:15:29.024000000',
    '2022-07-14T03:15:51.024000000', '2022-07-19T03:15:29.024000000',
    '2022-07-24T03:15:31.024000000', '2022-07-29T03:15:29.024000000',
    '2022-08-03T03:15:31.024000000', '2022-08-08T03:15:19.024000000',
    '2022-08-13T03:15:31.024000000', '2022-08-18T03:15:19.024000000',
    '2022-08-23T03:15:31.024000000'])
times = [t.split('T')[0][:7] for t in times]


def prep_s2_data(data, time=times):
    L=0.5
    prep_data = []
    for d in range(data.shape[0]):
        ds = pd.DataFrame()
        ds['time'] = time
        ds['red'] = data[d,0,:]
        ds['green'] = data[d,1,:]
        ds['blue'] = data[d,2,:]
        ds['nir'] = data[d,3,:]
        ds['ndvi'] = (ds.nir-ds.red)/(ds.nir+ds.red)
        ds['savi'] = (1+L)*(ds.nir-ds.red)/(ds.nir+ds.red+L)
        ds['evi'] = 2.5*((ds.nir-ds.red)/(ds.nir+6*ds.red-7.5*ds.blue)+1)
        sample = ds.groupby('time').mean(numeric_only=True).reset_index()[['ndvi','savi','evi']].to_numpy().reshape(3,10) #CHANGE FEATURES HERE 
        prep_data.append(sample)
    return np.array(prep_data)

def prep_s1_data(data):
    vv = data[:,0,:]
    vh = data[:,1,:]
    q = vh/vv
    n = q*(q+3)
    d = (q+1)**2
    rvi = n/d #CALCULATION OF Radar Vegetation Index
    ndvi_sar = (vh-vv)/(vh+vv)
    if len(data.shape) == 3:
        rvi = rvi.reshape(data.shape[0],1,data.shape[2])
        ndvi_sar = ndvi_sar.reshape(data.shape[0],1,data.shape[2])
    else:
        rvi = rvi.reshape(data.shape[0],1)
        ndvi_sar = ndvi_sar.reshape(data.shape[0],1)
    rvi = np.nan_to_num(rvi, nan=0)
    new_data = np.concatenate((data,rvi,ndvi_sar),axis=1).copy()
    return new_data



def create_col_names(features_sar=['VV','VH','RVI','NDVI_SAR'],features_o=['ndvi','savi','evi'],timesteps=[52,10]):
    cols_sar = [f'{feat}_{t}' for feat in features_sar for t in range(0,timesteps[0])]
    cols_o = [f'{feat}_{t}' for feat in features_o for t in range(0,timesteps[1])]
    cols = cols_sar+cols_o
    dicts = {t:c for t,c in zip(range(0,len(features_sar)*timesteps[0]+len(features_o)*timesteps[1]),cols)}
    return dicts
    
def proc_pipeline(s1data,s2data):
    if s1data is not None:
        data_s1 = s1data[:,:,:52].copy() #CHANGE HERE FOR TIMESTEPS
        data_s1 = prep_s1_data(data_s1)
        data_s1 = pd.DataFrame(data_s1.reshape(data_s1.shape[0],data_s1.shape[1]*data_s1.shape[2]))#.dropna(axis=1)
        if s2data is not None:
            data_s2 = prep_s2_data(s2data)
            data_s2 = pd.DataFrame(data_s2.reshape(data_s2.shape[0],data_s2.shape[1]*data_s2.shape[2]))
            complete_df = pd.concat([data_s1,data_s2],axis=1, ignore_index=True)
            return complete_df
        else:
            return data_s1
    else:
        data_s2 = prep_s2_data(s2data)
        data_s2 = pd.DataFrame(data_s2.reshape(data_s2.shape[0],data_s2.shape[1]*data_s2.shape[2]))
        return data_s2
    

def correlation_plot(dataframe):
    """
    Generates a correlation heatmap plot using Seaborn.

    Parameters:
    - dataframe: pandas DataFrame containing numeric columns.

    Returns:
    - None (displays the plot).
    """
    corr_matrix = dataframe.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidths=0.5, annot_kws={"size": 6})
    plt.title('Correlation Heatmap')
    plt.show()
    #TODO: ADD WANDB ARTIFACT SAVE

def make_violinplot(df):
    traces = []

    for feature_name, feature_values in df.items():
        trace = go.Violin(y=feature_values, x=df['Target'], name=feature_name, box_visible=True, meanline_visible=True)
        traces.append(trace)

    fig = go.Figure(data=traces)
    fig.update_layout(title='Violin Plot of Feature Variables by Target Class', yaxis_title='Feature Value', xaxis_title='Rice or Non-Rice')
    fig.show()
    #TODO: ADD WANDB ARTIFACT SAVE
    return fig

def tsne_plot(dataframe, target_column):
    """
    Generates a t-SNE plot using Plotly with hover information to visualize high-dimensional data.

    Parameters:
    - dataframe: pandas DataFrame containing numeric columns.
    - target_column: Name of the target column in the DataFrame.

    Returns:
    - fig: Plotly Figure object containing the t-SNE plot with hover information.
    """
    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    hover_text = ['Index: {}'.format(index) for index in dataframe.index]
    fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=y, labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2'}, color_continuous_scale='viridis', title='t-SNE Plot', hover_name=hover_text)
    fig.update_coloraxes(colorbar_title=target_column)
    #TODO: ADD WANDB ARTIFACT SAVE
    return fig, X_tsne


def rank_seeds(clf, X_train, y_train, topk=10):
    sd = {}
    for seed in tqdm(range(0,NUM_TRIALS)):
        scv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        score = cross_val_score(clf, X_train, y_train, cv=scv.split(X_train, y_train), scoring='accuracy').mean()
        sd.update({seed:score})
    seeds = pd.DataFrame(list(sd.items()), columns=['Seed', 'Score']).sort_values('Score',ascending=True).index[0:topk].values.tolist()
    return seeds

def score_clf(clf, X_train, y_train, seeds2):
    scores = []
    for seed in seeds2:
        scv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        score = cross_val_score(clf, X_train, y_train, cv=scv.split(X_train, y_train), scoring='accuracy').mean()
        scores.append(score)
    return {'score':np.mean(scores),'sd':np.std(scores)}

def plot_fi(feat_imp):
    sorted_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
    features = [item[0] for item in sorted_features]
    importance_scores = [item[1] for item in sorted_features]

    fig = go.Figure(data=[go.Bar(x=features, y=importance_scores)])
    fig.update_layout(title='Feature Importance', xaxis_title='Feature', yaxis_title='Importance Score')
    fig.show()
    #TODO: ADD WANDB ARTIFACT SAVE


#NESTED CV STRATEGY TO SCORE THE VALIDATION SET
def nested_cv(
    pipe,
    X,
    y,
    grid,
    splits,
    iters=5,
    seed=42,
    metrics="accuracy",
):
    inner_cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    outer_cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    clf = RandomizedSearchCV(estimator=pipe, n_iter=iters, param_distributions=grid, cv=inner_cv, 
                            scoring=metrics, n_jobs=-1, random_state=SEED)
    scores = cross_validate(clf, X=X, y=y, cv=outer_cv, scoring=metrics, return_estimator=True)
    model_params = [e.best_estimator_ for e in scores["estimator"]]
    return {
        "model_params": model_params,
        "accuracy": scores["test_score"],
    }

#FUNCTION TO RUN MULTIPLE NESTED CV TRIALS
def run_cvs(pipe, X, y, grid, splits=10, iters=5, metrics = 'accuracy'):
    cv_results = pd.DataFrame()
    row_res = {}

    for i in tqdm(range(NUM_TRIALS)): #ITERATE THROUGH NUMBER OF TRIALS
        row_res["seed"] = i
        cv_res = nested_cv(pipe, X, y, grid=grid, splits=splits, iters=iters, seed=i, metrics=metrics)
        row_res.update(cv_res)
        temp_res = pd.DataFrame(row_res, columns=list(row_res.keys()))
        cv_results = pd.concat([cv_results, temp_res], axis=0, ignore_index=True)
        row_res = {}
    return cv_results

#FUNCTION TO FIND THE WORST PERFORMING TRIAL OUT OF ALL THE TRIALS
def find_worst_seeds(res, topk=5):
    seeds = []
    for seed in res.groupby('seed')['accuracy'].mean().sort_values(ascending=True).index[0:topk].values:
        seeds.append(seed)
    return seeds

#MAKING VOTING CLASSIFIER USING A LIST OF MODELS
def make_vc(search_list, name_list):
    estimator_list = [(str(n), s) for n,s in zip(name_list, search_list)]
    return VotingClassifier(estimators=estimator_list, voting='soft')

def evaluate(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(classification_report(y_test,predictions))

def predict_submission(clf, proc_pipe, X, y, use_s2=False):
    testdfs1 = np.load('sar_data_test.npy')
    if use_s2:
        testdfs2 = np.load('sentinel2_data_test.npy')
        proc_testdf = proc_pipeline(testdfs1,testdfs2)
    else:
        proc_testdf = proc_pipeline(testdfs1,None)
    X_sub = proc_pipe.transform(proc_testdf)
    clf.fit(X,y)
    submission_predictions = clf.predict(X_sub)
    submission_probs = clf.predict_proba(X_sub)
    return submission_predictions, submission_probs

#THE FUNCTION TO CREATE VOTING CLASSIFIER BASED ON PERFORMANCE ON WORST PERFORMING SEED
def score_worst_seeds(clf, params, X, y, seeds, iters=100):
    model_ls = []
    valid_scores = []

    for seed in tqdm(seeds):
        cv_res = nested_cv(clf, X, y, grid=params, splits=5, iters=iters, seed=seed, metrics='f1_weighted')
        cv_models = cv_res['model_params']
        model = make_vc(cv_models, list([i for i in range(0,5)]))
        score = cv_res['accuracy'].mean()
        model_ls.append(model)
        valid_scores.append(score*100)

    print(f'The mean accuracy for the {len(seeds)} worst seeds is {np.mean(valid_scores)} and the std. dev. is {np.std(valid_scores)}')
    return make_vc(model_ls, seeds)

def make_ensemble(search_list, name_list):
    vclf = make_vc(search_list, name_list)
    return vclf

#use itertools to make N>2 combinations of rf, svm, xgb, lgbm, mlp
def score_combos(model_dict, N=2):
    from itertools import combinations
    combos = [list(combo) for combo in combinations(['rf', 'svm', 'xgb', 'lgbm', 'mlp'], N)]
    for combo in combos:
        search_combo = []
        for c in combo:
            search_combo.append(model_dict[c])
        vclf = make_ensemble(search_combo, combo)
        score_clf(vclf)
        evaluate(vclf)
    #TODO: ADD WANDB ARTIFACT SAVE

