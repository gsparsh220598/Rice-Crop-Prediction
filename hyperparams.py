import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

SEED = 42420

lr_params = {
    "fit_intercept": [True, False],
    "C": np.linspace(0.01, 0.5, 40, dtype=np.float32),
    # "l1": np.linspace(0.1, 1.0, 40, dtype=np.float32),
    "solver": ["lbfgs"],
    "penalty": ["l2"],
}

svm_params = {
    "kernel": ["rbf"],
    "C": np.linspace(0.001, 5.0, 50, dtype=np.float32),
    "gamma": np.linspace(0.01, 5.0, 50, dtype=np.float16),
}

mlp_params = {
    # "hidden_layer_sizes": [(64, 32), (32, 16, 8), (16, 8), (16, 8, 4), (8, 4, 2)],
    "activation": ["relu", "tanh", "logistic"],
    # 'solver': ['adam'],
    "alpha": np.linspace(0.00001, 1.0, 10, dtype=np.float16),
    "learning_rate": ["constant", "adaptive", "invscaling"],
    "learning_rate_init": np.linspace(1e-6, 1e-4, 10, dtype=np.float16),
    # 'batch_size': np.linspace(1, 16, 8, dtype=np.int16),
    # 'max_iter': np.linspace(1000, 10000, 20, dtype=np.int16),
    "beta_1": np.linspace(0.00001, 1.0, 10, dtype=np.float16),
    "beta_2": np.linspace(0.00001, 1.0, 10, dtype=np.float16),
    "epsilon": np.linspace(1e-8, 1e-5, 10, dtype=np.float16),
}

rf_params = {
    "n_estimators": np.linspace(20, 100, 5, dtype=np.int16),
    "max_depth": np.linspace(2, 15, 10, dtype=np.int16),
    "min_samples_split": np.linspace(2, 20, 10, dtype=np.int16),
    "min_samples_leaf": np.linspace(2, 20, 10, dtype=np.int16),
    "criterion": ["gini", "entropy", "log_loss"],
    "ccp_alpha": np.linspace(0.0, 5.0, 20, dtype=np.float16),
}

xgb_params = {
    "n_estimators": np.linspace(20, 100, 10, dtype=np.int16),
    "max_depth": np.linspace(2, 30, 15, dtype=np.int16),
    "max_leaves": np.linspace(2, 10, 6, dtype=np.int16),
    "learning_rate": np.linspace(1e-3, 1.0, 40, dtype=np.float16),
    "subsample": np.linspace(0.2, 1.0, 40, dtype=np.float16),
    "colsample_bytree": np.linspace(0.2, 1.0, 40, dtype=np.float16),
    "reg_alpha": np.linspace(0.0, 1.0, 20, dtype=np.float16),
    "reg_lambda": np.linspace(0.0, 1.0, 20, dtype=np.float16),
    "monotone_constraints": [None, (1, -1), (-1, 1)],
    "grow_policy": ["depthwise", "lossguide"],
}


lgbm_params = {
    "n_estimators": np.linspace(20, 100, 10, dtype=np.int16),
    "max_depth": np.linspace(2, 20, 20, dtype=np.int16),
    "num_leaves": np.linspace(2, 10, 6, dtype=np.int16),
    "learning_rate": np.linspace(1e-3, 1.0, 40, dtype=np.float16),
    "subsample": np.linspace(0.2, 1.0, 40, dtype=np.float16),
    "colsample_bytree": np.linspace(0.2, 1.0, 40, dtype=np.float16),
    "reg_alpha": np.linspace(0.0, 1.0, 20, dtype=np.float16),
    "reg_lambda": np.linspace(0.0, 1.0, 20, dtype=np.float16),
    "min_child_samples": np.linspace(2, 7, 5, dtype=np.int16),
}

gbm_params = {
    "n_estimators": np.linspace(20, 100, 10, dtype=np.int16),
    "max_depth": np.linspace(2, 20, 15, dtype=np.int16),
    "min_samples_split": np.linspace(2, 20, 10, dtype=np.int16),
    "min_samples_leaf": np.linspace(2, 20, 10, dtype=np.int16),
    "max_features": ["sqrt", "log2", None],
    "learning_rate": np.linspace(1e-3, 1.0, 10, dtype=np.float16),
    "criterion": ["friedman_mse", "squared_error"],
    "loss": ["log_loss", "deviance", "exponential"],
}

xt_params = {
    "n_estimators": np.linspace(20, 100, 10, dtype=np.int16),
    "max_depth": np.linspace(2, 20, 15, dtype=np.int16),
    "min_samples_split": np.linspace(2, 20, 10, dtype=np.int16),
    "min_samples_leaf": np.linspace(2, 20, 10, dtype=np.int16),
    "max_leaf_nodes": np.linspace(2, 20, 10, dtype=np.int16),
    "max_features": ["sqrt", "log2", None],
    "criterion": ["gini", "entropy", "log_loss"],
    "bootstrap": [True, False],
}

bag_params = {
    "estimator": [
        GaussianNB(),
        DecisionTreeClassifier(random_state=SEED),
        LogisticRegression(random_state=SEED),
    ],
    "n_estimators": np.linspace(20, 200, 10, dtype=np.int16),
    # "max_samples": np.linspace(0.2, 1.0, 10, dtype=np.float16),
    "bootstrap": [True, False],
    # "warm_start": [True, False],
}

ada_params = {
    "estimator": [
        GaussianNB(),
        DecisionTreeClassifier(random_state=SEED),
        LogisticRegression(random_state=SEED),
    ],
    "n_estimators": np.linspace(20, 200, 10, dtype=np.int16),
    "learning_rate": np.linspace(1e-4, 2.0, 20, dtype=np.float16),
}

knn_params = {
    "n_neighbors": np.linspace(3, 20, 10, dtype=np.int16),
    "weights": ["uniform", "distance"],
    "algorithm": ["ball_tree", "kd_tree", "brute"],
    "leaf_size": np.linspace(10, 100, 40, dtype=np.int16),
    # "p": np.linspace(1, 3, 3, dtype=np.int16),
    # "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
}

experiment_space = {
    "pca": ["yes", "no"],
    "nfeats": np.linspace(start=40, stop=100, num=20, dtype=np.int16).tolist(),
    "knots": np.linspace(start=5, stop=20, num=10, dtype=np.int16).tolist(),
    "degree": np.linspace(start=2, stop=5, num=5, dtype=np.int16).tolist(),
    "ncomps": np.linspace(start=2, stop=10, num=5, dtype=np.int16).tolist(),
    "kernel": ["linear", "poly", "rbf", "sigmoid", "cosine"],
    "extrap": ["constant", "linear", "continue", "periodic"],
    "nseeds": np.linspace(start=2, stop=4, num=3, dtype=np.int16).tolist(),
    # "mlp_iters": np.linspace(start=10, stop=100, num=20, dtype=np.int16).tolist(),
    "rf_iters": np.linspace(start=100, stop=500, num=10, dtype=np.int16).tolist(),
    "xgb_iters": np.linspace(start=100, stop=500, num=10, dtype=np.int16).tolist(),
    "lgbm_iters": np.linspace(start=100, stop=1000, num=10, dtype=np.int16).tolist(),
    "svm_iters": np.linspace(start=100, stop=500, num=10, dtype=np.int16).tolist(),
    "ada_iters": np.linspace(start=100, stop=500, num=10, dtype=np.int16).tolist(),
    "bag_iters": np.linspace(start=100, stop=500, num=10, dtype=np.int16).tolist(),
    "knn_iters": np.linspace(start=100, stop=500, num=10, dtype=np.int16).tolist(),
    "gbm_iters": np.linspace(start=100, stop=500, num=10, dtype=np.int16).tolist(),
    "xt_iters": np.linspace(start=100, stop=200, num=10, dtype=np.int16).tolist(),
    "Ncombs": np.linspace(start=2, stop=5, num=3, dtype=np.int16).tolist(),
}
