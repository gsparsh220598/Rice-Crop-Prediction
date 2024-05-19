import numpy as np

lr_params = {
    "fit_intercept": [True, False],
    "C": np.linspace(0.01, 0.5, 40, dtype=np.float32),
    # "l1": np.linspace(0.1, 1.0, 40, dtype=np.float32),
    "solver": ["lbfgs"],
    "penalty": ["l2"],
}

mlp_params = {
    'hidden_layer_sizes': [(64,32), (32,16,8), (16,8), (16,8,4), (8,4,2)],
    'activation': ['relu', 'tanh', 'logistic'],
    # 'solver': ['adam'],
    'alpha': np.linspace(0.00001, 1.0, 10, dtype=np.float16),
    'learning_rate': ['constant', 'adaptive', 'invscaling'],
    'learning_rate_init': np.linspace(1e-6, 1e-4, 10, dtype=np.float16),
    # 'batch_size': np.linspace(1, 16, 8, dtype=np.int16),
    # 'max_iter': np.linspace(1000, 10000, 20, dtype=np.int16),
    'beta_1': np.linspace(0.00001, 1.0, 10, dtype=np.float16),
    'beta_2': np.linspace(0.00001, 1.0, 10, dtype=np.float16),
    'epsilon': np.linspace(1e-8, 1e-5, 10, dtype=np.float16),
}

rf_params = {
    "n_estimators": np.linspace(50, 100, 5, dtype=np.int16),
    "max_depth": np.linspace(2, 15, 10, dtype=np.int16),
    "min_samples_split": np.linspace(2, 20, 10, dtype=np.int16),
    "min_samples_leaf": np.linspace(2, 20, 10, dtype=np.int16),
    "criterion": ["gini", "entropy", "log_loss"],
    "ccp_alpha": np.linspace(0.0, 5.0, 20, dtype=np.float16),
}

xgb_params = {
    "n_estimators": np.linspace(50, 100, 10, dtype=np.int16),
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
    "n_estimators": np.linspace(50, 100, 10, dtype=np.int16),
    "max_depth": np.linspace(2, 20, 20, dtype=np.int16),
    "num_leaves": np.linspace(2, 10, 6, dtype=np.int16),
    "learning_rate": np.linspace(1e-3, 1.0, 40, dtype=np.float16),
    "subsample": np.linspace(0.2, 1.0, 40, dtype=np.float16),
    "colsample_bytree": np.linspace(0.2, 1.0, 40, dtype=np.float16),
    "reg_alpha": np.linspace(0.0, 1.0, 20, dtype=np.float16),
    "reg_lambda": np.linspace(0.0, 1.0, 20, dtype=np.float16),
    "min_child_samples": np.linspace(2, 7, 5, dtype=np.int16),
}