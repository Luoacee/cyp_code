import numpy as np


rf_params = dict(

    n_estimators=list(range(50, 150, 10)),
    criterion=['gini'],
    n_jobs=[-1],
    max_depth=list(range(1, 100)),
    min_samples_split=list(range(2, 5, 1)),
    min_samples_leaf=[1, 2, 3],
    random_state=[100]

)


xgboost_params = dict(
    n_estimators=list(range(7500, 10000, 500)),
    learning_rate=[0.001, 0.01, 0.1, 1],
    gamma=[0, 0.0025, 0.05],
    reg_alpha=[0, 0.1, 0.3, 0.5, 0.7, 0.75, 1],
    reg_lambda=[0, 0.1, 0.5, 1],
    max_depth=[3, 5, 6, 7, 9, 10],
    min_child_weight=[1, 3, 5, 7],
    n_jobs=[-1],
    random_state=[100],
    tree_method=["gpu_hist"],
    gpu_id=[0]

)

catboost_params = dict(
    depth=[4, 5, 6, 7, 8, 9, 10],
    learning_rate=[0.001, 0.01, 0.1, 1],
    random_state=[100],
    l2_leaf_reg=[5],
    verbose=[False],
    task_type=["GPU"]
)

lightgbm_params = dict(
    learning_rate=[0.05, 0.1, 0.5, 1],
    n_estimators=[50, 70, 100, 120, 140, 160],
    max_depth=[3, 4, 5, 6, 7],
    verbose=[-1],
    lambda_l1=[1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0],
    lambda_l2=[1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0],
    categorical_column=[False]
)

svm_params = dict(
    kernel=['rbf'],
    gamma=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.5, 1],
    C=[1, 2, 3, 4, 5, 10, 100],
    class_weight=['balanced'],
    probability=[True],
    random_state=[100]
)
