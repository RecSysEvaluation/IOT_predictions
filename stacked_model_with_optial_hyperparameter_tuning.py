############################# IMPORT ALL LIBRARIES AND REQUIRED FILES ####################################

from preprocessing.TON_IOT_multi_classification import *
from helper_functions import *
from bayes_opt import BayesianOptimization
from accuracy.multi_accuracy import *
from algorithms.RF.RF import *
from algorithms.CatB.CatB import *
from algorithms.LightB.LightB import *
from algorithms.XGBoost.XGBoost import *

from Optimization_files.opt_DT import *
from Optimization_files.opt_CatBoost import *
from Optimization_files.opt_LightGBM import *
from Optimization_files.opt_XGBoost import *

import pandas as pd
from scipy.stats import randint, uniform
from pathlib import Path
from functools import partial
from sklearn.model_selection import StratifiedKFold


from algorithms.MLP.MLP import *
from algorithms.LR.lr import *
from algorithms.NB.nb import *

models_object_dict = dict()
############################# LOAD DATASET ###############################################################
DATA_PATH = r'./data/raw/'
data_name = "ToN_IoT_train_test_network"

X, y = data_load(DATA_PATH, data_name)
X_train, X_test, y_train, y_test = split_data_train_test(X, y)


############################# FIND OPTIMAL HYPER-PARAMETER VALUES FOR DT #################################
INTIAL_POINTS = 5
N_ITERATIONS = 10

cv_strategy = StratifiedKFold(n_splits=5)
opt_func = partial(optimize_dtree, X= X_train, y=y_train, cv=cv_strategy)
optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=dtbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points = INTIAL_POINTS, n_iter = N_ITERATIONS)
DT_optimal_hyperparameter_values = optimizer.max
print("file")
models_object_dict["DT"] = DTree(criterion = criterion_map[DT_optimal_hyperparameter_values["params"]["criterion"]], 
                                 max_depth = DT_optimal_hyperparameter_values["params"]['max_depth'], 
                                 splitter = splitter_map[DT_optimal_hyperparameter_values["params"]['splitter']])


############################# FIND OPTIMAL HYPER-PARAMETER VALUES FOR CatBOOST #################################
opt_func = partial(optimize_catb, X = X_train, y = y_train, cv = cv_strategy)
optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=catbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points= INTIAL_POINTS, n_iter=N_ITERATIONS)
CatBoost_optimal_hyperparameter_values = optimizer.max

models_object_dict["CatB"] = CatB(iterations = 300, learning_rate = 0.01, depth = 8, l2_leaf_reg = 1)

############################# FIND OPTIMAL HYPER-PARAMETER VALUES FOR LIGHTGBM #################################
opt_func = partial(optimize_lightb, X = X_train, y = y_train, cv = cv_strategy)
optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=lightbounds,
    random_state=42,
    verbose=2
)

#optimizer.maximize(init_points= INTIAL_POINTS, n_iter=N_ITERATIONS)
#lightBoost_optimal_hyperparameter_values = optimizer.max


accuracy_objects_dict = dict()
accuracy_objects_dict["Accuracy"] = Acc()
accuracy_objects_dict["Precision"] = Precision()
accuracy_objects_dict["Recall"] = Recall()
accuracy_objects_dict["F1_score"] = F1_score()



# LightBoost-n_estimators=70-learning_rate=0.05-max_depth=35-num_leaves=127-min_child_samples=20
models_object_dict["LightB"] = LightB(n_estimators = 70, learning_rate = 0.05, max_depth = 35, 
                 num_leaves = 127, min_child_samples = 20)


meta_features_trainX, meta_features_trainY = k_fold_return_meta_features(X_train, y_train, models_object_dict, accuracy_objects_dict, defaultHyperparameters = False)
meta_features_testX, meta_features_testY = return_metafeatures_for_single_splits(X_train, y_train, X_test, y_test, models_object_dict, accuracy_objects_dict, defaultHyperparameters = False)


############################# FIND OPTIMAL HYPER-PARAMETER VALUES FOR LIGHTGBM #################################
opt_func = partial(
    optimize_model,
    X_train=meta_features_trainX,
    y_train=meta_features_trainY,
    X_valid=meta_features_testX,
    y_valid=meta_features_testY
)
optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=xgbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points=5, n_iter=25)




#meta_features_trainX, meta_features_trainY, meta_features_testX, meta_features_testY