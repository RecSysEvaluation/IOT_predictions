from preprocessing.TON_IOT_multi_classification import *
from helper_functions import *
from accuracy.multi_accuracy import *
from algorithms.RF.RF import *
from algorithms.CatB.CatB import *
from algorithms.LightB.LightB import *
from algorithms.XGBoost.XGBoost import *
import pandas as pd
from algorithms.DTree.DTree import * 
from algorithms.MLP.MLP import *
from algorithms.LR.lr import *
from algorithms.NB.nb import *

accuracy_objects_dict = dict()
accuracy_objects_dict["Accuracy"] = Acc()
accuracy_objects_dict["Precision"] = Precision()
accuracy_objects_dict["Recall"] = Recall()
accuracy_objects_dict["F1_score"] = F1_score()

# import models
models_object_dict = dict()
#models_object_dict["RandomForest"] = RF()

models_object_dict["NB"] = NB()
models_object_dict["LR"] = LR()
models_object_dict["MLP"] = MLP()
models_object_dict["DT"] = DTree()
models_object_dict["CatB"] = CatB()
models_object_dict["LightB"] = LightB()

DATA_PATH = r'./data/raw/'
data_name = "ToN_IoT_train_test_network"

X, y = data_load(DATA_PATH, data_name)
X_train, X_test, y_train, y_test = split_data_train_test(X, y)
meta_features_trainX, meta_features_trainY = k_fold_return_meta_features(X_train, y_train, models_object_dict, 
                                                                         accuracy_objects_dict, defaultHyperparameters=True)
meta_features_testX, meta_features_testY = return_metafeatures_for_single_splits(X_train, y_train, X_test, y_test, 
                                                                                 models_object_dict, accuracy_objects_dict, 
                                                                                 defaultHyperparameters=True)

# use XGBoost as stacked model.............................................
stacked_model_dict = dict()
stacked_model_dict["XGBoost"] = XGBoost()
stacked_model_object_dictAND_accuracy_dict(meta_features_trainX, meta_features_trainY, meta_features_testX, 
                                           meta_features_testY, stacked_model_dict, accuracy_objects_dict )

