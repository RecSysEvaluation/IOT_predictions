
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from pathlib import Path




def k_fold_return_meta_features(X_train, y_train, models_object_dict, accuracy_objects_dict, n_splits = 5, random_state = 42, defaultHyperparameters = False, binary = False):
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    
    if defaultHyperparameters:
        if binary:
            path = Path("results/binary/defaultHyperparameters/")
            path.mkdir(parents=True, exist_ok=True)
        else:
            path = Path("results/defaultHyperparameters/")
            path.mkdir(parents=True, exist_ok=True)
    else:
        path = Path("results/optimalHyperparameters/")
        path.mkdir(parents=True, exist_ok=True)



    meta_features = dict()
    for key in models_object_dict.keys():
        meta_features[key] = list()
    meta_y = list()

    # We are using out of fold strategy to avoid data leakage issue............
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    result_dataframe = dict()

    for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        for key in models_object_dict.keys():
            print(f"*********************** {key} ***********************")
            models_object_dict[key].fit(X_train[train_index], y_train[train_index])

            y_predict = models_object_dict[key].predict(X_train[test_index])
            y_predict_prob = models_object_dict[key].predict_proba(X_train[test_index])

            column_names = [i for i in range(y_predict_prob.shape[1])]
            temp_df = pd.DataFrame(y_predict_prob, columns= column_names)
            meta_features[key].append(temp_df)
            
            # print accuracy values on out of fold
            temp = dict()
            for acc_key, acc_object in accuracy_objects_dict.items():
                acc_object.compute(y_predict, y_train[test_index])
                temp[acc_key] = round(acc_object.result()[1], 4)
                print(acc_object.result())
                
            result_dataframe[str(key) +"_fold_"+str(fold)] = temp

        meta_y.append(y_train[test_index])   
    

    result_dataframe = pd.DataFrame.from_dict(result_dataframe, orient='index')
    path = path / "results_with_folding.csv"
    result_dataframe.to_csv(path, sep = "\t")

    meta_features_df = pd.DataFrame()
    for key, value in meta_features.items():
        temp = pd.concat(value, axis=0, ignore_index=True)
        meta_features_df = pd.concat([meta_features_df, temp], axis= 1, ignore_index=True)

    # make meta features frame.........................................................
    new_column_names = [i for i in   range(  meta_features_df.shape[1] )]  
    meta_features_df.columns = new_column_names
    meta_features_y = [item   for sublist in meta_y for item in list(sublist)]

    return meta_features_df, meta_features_y



def return_metafeatures_for_single_splits(X_train, y_train, X_test, y_test, models_object_dict, accuracy_objects_dict, defaultHyperparameters = False, binary = False):
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    
    if defaultHyperparameters:
        if binary:
            path = Path("results/binary/defaultHyperparameters/")
            path.mkdir(parents=True, exist_ok=True)
        else:
            path = Path("results/defaultHyperparameters/")
            path.mkdir(parents=True, exist_ok=True)
    else:
        path = Path("results/optimalHyperparameters/")
        path.mkdir(parents=True, exist_ok=True)

    result_dataframe = dict()

    meta_features = dict()
    for key in models_object_dict.keys():
        meta_features[key] = list()
    # results with full data
    for key in models_object_dict.keys():
        print(f"*********************** Results with full data: {key} ***********************")
        models_object_dict[key].fit(X_train, y_train)

        
        y_predict = models_object_dict[key].predict(X_test)
        y_predict_prob = models_object_dict[key].predict_proba(X_test)

        column_names = [i for i in range(y_predict_prob.shape[1])]
        temp_df = pd.DataFrame(y_predict_prob, columns= column_names)
        meta_features[key].append(temp_df)
            
            # print accuracy values on out of fold
        temp = dict()
        for acc_key, acc_object in accuracy_objects_dict.items():
            acc_object.compute(y_predict, y_test)
            temp[acc_key] = round(acc_object.result()[1], 4)
            print(acc_object.result())

        result_dataframe[str(key)] = temp
           
    result_dataframe = pd.DataFrame.from_dict(result_dataframe, orient='index')
    path = path / "results_with_single_split.csv"
    result_dataframe.to_csv(path, sep = "\t")

    meta_features_df = pd.DataFrame()
    for key, value in meta_features.items():
        meta_features_df = pd.concat([meta_features_df, value[0]], axis= 1, ignore_index=True)

    # make meta features frame.........................................................
    new_column_names = [i for i in   range(  meta_features_df.shape[1] )]  
    meta_features_df.columns = new_column_names
    return meta_features_df, y_test

def stacked_model_object_dictAND_accuracy_dict(X_train, y_train, X_test, y_test, models_object_dict, accuracy_objects_dict):

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    

    result_dataframe = dict()
    for key in models_object_dict.keys():
        print(f"*********************** Results with full data: {key} ***********************")
        models_object_dict[key].fit(X_train, y_train)
        

        y_predict = models_object_dict[key].predict(X_test)
        y_predict_prob = models_object_dict[key].predict_proba(X_test)

        X_test
          
        # print accuracy values on out of fold
        temp = dict()
        for acc_key, acc_object in accuracy_objects_dict.items():
            acc_object.compute(y_predict, y_test)
            print(acc_object.result())
            temp[acc_key] = round(acc_object.result()[1], 4)

        result_dataframe[str(key)] = temp

    return result_dataframe