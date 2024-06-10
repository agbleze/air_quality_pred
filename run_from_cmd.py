
import subprocess
import numpy as np
import wandb
import torch
import time
import os
import json
import math

hyperparameter_space = {
                        'learning_rate': np.random.uniform(low=0.03, high=1, size=100),
                        'max_leaf_nodes': np.random.randint(low=20, high=300, size=250),
                        'min_samples_leaf': np.random.randint(low=25, high=300, size=250),
                        'l2_regularization': np.random.uniform(low=0.01, high=1, size=100),
                        'max_iter': np.random.randint(low=200, high=700, size=500)
            }



categorical_feature=['month','day_name', 
                    'quater', 'month_pm25_rate',
                    'ismiss_sulphurdioxide_so2_column_number_density_15km',
                    'ismiss_nitrogendioxide_tropospheric_no2_column_number_density',
                    'ismiss_ozone_o3_slant_column_number_density',
                    'ismiss_formaldehyde_hcho_slant_column_number_density',
                    'ismiss_sulphurdioxide_so2_column_number_density_amf',
                    'ismiss_carbonmonoxide_co_column_number_density',
                    'ismiss_uvaerosolindex_absorbing_aerosol_index',
                    'ismiss_cloud_cloud_fraction', 'ismiss_cloud_surface_albedo',
                    'is_holiday'
                    ]
feature_name=['ozone_o3_slant_column_number_density',
                'uvaerosolindex_absorbing_aerosol_index','global_has_outlier_proba', 'out_nonout_ratio', 
                'has_outlier_proba', 'day_of_year',
                ]
            
          
learning_rate=0.003991140816697606
num_leaves=958
subsample= 0.25537284749459654
colsample_bytree= 0.9879171166038214
min_data_in_leaf= 18
objective="regression"
metric="rmse"
n_estimators=2000
#verbosity=-1
bagging_freq=1
device="cuda"
gpu_platform_id=0
gpu_device_id=0
num_threads=20
direction="minimize"
bagging_fraction=1
#save_binary=true
train_data="/home/lin/air_quality_pred/train_lgbdata.csv"
test_data="/home/lin/air_quality_pred/test_lgbdata.csv"
header=True
extra_trees = True
boosting = "dart"

_learning_rate= np.random.uniform(low=0.0001, high=0.1, size=10000)
_num_leaves= np.random.randint(low=2, high=2**10, size=500000)
_min_data_in_leaf=np.random.randint(low=2, high=2000, size=2000)
#_max_bin=np.random.randint(low=50, high=800, size=700)
_colsample_bytree=np.random.uniform(low=0.1, high=1.0, size=10000)
#_path_smooth = np.random.randint(low=0, high=10, size=10)
#_cat_l2 = np.random.randint(low=10, high=300, size=300)
#_cat_smooth = np.random.randint(low=10, high=300, size=300)
#_lambda_l2 = np.random.uniform(low=0.01, high=1, size=100)
_subsample= np.random.uniform(low=0.1, high=1, size=10000)



#_max_iter = np.random.randint(low=200, high=700, size=500)


# consider setting weight_column to proba of outlier column
# categorical_feature 
with open("/home/lin/LightGBM/cv_splitdata_path.json", "r") as file:
    cv_pathdata = json.load(file)
# %%
# for cvpath in cv_pathdata["cv_paths"]:
#     #for trnpath, tsnpath in zip(cvpath):
#     #print(cvpath[0], tsnpath[1])
#     trn_path, tsn_path = cvpath
#     print(trn_path)
#     print(tsn_path)
    
subprocess.call(["killall", "-9", "lightgbm"])
torch.cuda.empty_cache()
#torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = False
project = "airquality_pm2_5_prediction"
wandb.login()
rmse_list = []
hyperparam_list = []
num_cv = 10
num_trials = 300
tags = ["10 CV "]
for i in range(1, num_trials+1):
    learning_rate = np.random.choice(_learning_rate)
    num_leaves = np.random.choice(_num_leaves)
    subsample = np.random.choice(_subsample)
    min_data_in_leaf = np.random.choice(_min_data_in_leaf)
    colsample_bytree = np.random.choice(_colsample_bytree)
    #max_bin = np.random.choice(_max_bin)
    #path_smooth = np.random.choice(_path_smooth)
    #cat_l2 = np.random.choice(_cat_l2)
    #cat_smooth = np.random.choice(_cat_smooth)
    #lambda_l2 = np.random.choice(_lambda_l2)
    model_name = f"CV_LightGBM_model_{boosting}_{i}.txt"
    hyperparameter_space = {
                        'learning_rate': learning_rate,
                        'num_leaves': num_leaves,
                        'min_samples_leaf': min_data_in_leaf,
                        'colsample_bytree': colsample_bytree,
                        "subsample": subsample,
                        "n_estimators": n_estimators,
                        "bagging_freq": bagging_freq,
                        #"bagging_fraction": bagging_fraction,
                        #"extra_trees": extra_trees,
                        #"max_bin": max_bin,
                        "model_name": model_name,
                        "regressor": "lgbr"+boosting,
                        #"path_smooth": path_smooth,
                        #"cat_l2": cat_l2,
                        #"cat_smooth": cat_smooth,
                        #"lambda_l2": lambda_l2,
            }
    #print(hyperparameter_space)
    cv_rmse = []
    for fold, cvpath in enumerate(cv_pathdata["cv_paths"]):
        if fold+1 <= num_cv:
            #print(f"fold: {fold}")
            train_data, test_data = cvpath
            print(f"train_data: {train_data}")
            print(f"test_data: {test_data}")
            
            res= subprocess.check_output(["./lightgbm", "task=train",
                            #"config=/home/lin/LightGBM/config_gpu.conf",
                            "label_column=name:pm2_5",
                            f"boosting={boosting}",
                            f"learning_rate={learning_rate}",
                            f"num_leaves={num_leaves}",
                            f"min_data_in_leaf={min_data_in_leaf}",
                            f"subsample={subsample}",
                            f"colsample_bytree={colsample_bytree}",
                            #f"max_bin={max_bin}",
                            f"metric={metric}",
                            f"objective={objective}",
                            f"n_estimators={n_estimators}",
                            #verbosity=-1
                            f"bagging_freq={bagging_freq}",
                            #f"bagging_fraction={bagging_fraction}",
                            f"device={device}",
                            #f"gpu_platform_id={gpu_platform_id}",
                            #f"gpu_device_id={gpu_device_id}",
                            f"num_threads={num_threads}",
                            f"direction={direction}",
                            #save_binary=true
                            f"train_data={train_data}",
                            f"test_data={test_data}",
                            f"header=true",
                            #f"random_state={42}",
                            #f"num_threads={1}",
                            #f"extra_trees={extra_trees}",
                            f"output_model=/home/lin/LightGBM/cv_output_model__/{model_name}",
                            #f"path_smooth={path_smooth}",
                            #f"cat_l2={cat_l2}",
                            #f"cat_smooth={cat_smooth}",
                            #f"lambda_l2={lambda_l2}",
                            "categorical_feature=name:month,day_name,quater,month_pm25_rate"
                            
                            # "max_bin=63","num_leaves=255",
                            # "num_iterations = 50", "learning_rate = 0.1",
                            # "tree_learner = serial", "task = train",
                            # "is_training_metric = false",
                            # "min_data_in_leaf = 1", 
                            # "min_sum_hessian_in_leaf = 100",
                            # "ndcg_eval_at = 1,3,5,10",
                            # "device = cuda", "num_threads=20",
                            #"convert_model_language=cpp",
                            #"train_data=/home/lin/LightGBM/train_lgbdata.csv",
                            #"test_data=/home/lin/LightGBM/test_lgbdata.csv",
                            #  "gpu_platform_id = 0",
                            #  "objective=binary",
                            #  "metric=auc"
                            ])
            
            #print(f"res: {res}")
            
            try:
                rmse = np.float32(res.decode().split("rmse")[-1].split("\n")[0].split(":")[-1].split(" ")[-1])
                print(f"Trial {i}: fold {fold+1} rmse: {rmse}")
                cv_rmse.append(rmse)
            except ValueError:
                tags.append("ValueError occurred in rmse collection")
                print(f"Trial {i}: fold {fold+1} rmse: ValueError occurred in rmse collection")
    avg_cv_rmse = np.mean(cv_rmse)
    if math.isnan(avg_cv_rmse): #isinstance(avg_cv_rmse, np.nan):
        print(f"avg_cv_rmse is nan hence not uploaded")
    else:
        hyperparameter_space["rmse"] = avg_cv_rmse
        wandb.init(project=project,
                    config=hyperparameter_space,
                    reinit=True, save_code=True,
                    notes="CV with is_outlier used for stratified 20 fold splitting on all data of features with less than 2% missing values. Narrowed down param search based on initial training",
                    tags=tags
                    )
        wandb.log(hyperparameter_space)#({"rmse": rmse})
    rmse_list.append(avg_cv_rmse)
    hyperparam_list.append(hyperparameter_space)
    print(f"Trial {i}: {num_cv} fold CV AVG rmse: {avg_cv_rmse}")
    hyperparam_output_name = os.path.join("/home/lin/LightGBM/cv_hyperparam_output__", f"cv_hyperparam_{boosting}_{i}.json")
    with open(hyperparam_output_name, "w") as file:
        json.dump(hyperparameter_space, file, default=str)
    subprocess.call(["killall", "-9", "lightgbm"])
    #torch.cuda.set_device(0)
    #torch.cuda.empty_cache()
    #torch.cuda.set_device(1)
    #torch.backends.cudnn.benchmark = False
    #time.sleep(60)
    #print("waiting to start after 1 min")
    #time.sleep(60)
    #print("waiting to start after 2 min")
    #time.sleep(180)
    #print("waiting to start after 5 min")
#print(res.rstrip())
wandb.finish()
training_res = {"rmse": rmse_list, "params": hyperparam_list}

training_output = os.path.join("/home/lin/LightGBM/Output_model", "cv_training_res.json")

with open(training_output, "w") as file:
    json.dump(training_res, file, default=str)


#egoutput = b'[LightGBM] [Info] Finished loading parameters\n[LightGBM] [Warning] Using sparse features with CUDA is currently not supported.\n[LightGBM] [Info] Construct bin mappers from text data time 0.60 seconds\n[LightGBM] [Warning] Metric auc is not implemented in cuda version. Fall back to evaluation on CPU.\n[LightGBM] [Info] Finished loading data in 7.953604 seconds\n[LightGBM] [Info] Number of positive: 5564616, number of negative: 4935384\n[LightGBM] [Info] Total Bins 1524\n[LightGBM] [Info] Number of data points in the train set: 10500000, number of used features: 28\n[LightGBM] [Info] Finished initializing training\n[LightGBM] [Info] Started training...\n[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.529963 -> initscore=0.119997\n[LightGBM] [Info] Start training from score 0.119997\n[LightGBM] [Info] Iteration:1, valid_1 auc : 0.771835\n[LightGBM] [Info] 0.184250 seconds elapsed, finished iteration 1\n[LightGBM] [Info] Iteration:2, valid_1 auc : 0.778339\n[LightGBM] [Info] 0.359602 seconds elapsed, finished iteration 2\n[LightGBM] [Info] Iteration:3, valid_1 auc : 0.781228\n[LightGBM] [Info] 0.558983 seconds elapsed, finished iteration 3\n[LightGBM] [Info] Iteration:4, valid_1 auc : 0.783745\n[LightGBM] [Info] 0.707892 seconds elapsed, finished iteration 4\n[LightGBM] [Info] Iteration:5, valid_1 auc : 0.786385\n[LightGBM] [Info] 0.861660 seconds elapsed, finished iteration 5\n[LightGBM] [Info] Iteration:6, valid_1 auc : 0.787995\n[LightGBM] [Info] 1.023971 seconds elapsed, finished iteration 6\n[LightGBM] [Info] Iteration:7, valid_1 auc : 0.789627\n[LightGBM] [Info] 1.202551 seconds elapsed, finished iteration 7\n[LightGBM] [Info] Iteration:8, valid_1 auc : 0.791183\n[LightGBM] [Info] 1.369660 seconds elapsed, finished iteration 8\n[LightGBM] [Info] Iteration:9, valid_1 auc : 0.792727\n[LightGBM] [Info] 1.519339 seconds elapsed, finished iteration 9\n[LightGBM] [Info] Iteration:10, valid_1 auc : 0.794131\n[LightGBM] [Info] 1.679229 seconds elapsed, finished iteration 10\n[LightGBM] [Info] Iteration:11, valid_1 auc : 0.795505\n[LightGBM] [Info] 1.859134 seconds elapsed, finished iteration 11\n[LightGBM] [Info] Iteration:12, valid_1 auc : 0.79664\n[LightGBM] [Info] 2.111572 seconds elapsed, finished iteration 12\n[LightGBM] [Info] Iteration:13, valid_1 auc : 0.797967\n[LightGBM] [Info] 2.267165 seconds elapsed, finished iteration 13\n[LightGBM] [Info] Iteration:14, valid_1 auc : 0.799067\n[LightGBM] [Info] 2.433516 seconds elapsed, finished iteration 14\n[LightGBM] [Info] Iteration:15, valid_1 auc : 0.800821\n[LightGBM] [Info] 2.606518 seconds elapsed, finished iteration 15\n[LightGBM] [Info] Iteration:16, valid_1 auc : 0.801777\n[LightGBM] [Info] 2.765907 seconds elapsed, finished iteration 16\n[LightGBM] [Info] Iteration:17, valid_1 auc : 0.802916\n[LightGBM] [Info] 2.908901 seconds elapsed, finished iteration 17\n[LightGBM] [Info] Iteration:18, valid_1 auc : 0.803837\n[LightGBM] [Info] 3.073549 seconds elapsed, finished iteration 18\n[LightGBM] [Info] Iteration:19, valid_1 auc : 0.804934\n[LightGBM] [Info] 3.219513 seconds elapsed, finished iteration 19\n[LightGBM] [Info] Iteration:20, valid_1 auc : 0.805733\n[LightGBM] [Info] 3.392590 seconds elapsed, finished iteration 20\n[LightGBM] [Info] Iteration:21, valid_1 auc : 0.806726\n[LightGBM] [Info] 3.588942 seconds elapsed, finished iteration 21\n[LightGBM] [Info] Iteration:22, valid_1 auc : 0.807533\n[LightGBM] [Info] 3.772316 seconds elapsed, finished iteration 22\n[LightGBM] [Info] Iteration:23, valid_1 auc : 0.808295\n[LightGBM] [Info] 3.934672 seconds elapsed, finished iteration 23\n[LightGBM] [Info] Iteration:24, valid_1 auc : 0.809225\n[LightGBM] [Info] 4.103556 seconds elapsed, finished iteration 24\n[LightGBM] [Info] Iteration:25, valid_1 auc : 0.809843\n[LightGBM] [Info] 4.302463 seconds elapsed, finished iteration 25\n[LightGBM] [Info] Iteration:26, valid_1 auc : 0.810604\n[LightGBM] [Info] 4.473421 seconds elapsed, finished iteration 26\n[LightGBM] [Info] Iteration:27, valid_1 auc : 0.811235\n[LightGBM] [Info] 4.680212 seconds elapsed, finished iteration 27\n[LightGBM] [Info] Iteration:28, valid_1 auc : 0.811824\n[LightGBM] [Info] 4.849564 seconds elapsed, finished iteration 28\n[LightGBM] [Info] Iteration:29, valid_1 auc : 0.812494\n[LightGBM] [Info] 4.987879 seconds elapsed, finished iteration 29\n[LightGBM] [Info] Iteration:30, valid_1 auc : 0.81307\n[LightGBM] [Info] 5.143556 seconds elapsed, finished iteration 30\n[LightGBM] [Info] Iteration:31, valid_1 auc : 0.813657\n[LightGBM] [Info] 5.297795 seconds elapsed, finished iteration 31\n[LightGBM] [Info] Iteration:32, valid_1 auc : 0.814262\n[LightGBM] [Info] 5.442129 seconds elapsed, finished iteration 32\n[LightGBM] [Info] Iteration:33, valid_1 auc : 0.814724\n[LightGBM] [Info] 5.613432 seconds elapsed, finished iteration 33\n[LightGBM] [Info] Iteration:34, valid_1 auc : 0.815175\n[LightGBM] [Info] 5.777910 seconds elapsed, finished iteration 34\n[LightGBM] [Info] Iteration:35, valid_1 auc : 0.815625\n[LightGBM] [Info] 5.966608 seconds elapsed, finished iteration 35\n[LightGBM] [Info] Iteration:36, valid_1 auc : 0.816063\n[LightGBM] [Info] 6.134564 seconds elapsed, finished iteration 36\n[LightGBM] [Info] Iteration:37, valid_1 auc : 0.816476\n[LightGBM] [Info] 6.281741 seconds elapsed, finished iteration 37\n[LightGBM] [Info] Iteration:38, valid_1 auc : 0.816919\n[LightGBM] [Info] 6.446711 seconds elapsed, finished iteration 38\n[LightGBM] [Info] Iteration:39, valid_1 auc : 0.817325\n[LightGBM] [Info] 6.645830 seconds elapsed, finished iteration 39\n[LightGBM] [Info] Iteration:40, valid_1 auc : 0.817732\n[LightGBM] [Info] 6.863530 seconds elapsed, finished iteration 40\n[LightGBM] [Info] Iteration:41, valid_1 auc : 0.818117\n[LightGBM] [Info] 7.024245 seconds elapsed, finished iteration 41\n[LightGBM] [Info] Iteration:42, valid_1 auc : 0.818484\n[LightGBM] [Info] 7.182456 seconds elapsed, finished iteration 42\n[LightGBM] [Info] Iteration:43, valid_1 auc : 0.818831\n[LightGBM] [Info] 7.362253 seconds elapsed, finished iteration 43\n[LightGBM] [Info] Iteration:44, valid_1 auc : 0.819217\n[LightGBM] [Info] 7.523567 seconds elapsed, finished iteration 44\n[LightGBM] [Info] Iteration:45, valid_1 auc : 0.81956\n[LightGBM] [Info] 7.696196 seconds elapsed, finished iteration 45\n[LightGBM] [Info] Iteration:46, valid_1 auc : 0.819865\n[LightGBM] [Info] 7.897500 seconds elapsed, finished iteration 46\n[LightGBM] [Info] Iteration:47, valid_1 auc : 0.82025\n[LightGBM] [Info] 8.057357 seconds elapsed, finished iteration 47\n[LightGBM] [Info] Iteration:48, valid_1 auc : 0.820665\n[LightGBM] [Info] 8.238172 seconds elapsed, finished iteration 48\n[LightGBM] [Info] Iteration:49, valid_1 auc : 0.820917\n[LightGBM] [Info] 8.391436 seconds elapsed, finished iteration 49\n[LightGBM] [Info] Iteration:50, valid_1 auc : 0.821268\n[LightGBM] [Info] 8.552252 seconds elapsed, finished iteration 50\n[LightGBM] [Info] Finished training'
# 


#
