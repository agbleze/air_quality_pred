
#%%
from pytorch_tabular.utils import load_covertype_dataset
data, _, _, _ = load_covertype_dataset()
from rich import print
# One of the easiest ways to identify categorical features is using the pandas select_dtypes function.
categorical_features = data.select_dtypes(include=['object'])
print(categorical_features.columns)
# %%
# Another way is to use the unique values in each column.
for col in data.columns:
    print(col, len(data[col].unique()))
# %%
# This separation have already been done for you while loading this particular dataset from `PyTorch Tabular`. Let's load the dataset in the right way.
data, cat_col_names, num_col_names, target_col = load_covertype_dataset()
# Let's also print out a few details
print(f"Data Shape: {data.shape} | # of cat cols: {len(cat_col_names)} | # of num cols: {len(num_col_names)}")
print(f"[bold dodger_blue2] Features: {num_col_names + cat_col_names}[/bold dodger_blue2]")
print(f"[bold purple4]Target: {target_col}[/bold purple4]")
# %%
# Let's also check the data for missing values
print(data.isna().sum())
# %%
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, random_state=42, test_size=0.2)
train, val = train_test_split(train, random_state=42, test_size=0.2)
print(f"Train Shape: {train.shape} | Val Shape: {val.shape} | Test Shape: {test.shape}")
# %%
from pytorch_tabular.models import GANDALFConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)

data_config = DataConfig(
    target=[
        target_col
    ],  # target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
    continuous_cols=num_col_names,
    categorical_cols=cat_col_names,
    num_workers=19
)
trainer_config = TrainerConfig(
    batch_size=1024,
    max_epochs=1,
)
optimizer_config = OptimizerConfig()
model_config = GANDALFConfig(
    task="classification",
    gflu_stages=6,
    gflu_feature_init_sparsity=0.3,
    gflu_dropout=0.0,
    learning_rate=1e-3,
)
# %%
from pytorch_tabular import TabularModel

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    verbose=True
)
# %%
tabular_model.fit(train=train, validation=val)
# %%
pred_df = tabular_model.predict(test)
pred_df.head()

#%%
result = tabular_model.evaluate(test)

#%%
tabular_model.save_model("examples/basic")

#%%
loaded_model = TabularModel.load_model("examples/basic")


# %%
from utils.get_path import get_data_path
import pandas as pd

#%%
train_path = get_data_path(folder_name="zindi_pm25_pred", file_name="Train.csv")
test_path = get_data_path(folder_name="zindi_pm25_pred", file_name="Test.csv")
sample_path = get_data_path(folder_name="zindi_pm25_pred", file_name="SampleSubmission.csv")
# %%
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_df = pd.read_csv(sample_path)
# %%
from helper import (create_date_features, calculate_day_per_month_outier_non_outlier_ratio, 
                    calculate_global_proba_day_has_isoutier,calculate_monthly_outlier_nonoutlier_ratio, 
                    calculate_proba_day_per_month_isoutier, create_is_outlier, create_month_emission_rate,
                    add_outlier_probability_features, create_missing_value_features,
                    replace_all_missing_values,
                    OutlierImputer, ParameterImputation
                    )
# %%
train_with_datefeature = create_date_features(data=train_df)
train_with_mnth_emission_rate = create_month_emission_rate(data=train_with_datefeature)

# %%
train_with_outlierfeature, outlier_imputer = create_is_outlier(data=train_with_mnth_emission_rate, 
                                                                colname_to_detect_outlier="pm2_5"
                                                                )

# %%
dayofmnth_outliercnt = train_with_outlierfeature.groupby(["month","day_name"])["is_outlier"].value_counts().reset_index()
mth_outlier_ct = train_with_outlierfeature.groupby("month")["is_outlier"].value_counts().reset_index()


#%%
#month_out_nonout_ratio_df = calculate_monthly_outlier_nonoutlier_ratio(mth_outlier_ct=mth_outlier_ct)

day_month_out_nonout_ratio_df= calculate_day_per_month_outier_non_outlier_ratio(dayofmnth_outliercnt)

day_month_proba_df = calculate_proba_day_per_month_isoutier(dayofmnth_outliercnt=dayofmnth_outliercnt)

#
global_proba_day_has_isoutier_df = calculate_global_proba_day_has_isoutier(data=train_with_outlierfeature)



# %%
train_with_proba = add_outlier_probability_features(data=train_with_outlierfeature,
                                 global_proba_day_has_isoutier_df=global_proba_day_has_isoutier_df,
                                 day_month_out_nonout_ratio_df=day_month_out_nonout_ratio_df,
                                 day_month_proba_df=day_month_proba_df
                                 )
# %%
train_with_missfeat = create_missing_value_features(data=train_with_proba)

#%%
empty_feature = ["ozone_o3_slant_column_number_density", "uvaerosolindex_absorbing_aerosol_index"]

# %%
param_imp_obj = ParameterImputation(data=train_with_missfeat, 
                                    aggregation_type="mean",
                                    aggregation_var="month", 
                                    param=empty_feature
                                    )

median_value_per_mnth = param_imp_obj.get_df_for_all_params()
# %%
train_with_imputed_missing = param_imp_obj.replace_all_missing_values(empty_feature=empty_feature)


# %%
median_month_imputation = param_imp_obj.df_dict
# %%
for imput_feat in median_month_imputation.keys():
    median_month_imputation[imput_feat].to_csv(f"{imput_feat}.csv", index=False)
    
# %%
selected_variables = ['month',
                'ozone_o3_slant_column_number_density',
                'uvaerosolindex_absorbing_aerosol_index',
                'day_name',
                'day_of_year',
                'quater',
                'month_pm25_rate',
                'ismiss_sulphurdioxide_so2_column_number_density_15km',
                'ismiss_nitrogendioxide_tropospheric_no2_column_number_density',
                'ismiss_ozone_o3_slant_column_number_density',
                'ismiss_formaldehyde_hcho_slant_column_number_density',
                'ismiss_sulphurdioxide_so2_column_number_density_amf',
                'ismiss_carbonmonoxide_co_column_number_density',
                'ismiss_uvaerosolindex_absorbing_aerosol_index',
                'ismiss_cloud_cloud_fraction',
                'ismiss_cloud_surface_albedo',
                'global_has_outlier_proba',
                'out_nonout_ratio',
                'has_outlier_proba','pm2_5'
            ]

target_col = ['pm2_5']
categorical_features = ['month',
                        'day_name',
                        'day_of_year',
                        'quater',
                        'month_pm25_rate',
                        'ismiss_sulphurdioxide_so2_column_number_density_15km',
                        'ismiss_nitrogendioxide_tropospheric_no2_column_number_density',
                        'ismiss_ozone_o3_slant_column_number_density',
                        'ismiss_formaldehyde_hcho_slant_column_number_density',
                        'ismiss_sulphurdioxide_so2_column_number_density_amf',
                        'ismiss_carbonmonoxide_co_column_number_density',
                        'ismiss_uvaerosolindex_absorbing_aerosol_index',
                        'ismiss_cloud_cloud_fraction',
                        'ismiss_cloud_surface_albedo'
                        ]
continuous_features = [
                'ozone_o3_slant_column_number_density',
                'uvaerosolindex_absorbing_aerosol_index',
                'global_has_outlier_proba',
                'out_nonout_ratio',
                'has_outlier_proba'
                ]

prepared_train_df = train_with_missfeat[selected_variables]
# %%
from pytorch_tabular.config import (DataConfig, OptimizerConfig, TrainerConfig,
                                    ModelConfig, ExperimentConfig
                                    )
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.models.common.heads import LinearHead, LinearHeadConfig
from pytorch_tabular import TabularModel,  MODEL_SWEEP_PRESETS, model_sweep
from rich import print
import wandb
from pytorch_lightning.callbacks import Timer
from datetime import timedelta
from sklearn.model_selection import train_test_split


#%%

prepared_train_split, prepared_test_split = train_test_split(prepared_train_df, test_size=.2, 
                 random_state=42, 
                 stratify=prepared_train_df["month_pm25_rate"],)
#%%
timer = Timer(duration=timedelta(minutes=2))
#%%
wandb.login()
#%%
data_config = DataConfig(target=target_col, continuous_cols=continuous_features,
                            categorical_cols=categorical_features,
                            continuous_feature_transform="quantile_normal",
                            normalize_continuous_features=True,
                            num_workers=20,
                            pin_memory=True
                            )

#timer = Timer(duration="00:00:01:00")
train_config = TrainerConfig(batch_size=32,
                             accelerator="gpu", devices=-1,
                            max_epochs=200, 
                            min_epochs=200,
                            early_stopping_patience=200,
                            #max_time=1,
                            #fast_dev_run=True,
                            auto_lr_find=True,
                            checkpoints_path="saved_model_split"
                            )

optimizer_config = OptimizerConfig()

target_min = train_with_imputed_missing["pm2_5"].min() * 0.8
target_max = train_with_imputed_missing["pm2_5"].max() * 1.2

head_config = LinearHeadConfig(layers="", dropout=0.1,
                               initialization="kaiming"
                               ).__dict__

model_config = CategoryEmbeddingModelConfig(task="regression",
                                            target_range=[(target_min, target_max)],
                                            #virtual_batch_size=2, 
                                            layers="128-64-32-16", activation="LeakyReLU",
                                            initialization="kaiming", head="LinearHead", head_config=head_config,
                                            
                                            seed=42)
# %%
tabular_model = TabularModel(data_config=data_config, model_config=model_config,
                            optimizer_config=optimizer_config,
                            trainer_config=train_config, verbose=True,
                            suppress_lightning_logger=False
                            )

#%%
tabular_model.fit(train=prepared_train_split, validation=prepared_test_split)

#%%
tabular_model.summary()
#tabular_model.cross_validate(cv=5, train=prepared_train_df)

#%%
tabular_model.
#%%
tabular_model.load_best_model()

#%%
tabular_model.save_model(dir="/home/lin/codebase/air_quality_pred/completed_saved_split_model")

#%%
tabular_model.save_model_for_inference("/home/lin/codebase/air_quality_pred/completed_saved_split_model/model.pt")

#%%

submission_test_df_date_feat = create_date_features(data=test_df)
submission_test_df_with_missing_feat = create_missing_value_features(data=submission_test_df_date_feat)
submission_test_df_with_features = create_month_emission_rate(submission_test_df_with_missing_feat)


#%%
monthly_median_ozone_o3_slant_column_number_density = pd.read_csv("ozone_o3_slant_column_number_density.csv")
monthly_median_uvaerosolindex_absorbing_aerosol_index = pd.read_csv("uvaerosolindex_absorbing_aerosol_index.csv")

#%%
median_imput_dict = {"ozone_o3_slant_column_number_density": monthly_median_ozone_o3_slant_column_number_density,
   "uvaerosolindex_absorbing_aerosol_index": monthly_median_uvaerosolindex_absorbing_aerosol_index
}

empty_feature = ["ozone_o3_slant_column_number_density", "uvaerosolindex_absorbing_aerosol_index"]

#%%
global_proba_day_has_isoutier_df.to_csv("global_proba_day_has_isoutier_df.csv", index=False)
day_month_out_nonout_ratio_df.to_csv("day_month_out_nonout_ratio_df.csv", index=False)
day_month_proba_df.to_csv("day_month_proba_df.csv", index=False)

#%%

global_proba_day_has_isoutier_df_rd = pd.read_csv("global_proba_day_has_isoutier_df.csv")
day_month_out_nonout_ratio_df_rd = pd.read_csv("day_month_out_nonout_ratio_df.csv")
day_month_proba_df_rd = pd.read_csv("day_month_proba_df.csv")
#%%
submission_test_df_with_features = replace_all_missing_values(empty_feature=empty_feature, df_dict=median_imput_dict,
                                                                data=submission_test_df_with_features
                                                                )



#%%
submission_test_df_with_outlier_features = (submission_test_df_with_features.merge(right=global_proba_day_has_isoutier_df_rd, 
                                                        left_on="month_day", right_on="month_day",
                                                        ).merge(right=day_month_out_nonout_ratio_df_rd,
                                                        left_on="month_day",
                                                        right_on="month_day"
                                                        ).merge(right=day_month_proba_df_rd,
                                                            left_on="month_day",
                                                            right_on="month_day"
                                                            )
                                    )


#%%

preddata_cols = prepared_train_df.columns

prepared_test_data = submission_test_df_with_outlier_features[preddata_cols[:-1]]

#%%
test_predictions = tabular_model.predict(test=prepared_test_data)

#%%

test_predictions.describe()

#%%
submission_df = test_df.copy()
#%%
submission_df["pm2_5"] = test_predictions["pm2_5_prediction"]

#%%
submission_df[["id", "pm2_5"]].to_csv("/home/lin/codebase/air_quality_pred/submissions/submission21.csv", index=False)

#%%

test_df.drop(labels="pm2_5", inplace=True, axis=1)

#%%
tabular_model.feature_importance("/home/lin/codebase/air_quality_pred/infer_models")
#%% 
tabular_model.load_weights(path="/home/lin/codebase/air_quality_pred/saved_models/regression-6_epoch=16-valid_loss=45.61.ckpt")
# %%
#tabular_model.bagging_predict()



#%%

print(list(MODEL_SWEEP_PRESETS["full"]))
#%% using model sweep
import warnings
import wandb
from sklearn.model_selection import train_test_split

train_model_sweep_df, test_model_sweep_df = train_test_split(prepared_train_df, test_size=0.2,
                                                             random_state=42)
project_name = "airquality_pm2_5_prediction"
wandb.login()
experiment_config = ExperimentConfig(project_name=project_name,
                                    log_target="wandb", exp_watch="all"#,
                                    #log_logits=True
                                    )
with warnings.catch_warnings():
    #warnings.simplefilter("ignore")
    sweep_df, best_model = model_sweep(task="regression",
                                        train=train_model_sweep_df,
                                        test=test_model_sweep_df,
                                        data_config=data_config,
                                        optimizer_config=optimizer_config,
                                        trainer_config=train_config,
                                        model_list="standard",
                                        common_model_args=dict(head="LinearHead", 
                                                               head_config=head_config
                                                               ),
                                        experiment_config=experiment_config
                                        )
# %%
sweep_df
# %%
sweep_df.drop(columns=["params", "time_taken", "epochs"]).style.background_gradient(
    subset=["test_mean_squared_error"], cmap="RdYlGn"
).background_gradient(subset=["time_taken_per_epoch", "test_loss"], cmap="RdYlGn_r")
