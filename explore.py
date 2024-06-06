
#%%
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from utils.get_path import get_data_path

#%%
train_path = get_data_path(folder_name="zindi_pm25_pred", file_name="Train.csv")
test_path = get_data_path(folder_name="zindi_pm25_pred", file_name="Test.csv")
sample_path = get_data_path(folder_name="zindi_pm25_pred", file_name="SampleSubmission.csv")
# %%
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_df = pd.read_csv(sample_path)
# %%
train_df
# %%
test_df
# %%
train_df.info()
# %%
train_df.describe()
# %%
mask = train_df.isnull().any()
no_missing_cl = train_df.columns[~mask]
no_missing_data_features = no_missing_cl[1:5]


#%%
train_df[['uvaerosolindex_absorbing_aerosol_index']].describe()

#%%
train_df[['cloud_cloud_fraction', 'cloud_surface_albedo']].describe()


#%%

train_df[['cloud_cloud_fraction', 'formaldehyde_cloud_fraction',
          "nitrogendioxide_cloud_fraction"
          ]].describe()


#%%

train_df[['formaldehyde_cloud_fraction',
          "nitrogendioxide_cloud_fraction"]].tail(20)


#%%
selected_features = ['sulphurdioxide_so2_column_number_density',
       'sulphurdioxide_so2_column_number_density_amf',
       'sulphurdioxide_so2_slant_column_number_density',
       'sulphurdioxide_so2_column_number_density_15km',
       'month',
       'carbonmonoxide_co_column_number_density',
       'carbonmonoxide_h2o_column_number_density',
       'nitrogendioxide_no2_column_number_density',
       'nitrogendioxide_tropospheric_no2_column_number_density',
       'nitrogendioxide_stratospheric_no2_column_number_density',
       'nitrogendioxide_no2_slant_column_number_density',
       'nitrogendioxide_tropopause_pressure',
       'formaldehyde_tropospheric_hcho_column_number_density',
       'formaldehyde_tropospheric_hcho_column_number_density_amf',
       'formaldehyde_hcho_slant_column_number_density',
       'ozone_o3_column_number_density',
       'ozone_o3_column_number_density_amf',
       'ozone_o3_slant_column_number_density',
       'ozone_o3_effective_temperature',
       'cloud_cloud_fraction', 
       'cloud_surface_albedo',
       'uvaerosolindex_absorbing_aerosol_index',
       'pm2_5'
       
       ]

#%%
selected_train_df = train_df[selected_features]
#%%
features_with_highest_pm25_cor= ['carbonmonoxide_co_column_number_density',
                               'nitrogendioxide_no2_column_number_density',
                               'nitrogendioxide_tropospheric_no2_column_number_density',
                               'nitrogendioxide_no2_slant_column_number_density',
                               "pm2_5"
                               ]
#%%
## features showing some relationship with pm2_5 on scatter plot
feat_scatted_to_pm25 = ["formaldehyde_hcho_slant_column_number_density",  
                        'formaldehyde_tropospheric_hcho_column_number_density',  # not selected
                        'nitrogendioxide_no2_slant_column_number_density',  # not selected
                        'nitrogendioxide_tropospheric_no2_column_number_density',
                        'nitrogendioxide_no2_column_number_density',
                        'carbonmonoxide_co_column_number_density',
                        'sulphurdioxide_so2_column_number_density_amf',
                        ]

#%% # have missing values occuring at same places
corr_formal_feat = ['formaldehyde_tropospheric_hcho_column_number_density',
                    'formaldehyde_hcho_slant_column_number_density',  # choose because it has lsesser outliers but note
                                                                        # it has lower corr with 
                                                                        # pm2_5 compared with formaldehyde_tropospheric_hcho_column_number_density
                    ]

#%%
corr_no2_feat = ['nitrogendioxide_no2_slant_column_number_density',
 'nitrogendioxide_no2_slant_column_number_density',
 'nitrogendioxide_tropospheric_no2_column_number_density',  ## should be kept and other values used to impute it. It has higher corr to pm2_5
 ]

#%%
corr_ozone_feat = ['ozone_o3_column_number_density_amf',
                'ozone_o3_slant_column_number_density'
                ]

# ozone_o3_slant_column_number_density is chosen because it has a higher correlation 
# with pm2_5 compared to 'ozone_o3_column_number_density_amf'
# 'ozone_o3_column_number_density_amf' and ozone_o3_slant_column_number_density
# have correlation of 0.97

#%%
corr_so2_feat = ['sulphurdioxide_so2_column_number_density',
                'sulphurdioxide_so2_slant_column_number_density',
                'sulphurdioxide_so2_column_number_density_15km',  # choose because has least outliers and higest corr with pm2_5
                
                ]

#%%
'nitrogendioxide_tropospheric_no2_column_number_density' and 
 'nitrogendioxide_no2_column_number_density' look exactly the same, find their correlation
 and drop one of them. OR choose base of number of data points and outliers present


#%%
# pramaters with same names appear to have missing values at same places
# try replace values using unrelated params that have corr values 

#%%
corr_matrix = selected_train_df[['nitrogendioxide_tropospheric_no2_column_number_density', 
                                'nitrogendioxide_no2_column_number_density']].corr()

corr_matrix

# 'nitrogendioxide_tropospheric_no2_column_number_density' and 
# 'nitrogendioxide_no2_column_number_density' have correlation of 0.97
# hence only one of the should be use
# happens for other features
#%%
selected_train_df[['nitrogendioxide_tropospheric_no2_column_number_density', 
                                'nitrogendioxide_no2_column_number_density']].info()

#%%
# function to create boxplot
def make_boxplot(data: pd.DataFrame, variable_name: str):
    """This function accepts a data and variable name and returns a boxplot

    Args:
        data (pd.DataFrame): Data to visualize
        variable_name (str): variable to visualize with boxplot
    """
    fig = px.box(data_frame=data, y = variable_name,
                 template='plotly_dark', 
                 title = f'Boxplot to visualize outliers in {variable_name}'
                 )
    fig.show()
    
#%%
high_corvar = ['nitrogendioxide_tropospheric_no2_column_number_density', 
                'nitrogendioxide_no2_column_number_density'
                ]

#%%
for var in corr_formal_feat:
    make_boxplot(data=selected_train_df, variable_name=var)
    
#%%
for var in high_corvar:
    make_boxplot(data=selected_train_df, variable_name=var)

#%% for var in high_corvar, we will choose the one with lower outliers
class OutlierImputer(object):
  def __init__(self, data: pd.DataFrame, colname: str):
    self.data = data
    self.colname = colname
    self.first_quantile = self.data[self.colname].quantile(q=0.25)
    self.third_quantile = self.data[self.colname].quantile(q=0.75)
    self.inter_quantile_rng = 1.5*(self.third_quantile-self.first_quantile)
    self.upper_limit = self.inter_quantile_rng + self.third_quantile
    self.lower_limit = self.first_quantile - self.inter_quantile_rng

  @property
  def get_outlier_samples(self):
    outlier_samples = (self.data[(self.data[self.colname] > self.upper_limit) | 
                               (self.data[self.colname] < self.lower_limit)]
                              [[self.colname]]
                      )
    return outlier_samples



  def impute_outlier(self):
    self.outlier_data = self.data.copy()
    self.outlier_data[f'{self.colname}_outlier_imputed'] = (
                    np.where(self.outlier_data[self.colname] > self.upper_limit, 
                                                   self.upper_limit, 
                             np.where(self.outlier_data[self.colname] < self.lower_limit, 
                                                   self.lower_limit, 
                             self.outlier_data[self.colname]
                            )
                            )
                  )
    
    return self.outlier_data
    
#%%
# initializing the outlier imputer class
for var in high_corvar:
    outlier_imputer = OutlierImputer(data=selected_train_df, colname=var)
    print(outlier_imputer.get_outlier_samples)

## nitrogendioxide_tropospheric_no2_column_number_density is selected due to 
# lower number of outliers

#%%
for var in corr_formal_feat:
    outlier_imputer = OutlierImputer(data=selected_train_df, colname=var)
    print(outlier_imputer.get_outlier_samples)

#%%
#%%
for var in corr_so2_feat:
    outlier_imputer = OutlierImputer(data=selected_train_df, colname=var)
    print(outlier_imputer.get_outlier_samples.nunique())
    
#%%
selected_train_df[['sulphurdioxide_so2_column_number_density',
                'sulphurdioxide_so2_slant_column_number_density',
                'sulphurdioxide_so2_column_number_density_15km',
                'pm2_5'
                ]].corr()
#%%
selected_train_df[['ozone_o3_column_number_density_amf',
'ozone_o3_slant_column_number_density', "pm2_5"
]].corr()

# ozone_o3_slant_column_number_density is chosen because it has a higher correlation 
# with pm2_5 compared to 'ozone_o3_column_number_density_amf'
# 'ozone_o3_column_number_density_amf' and ozone_o3_slant_column_number_density
# have correlation of 0.97


#%% check correlation for all feat_scatted_to_pm25
corr_matrix = selected_train_df[selected_features].corr(method='spearman')

corr_matrix
#%% Create a mask to hide the upper triangle
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True

# visualize correlation matrix
sns.heatmap(corr_matrix, mask=mask, cmap=sns.color_palette("GnBu_d"), 
            square=True, linewidths=.5, cbar_kws={"shrink": .9},
            annot=True, annot_kws={"size": 5}, 
            yticklabels=1
            )

plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
plt.show()

#%% corr of highest correlated fest with pm25  ##########

selected_train_df[features_with_highest_pm25_cor].corr()


#%% ####### investigate missing data
# check if they offer any insights
# Total missing data as a percentage of all data points is estimated as follows
def get_missing_data_percent(data: pd.DataFrame, variable: str):
    total_missing = data[variable].isnull().sum()
    total_data = data.shape[0]
    percent_missing = (total_missing / total_data) * 100
    print(f'Percentage of data missing in {variable}: {round(percent_missing, 2)}%')

## implement function for percentage of missing data per variable
for variable_name in selected_train_df.columns:
    get_missing_data_percent(data=selected_train_df, variable=variable_name)

#%%
for variable_name in corr_ozone_feat:
    get_missing_data_percent(data=selected_train_df, variable=variable_name)

#%%
for variable_name in corr_formal_feat:
    get_missing_data_percent(data=selected_train_df, variable=variable_name)
    
#%%
for variable_name in corr_so2_feat:
    get_missing_data_percent(data=selected_train_df, variable=variable_name)
#%%

selected_train_df = train_df[selected_features]

#%%
for col in selected_train_df.columns:
    plot = px.box(data_frame=train_df,
            y=col,
            title=col,
            template='plotly_dark'
            )
    plot.show()

#%%
for col in selected_train_df.columns:
    plot = px.histogram(data_frame=train_df,
            x=col,
            title=col,
            template='plotly_dark'
            )
    plot.show()

#%%
for col in selected_train_df.columns:
    plot = px.bar(data_frame=train_df,
            x="month", y=col,
            title=col,
            template='plotly_dark'
            )
    plot.show()

#%% plot scatter plot
for col in selected_train_df.columns:
    if col != "pm2_5":
        plot = px.scatter(data_frame=train_df,
                x=col, y="pm2_5",
                title=col,
                template='plotly_dark'
                )
        plot.show()

#%%
        
#%%
def plot_scatterplot(data: pd.DataFrame,
                      x_colname: str,
                      y_colname: str = 'hits'
                      ):
    """ Scatterplot to visualize relationship between two variables. 
    Args:
        data (pd.DataFrame): Data which contains variables to plot
        
        y_colname (str): column name (variable) to plot on y-axis
        x_colname (str): column name (variable) to plot on x-axis
    """
    scatter_graph = (ggplot(data=data, mapping=aes(y=y_colname, x=x_colname)) 
                            + geom_point() + geom_smooth(method='lm')
                            + ggtitle(f'Scatter plot to visualize relationship between {y_colname} and {x_colname}')
                    )
    print(scatter_graph)


#%%
for var in selected_train_df.columns:
    if var != "pm2_5":
        print(var)
        plot_scatterplot(data=selected_train_df, 
                        x_colname=var, 
                        y_colname='pm2_5'
                        )

#%%
selected_train_df.nitrogendioxide_tropopause_pressure.nunique()

#%%
avg_pm_per_mnth =selected_train_df.groupby("month")["pm2_5"].aggregate("mean").reset_index()


#%%

px.bar(data_frame=avg_pm_per_mnth,
       x="month", y="pm2_5",
       title=col, color="month",
        template='plotly_dark'
        )

## explore other date features 
## cluster the data and 

#%%   ###########  fit model   #########
# import modules
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import (ggplot, ggtitle, aes, geom_col, theme_dark, 
                      theme_light, scale_x_discrete,
                      position_dodge, geom_text, element_text, theme,
                      geom_histogram, geom_bar, xlab, ylab, scale_y_log10, scale_x_log10,
                      geom_point, geom_smooth, geom_boxplot, coord_flip
                    )
import scipy.stats as stats
import pingouin as pg
import numpy as np
import ast
from bioinfokit.analys import stat as infostat
import seaborn as sns

import numpy as np
from scipy.stats import iqr
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import (train_test_split, 
                                     RandomizedSearchCV,
                                     cross_validate
                                     )
from argparse import Namespace
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import  HistGradientBoostingRegressor, BaggingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer

#%%
histgb = HistGradientBoostingRegressor(random_state=42)

y_target = selected_train_df[["pm2_5"]]

X_selected_features = selected_train_df.drop(columns="pm2_5")
#%%
histgb_cv = cross_validate(estimator=histgb,
                           X=X_selected_features, 
                            y=y_target,
                            cv=20, 
                            scoring='neg_root_mean_squared_error',
                            return_estimator=True,
                            return_train_score=True,
                            verbose=3
                        )

#%%
histgb_cv.keys()
#%%
histgb_cv_mean = histgb_cv['test_score'].mean()


print(f'20 fold Cross validation Negative RMSE is {histgb_cv_mean}')

#%%

X_train, X_test, y_train, y_test = train_test_split(X_selected_features, y_target, 
                                                    test_size=.3, 
                                                    random_state=42
                                                    )
#%%
histgb.fit(X=X_train, y=y_train)

train_rmse = mean_squared_error(y_true=y_train, y_pred=histgb.predict(X_train), squared=False)

print(f'Model training error (RMSE) is: {train_rmse: .3f}')

#%%
histgb.get_params()

#%%
test_rmse = mean_squared_error(y_true=y_test, 
                               y_pred=histgb.predict(X_test),
                               squared=False
                               )


print(f'Model test error (RMSE) is: {test_rmse: .3f}')

#%%
hyperparameter_space = {
                        'learning_rate': np.random.uniform(low=0.03, high=1, size=100),
                        'max_leaf_nodes': np.random.randint(low=20, high=300, size=250),
                        'min_samples_leaf': np.random.randint(low=25, high=300, size=250),
                        'l2_regularization': np.random.uniform(low=0.01, high=1, size=100),
                        'max_iter': np.random.randint(low=200, high=700, size=500)
            }

#%%
random_search = RandomizedSearchCV(estimator=histgb, param_distributions=hyperparameter_space,
                                  n_iter=1000,
                                  cv=20,
                                  refit=True,
                                  return_train_score=True, 
                                  n_jobs=-1,
                                  random_state=42,
                                  scoring='neg_root_mean_squared_error',
                                  verbose=4,
                                  
                                  )

#%%
random_search.fit(X=X_selected_features, y=y_target) 

#%%
random_search.best_score_

#%%

random_search_20cv = cross_validate(estimator=random_search.best_estimator_, 
                                        X=X_selected_features, 
                                        y=y_target, cv=20, n_jobs=-1, 
                                        scoring='neg_root_mean_squared_error'
                                    )

random_search_20cv['test_score'].mean()

#%%
random_20cv_mean = random_search_20cv['test_score'].mean()

print(f'20 fold Cross validation Negative RMSE is {random_20cv_mean: .3f}')

#%%
# evaluation of tuned model on test set
mean_squared_error(y_test, random_search.predict(X_test), squared=False)

#%%
improved_histboostgb = random_search.best_estimator_
hist_train_RMSE = mean_squared_error(y_train, improved_histboostgb.predict(X_train), squared=False)


hist_test_RMSE = mean_squared_error(y_test, improved_histboostgb.predict(X_test), squared=False)

print(f"The training RMSE for Best tuned HistGradientBoosting is: {hist_train_RMSE: .3f}")

print(f"The test RMSE for Best tuned HistGradientBoosting is: {hist_test_RMSE: .3f}")


#%%
random_search_20cv

#%%
random_search.best_params_


#%%
test_df_sub = test_df.copy()
test_df_sub["pm2_5"] =  improved_histboostgb.predict(test_df[selected_features[0:-1]])
#test_df[selected_features[0:-1]]

#%%
import os
subm_dir = "/home/lin/codebase/air_quality_pred/submissions"
subfile = os.path.join(subm_dir, "submission2.csv")
test_df_sub[["id", "pm2_5"]].to_csv(subfile, index=False)


#%%
model_path = get_data_path(folder_name="models", file_name="improved_histboostgb.model")
import joblib

joblib.dump(improved_histboostgb, model_path)


#%%  ########### USE ALL FEATURES AS IN DEFAULT   #########
train_num_df = train_df.select_dtypes(include=['number'])
y_num = train_num_df["pm2_5"]
X_num = train_num_df.drop(columns="pm2_5")


#%%
randomsearch_num20cv = cross_validate(estimator=random_search.best_estimator_, 
                                        X=X_num, 
                                        y=y_num, cv=20, n_jobs=-1, 
                                        scoring='neg_root_mean_squared_error',
                                        verbose=4,
                                        return_estimator=True,
                                        return_train_score=True
                                    )

#%%
randomsearch_num20cv["test_score"].mean()

#%%
randomsearch_num20cv["train_score"].mean()

#%%
X_train42rm, X_test42rm, y_train42rm, y_test42rm = train_test_split(X_num, y_num,
                                                                    test_size=.3,
                                                                    random_state=42
                                                                    )

#%%

mean_squared_error(y_true=y_test42rm, 
                   y_pred=randomsearch_num20cv["estimator"][0].predict(X_test42rm),
                   squared=False
                   )

#%%

num_cols = train_num_df.columns[:-1]
#%%
test_dfcp3 = test_df.copy()
test_dfcp3["pm2_5"] = randomsearch_num20cv["estimator"][0].predict(test_df[num_cols])
#test_df
#%%
subm3 = os.path.join(subm_dir, "submission3.csv")
test_dfcp3[["id", "pm2_5"]].to_csv(subm3, index=False)

#%%




"""
submission 3 improved model baseline model
"""

#%%
#%%
randomsearch_num20cv['test_score'].mean()


#%%  ################ Imputation of missing values based on hour, day, month

train_df.groupby("hour")["pm2_5"].aggregate("mean").reset_index()

#%%
params_with_miss = []
for var in selected_features:
    if var != "month":
        grp_df = train_df.groupby("month")[var].aggregate("median").rename({"month": "_month"}, axis=0).reset_index()[["month",var]]
        miss_exist = [True if grp_df.any().isnull().sum()  > 0 else False][0]
        if miss_exist:
            params_with_miss.append(var)
        print(f"{var}  Miss-value-exist: {miss_exist} \n")
        print(grp_df)


#%% write a transformation class for this  ----

class ParameterImputation(object):
    def __init__(self, data, aggregation_var, aggregation_type, param):
        self.data = data
        self.aggregation_var = aggregation_var
        self.aggregation_type = aggregation_type
        self.param = param
    def get_param_df(self):
        self.grp_df = (self.data.groupby(self.aggregation_var)[self.param]
                  .aggregate(self.aggregation_type).reset_index()[[self.aggregation_var,
                                                                   self.param
                                                                   ]]
                  )
        return self.grp_df
    def get_df_for_all_params(self):
        self.df_dict = {}
        if self.param:
            if isinstance(self.param, str):
                param = [self.param]
            param = self.param
        else:
            param = self.data.columns
        for var in param:
            if var != self.aggregation_var:
                grp_df = (self.data.groupby(self.aggregation_var)[var]
                          .aggregate(self.aggregation_type)
                          .reset_index()[[self.aggregation_var,var]]
                          )
                self.df_dict[var] = grp_df
        return self.df_dict
    
    def get_features_with_missing_values(self, data=None):
        try:
            if not data:
                data = self.data
        except ValueError:
            data = data
        self.empty_feature = []
        for col in data.columns:
            if data[col].isnull().sum() > 0:
                self.empty_feature.append(col)
        return self.empty_feature
                
    def replace_all_missing_values(self, empty_feature=None, df_dict=None,
                                   data=None
                                   ):
        try:
            if not data:
                data = self.data
        except ValueError:
            data = data
            
        if not empty_feature:
            if not hasattr(self, "empty_feature"):
                empty_feature = self.get_features_with_missing_values()
            else:
                empty_feature = self.empty_feature
        if not df_dict:
            if not hasattr(self, "df_dict"):
                df_dict = self.get_df_for_all_params()
            else:
                df_dict = self.df_dict
        for column in empty_feature:
            for index, row in data.iterrows():
                if pd.isnull(row[column]):
                    month = row.month
                    param_imput = df_dict[column]
                    nafill = param_imput[param_imput.month==month][column].values[0]
                    data.loc[index, column] = nafill
        return data

#%%
param_imp_obj = ParameterImputation(data=train_df, 
                                    aggregation_type="mean",
                                    aggregation_var="month", 
                                    param=selected_features
                                    )

#%%
median_value_per_mnth = param_imp_obj.get_df_for_all_params()

#%%
median_value_per_mnth.keys()        
# month aggregation have no NA but hour have NA for 14

#%%
empty_feature = []
for col in train_df.columns:
    if train_df[col].isnull().sum() > 0:
        empty_feature.append(col)
        
#%%
egdf = train_df[["month", empty_feature[0]]]#.isna()

#%%
egdf[egdf[empty_feature[0]].isna()]

#%%
mn = egdf.filter(items=[0], axis=0).month.values[0]


#%%
med_df = median_value_per_mnth[empty_feature[0]]

#%%
nafill = med_df[med_df.month==mn][empty_feature[0]].values[0]

#%%
egdf.filter(items=[0], axis=0)[empty_feature[0]].fillna(nafill)

#%%
egdf.iloc[0].sulphurdioxide_so2_column_number_density = nafill

# select col
# filter to na 
# for each row get the month -- mn
# go to impuation values and retrieved parameter df -- pmdf
# retrieve imputation value using month mn -- nafill
# assign nafill

#%%
imp_df = train_num_df.copy()
#imp_df[empty_feature]
#%%
param_imp_obj = ParameterImputation(data=train_num_df, 
                                    aggregation_type="mean",
                                    aggregation_var="month", 
                                    param=None,
                                    )

#%%
numeric_median_value_per_mnth = param_imp_obj.get_df_for_all_params()

#%%
numeric_empty_feature = []
for col in train_num_df.columns:
    if train_num_df[col].isnull().sum() > 0:
        numeric_empty_feature.append(col)

for column in numeric_empty_feature:
    for index, row in imp_df.iterrows():
        #if count == 5: break
        ##print(index, row)
        if pd.isnull(row[column]):
            month = row.month
            param_imput = numeric_median_value_per_mnth[column]
            nafill = param_imput[param_imput.month==month][column].values[0]
            imp_df.loc[index, column] = nafill  
            #count += 1

#%%

train_df[['month', 'carbonmonoxide_co_column_number_density']]

#%%
imp_df[['month', 'carbonmonoxide_co_column_number_density']]

#%%
numeric_median_value_per_mnth['carbonmonoxide_co_column_number_density']

#%%   ########## test median month value imputation on HistGradient
# take 10% of data for test
modelling_data, unseen_data = train_test_split(train_df, test_size=0.1, random_state=42, 
                                                stratify=train_df["month"]
                                                )

#%% # NA_imputation from do modelling data
modelling_numeric_df = modelling_data[train_num_df.columns]
param_imp_obj = ParameterImputation(data=modelling_numeric_df, 
                                    aggregation_type="median",
                                    aggregation_var="month", 
                                    param=None
                                    )

modelling_data_imputed = param_imp_obj.replace_all_missing_values()

#%%
median_month_imputation = param_imp_obj.df_dict

#%%
y_num_median_imputed = modelling_data_imputed["pm2_5"]

X_num_median_imputed = modelling_data_imputed.drop(columns="pm2_5")

#%%
randomsearch_num_median_imputed20cv = cross_validate(estimator=random_search.best_estimator_, 
                                                    X=X_num_median_imputed, 
                                                    y=y_num_median_imputed, cv=20, n_jobs=-1, 
                                                    scoring='neg_root_mean_squared_error',
                                                    verbose=4,
                                                    return_estimator=True,
                                                    return_train_score=True
                                                )

#%%
randomsearch_num_median_imputed20cv["test_score"].mean()

#%%
randomsearch_num_median_imputed20cv["train_score"].mean()

#%%
#randomsearch_num_median_imputed20cv["test_score"]

#%%  prediction with imputation
# test_param_imp_obj = ParameterImputation(data=unseen_data, 
#                                         aggregation_type="min",
#                                         aggregation_var="month", 
#                                         param=None
#                                         )
unseen_data_NAfeatures = param_imp_obj.get_features_with_missing_values(data=unseen_data)
test_data_median_imputed = param_imp_obj.replace_all_missing_values(df_dict=median_month_imputation,
                                                                    empty_feature=unseen_data_NAfeatures,
                                                                    data=unseen_data
                                                                    )

#%%
y_unseen = test_data_median_imputed["pm2_5"]
X_unseen = test_data_median_imputed[train_num_df.columns].drop(columns="pm2_5")

#%%
mean_squared_error(y_true=y_unseen, y_pred=
                    randomsearch_num_median_imputed20cv["estimator"][0].predict(X_unseen),
                    squared=False
                    )

#%%
def imputed_and_predict(imputation_df, test_data, params, model,
                        param_imp_obj, include_columns=["id", "pm2_5"]
                        ):
    NA_features = param_imp_obj.get_features_with_missing_values(data=test_data)
    # param_imp_obj = ParameterImputation(data=test_data[params], 
    #                                     aggregation_type=aggregation_type,
    #                                     aggregation_var="month", 
    #                                     param=None
    #                                     )
    test_data_imputed = param_imp_obj.replace_all_missing_values(df_dict=imputation_df,
                                                                 empty_feature=NA_features,
                                                                 data=test_data[params]
                                                                 )
    predictions = model.predict(test_data_imputed)
    test_data["pm2_5"] = predictions
    return test_data[include_columns]

#%%
prediction = imputed_and_predict(imputation_df=median_month_imputation,
                    test_data=test_df, params=num_cols, param_imp_obj=param_imp_obj
                    model=randomsearch_num_median_imputed20cv["estimator"][0]
                    )

#%% # monthly median imputed for all numberic features HistGradBost
subm7_path = get_data_path(folder_name="submissions", file_name="submission7.csv")   
prediction.to_csv(subm7_path, index=False)    
    
# %% fit of full training data  

param_imp_obj = ParameterImputation(data=train_num_df, 
                                    aggregation_type="median",
                                    aggregation_var="month", 
                                    param=None
                                    )
train_data_imputed = param_imp_obj.replace_all_missing_values()
median_month_imputation = param_imp_obj.df_dict
#%%
full_y_num_median_imputed = train_data_imputed["pm2_5"]

full_X_num_median_imputed = train_data_imputed.drop(columns="pm2_5")

randomsearch_num_median_imputed20cv = cross_validate(estimator=random_search.best_estimator_, 
                                                    X=full_X_num_median_imputed, 
                                                    y=full_y_num_median_imputed, cv=20, n_jobs=-1, 
                                                    scoring='neg_root_mean_squared_error',
                                                    verbose=4,
                                                    return_estimator=True,
                                                    return_train_score=True
                                                )

#%%
randomsearch_num_median_imputed20cv["test_score"].mean()

#%%
randomsearch_num_median_imputed20cv["train_score"].mean()

fulldata_prediction = imputed_and_predict(imputation_df=median_month_imputation,
                    test_data=test_df, params=num_cols, param_imp_obj=param_imp_obj,
                    model=randomsearch_num_median_imputed20cv["estimator"][0]
                    )

#%% # full data fit monthly median imputed for all numberic features HistGradBost
subm9_path = get_data_path(folder_name="submissions", file_name="submission9.csv")   
fulldata_prediction.to_csv(subm9_path, index=False)    

#%%  experiment with optuna
import optuna
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              BaggingRegressor, ExtraTreesRegressor,
                              VotingRegressor, AdaBoostRegressor,
                              )
from xgboost import XGBRFRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import (LinearRegression, LassoCV,
                                  RidgeCV,
                                  ridge_regression,
                                  )
import logging

logger = logging.getLogger()
logfile = get_data_path("logs_dir", "optuna_log.log")
logger.addHandler(logging.FileHandler(logfile, mode="w"))
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()
hyperparameter_space = {
                        'learning_rate': np.random.uniform(low=0.03, high=1, size=100),
                        'max_leaf_nodes': np.random.randint(low=20, high=300, size=250),
                        'min_samples_leaf': np.random.randint(low=25, high=300, size=250),
                        'l2_regularization': np.random.uniform(low=0.01, high=1, size=100),
                        'max_iter': np.random.randint(low=200, high=700, size=500)
            }
one = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler()
#%%
def objective(trial, X_features=X_num_median_imputed, 
              y_target=y_num_median_imputed,
              cv=20, verbose=3,
              scoring='neg_root_mean_squared_error'
              ):
    preprocess_pipeline =  make_column_transformer((scaler, X_num_median_imputed.columns),
                                                      n_jobs=-1)
    learning_rate = trial.suggest_float("learning_rate", 0.03, 1, log=True)
    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 1, 300, log=True)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 300, log=True)
    l2_regularization = trial.suggest_float("l2_regularization", 0.01, 1, log=True)
    max_iter = trial.suggest_int("max_iter", 200, 700, log=True)
    loss = trial.suggest_categorical("loss", ['linear', 'square', 'exponential'])
    n_estimators = trial.suggest_int("n_estimators", 50, 500, log=True)
    max_samples = trial.suggest_int("max_samples", 1, 500, log=True)
    max_features = trial.suggest_int("max_features", 1, 70)
    rfcriterion = trial.suggest_categorical("criterion", ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']) #Literal['squared_error', 'absolute_error', 'friedman_mse', 'poisson'] = "squared_error",
    max_depth = trial.suggest_int("max_depth", 3, 700, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 300, log=True)
    min_weight_fraction_leaf = trial.suggest_float("min_weight_fraction_leaf", 0.1, 0.5, log=True)
    
    min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0.1, 0.5, log=True)
    
    ccp_alpha= trial.suggest_float("ccp_alpha", 0.3, 10, log=True)
    regressor_name = trial.suggest_categorical("regressor", ["rfr", "gbr","etr"])
    
    histgb = HistGradientBoostingRegressor(random_state=42,
                                           learning_rate=learning_rate,
                                           max_leaf_nodes=max_leaf_nodes,
                                           min_samples_leaf=min_samples_leaf,
                                           l2_regularization=l2_regularization,
                                           max_iter=max_iter
                                           )
    rfr = RandomForestRegressor(random_state=42, verbose=verbose,
                                max_depth=max_depth, n_jobs=-1,
                                n_estimators=n_estimators,
                                criterion=rfcriterion,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                min_impurity_decrease=min_impurity_decrease,
                                ccp_alpha=ccp_alpha, max_features=max_features
                                )
    gbr = GradientBoostingRegressor(random_state=42, verbose=verbose,
                                    min_samples_leaf=min_samples_split,
                                    min_samples_split=min_samples_split,
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    max_features=max_features,
                                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                                    min_impurity_decrease=min_impurity_decrease,
                                    ccp_alpha=ccp_alpha,
                                    warm_start=True
                                    )
    etr = ExtraTreesRegressor(random_state=42, verbose=verbose,
                              n_estimators=n_estimators,
                              max_depth=max_depth,
                              min_impurity_decrease=min_impurity_decrease,
                              ccp_alpha=ccp_alpha,
                                warm_start=True, n_jobs=-1,
                              min_samples_leaf=min_samples_split,
                            min_samples_split=min_samples_split,
                                    max_features=max_features,
                            min_weight_fraction_leaf=min_weight_fraction_leaf,
                              )
    hist_study_bestparam = HistGradientBoostingRegressor(learning_rate=0.05404929713850879,
                                                    max_leaf_nodes=43,
                                                    min_samples_leaf=67,
                                                    l2_regularization=0.03278676565828471,
                                                    max_iter= 365,
                                                    scoring='neg_root_mean_squared_error',
                                                    verbose=1, random_state=42
                                                    )
    # bagg = BaggingRegressor(estimator=hist_study_bestparam, n_estimators=n_estimators,
    #                         n_jobs=-1, verbose=verbose, 
    #                         max_features=max_features,
    #                         max_samples=max_samples,
    #                         warm_start=True
    #                         )
    # adaboostr = AdaBoostRegressor(random_state=42,
    #                               loss=loss, n_estimators=n_estimators,
    #                               learning_rate=learning_rate,
    #                               estimator=hist_study_bestparam
    #                               )
    #xgbr = XGBRFRegressor()
    
    if regressor_name == "rfr":
        regressor_obj = rfr
    elif regressor_name == "gbr":
        regressor_obj = gbr
    elif regressor_name == "etr":
        regressor_obj = etr
    
    
    # model_pipeline = make_pipeline(preprocess_pipeline,
    #                                 regressor_obj, verbose=True
    #                                 )
    
     
    histgb_cv = cross_validate(estimator=regressor_obj,
                                X=X_features, 
                                y=y_target,
                                cv=cv, 
                                scoring=scoring,
                                return_estimator=True,
                                return_train_score=True,
                                verbose=verbose,
                                n_jobs=-1
                        )
    neg_rmse = histgb_cv["test_score"].mean()
    return neg_rmse

#%%
study = optuna.create_study(direction="maximize")
logger.info("Start optimization.")
study.optimize(objective, n_trials=200)

#%%
print(study.best_trial)  

#%%

study.best_trials

#%%
# trial 182 is based on normalized data of all numeric features
#Trial 182 finished with value: -21.237733501462362 and parameters: 
    
    {'learning_rate': 0.11580655894138978, 'max_leaf_nodes': 5, 'min_samples_leaf': 9, 'l2_regularization': 0.192520008113871, 'max_iter': 358, 'loss': 'exponential', 'n_estimators': 460, 'max_samples': 22, 'max_features': 64, 'criterion': 'friedman_mse', 'max_depth': 89, 'min_samples_split': 62, 'min_weight_fraction_leaf': 0.1038770913576251, 'min_impurity_decrease': 0.39235161038804933, 'ccp_alpha': 0.3019921927362563, 'regressor': 'gbr'}. Best is trial 182 with value: -21.237733501462362.
GradientBoostingRegressor = params={'learning_rate': 0.11580655894138978, 'max_leaf_nodes': 5, 'min_samples_leaf': 9, 'l2_regularization': 0.192520008113871, 'max_iter': 358, 'loss': 'exponential', 'n_estimators': 460, 'max_samples': 22, 'max_features': 64, 'criterion': 'friedman_mse', 'max_depth': 89, 'min_samples_split': 62, 'min_weight_fraction_leaf': 0.1038770913576251, 'min_impurity_decrease': 0.39235161038804933, 'ccp_alpha': 0.3019921927362563, 'regressor': 'gbr'}, 
 

# trial 172 is based on non-normalized data of all numeric features
# Trial 172 finished with value: -21.219905169989556 and parameters: {'learning_rate': 0.14393069927798605, 'max_leaf_nodes': 72, 'min_samples_leaf': 136, 'l2_regularization': 0.3626036521237126, 'max_iter': 322, 'loss': 'square', 'n_estimators': 389, 'max_samples': 73, 'max_features': 47, 'criterion': 'poisson', 'max_depth': 44, 'min_samples_split': 52, 'min_weight_fraction_leaf': 0.1001200211518383, 'min_impurity_decrease': 0.46077638702385226, 'ccp_alpha': 0.31737228475254087, 'regressor': 'gbr'}. Best is trial 172 with value: -21.219905169989556.

# user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'learning_rate': FloatDistribution(high=1.0, log=True, low=0.03, step=None), 'max_leaf_nodes': IntDistribution(high=300, log=True, low=1, step=1), 'min_samples_leaf': IntDistribution(high=300, log=True, low=1, step=1), 'l2_regularization': FloatDistribution(high=1.0, log=True, low=0.01, step=None), 'max_iter': IntDistribution(high=700, log=True, low=200, step=1), 'loss': CategoricalDistribution(choices=('linear', 'square', 'exponential')), 'n_estimators': IntDistribution(high=500, log=True, low=50, step=1), 'max_samples': IntDistribution(high=500, log=True, low=1, step=1), 'max_features': IntDistribution(high=70, log=False, low=1, step=1), 'criterion': CategoricalDistribution(choices=('squared_error', 'absolute_error', 'friedman_mse', 'poisson')), 'max_depth': IntDistribution(high=700, log=True, low=3, step=1), 'min_samples_split': IntDistribution(high=300, log=True, low=2, step=1), 'min_weight_fraction_leaf': FloatDistribution(high=0.5, log=True, low=0.1, step=None), 'min_impurity_decrease': FloatDistribution(high=0.5, log=True, low=0.1, step=None), 'ccp_alpha': FloatDistribution(high=10.0, log=True, low=0.3, step=None), 'regressor': CategoricalDistribution(choices=('rfr', 'gbr', 'etr'))}, trial_id=182, value=None)
#%%   use best study params 

hist_study_bestparam = HistGradientBoostingRegressor(learning_rate=0.05404929713850879,
                                                    max_leaf_nodes=43,
                                                    min_samples_leaf=67,
                                                    l2_regularization=0.03278676565828471,
                                                    max_iter= 365,
                                                    scoring='neg_root_mean_squared_error',
                                                    verbose=3, random_state=42
                                                    )

#%%
hist_study_obj = cross_validate(estimator=hist_study_bestparam,
                           X=X_num_median_imputed, 
                            y=y_num_median_imputed,
                            cv=20, 
                            scoring='neg_root_mean_squared_error',
                            return_estimator=True,
                            return_train_score=True,
                            verbose=3
                        )

#%%
hist_study_obj["test_score"].mean()

#%%
hist_study_obj["train_score"].mean()

#%%
unseen_data_NAfeatures = param_imp_obj.get_features_with_missing_values(data=unseen_data)
test_data_median_imputed = param_imp_obj.replace_all_missing_values(df_dict=median_month_imputation,
                                                                    empty_feature=unseen_data_NAfeatures,
                                                                    data=unseen_data
                                                                    )

y_unseen = test_data_median_imputed["pm2_5"]
X_unseen = test_data_median_imputed[train_num_df.columns].drop(columns="pm2_5")
mean_squared_error(y_true=y_unseen, y_pred=
                    hist_study_obj["estimator"][0].predict(X_unseen),
                    squared=False
                    )
studyhist_prediction = imputed_and_predict(imputation_df=median_month_imputation,
                    test_data=test_df, params=num_cols, param_imp_obj=param_imp_obj,
                    model=hist_study_obj["estimator"][0]
                    )

#%% # optuna optimized monthly median imputed for all numberic features HistGradBost 
# best prediction so far on LB 13.306174
subm9_path = get_data_path(folder_name="submissions", file_name="submission9.csv")
#%%   
studyhist_prediction.to_csv(subm9_path, index=False)

#%%  #### replace negative prediction with 0
## reduced error further 13.28368514
subm9_df = pd.read_csv(subm9_path)

#%%
subm9_df['pm2_5'] = subm9_df['pm2_5'].apply(lambda x: 0 if x < 0 else x)

#%%
subm11_path = get_data_path(folder_name="submissions", file_name="submission11_zep0na.csv")

subm9_df.to_csv(subm11_path, index=False)


#%% #########full data fit on optuna optimized model  fit of full training data  

param_imp_obj = ParameterImputation(data=train_num_df, 
                                    aggregation_type="median",
                                    aggregation_var="month", 
                                    param=None
                                    )
train_data_imputed = param_imp_obj.replace_all_missing_values()
#%%
median_month_imputation = param_imp_obj.df_dict
full_y_num_median_imputed = train_data_imputed["pm2_5"]

full_X_num_median_imputed = train_data_imputed.drop(columns="pm2_5")

histudy_num_median_imputed20cv = cross_validate(estimator=hist_study_obj["estimator"][0], 
                                                    X=full_X_num_median_imputed, 
                                                    y=full_y_num_median_imputed, 
                                                    cv=20, n_jobs=-1, 
                                                    scoring='neg_root_mean_squared_error',
                                                    verbose=4,
                                                    return_estimator=True,
                                                    return_train_score=True
                                                )

histudy_num_median_imputed20cv["test_score"].mean()

#%%
histudy_num_median_imputed20cv["train_score"].mean()

hist_fulldata_prediction = imputed_and_predict(imputation_df=median_month_imputation,
                    test_data=test_df, params=num_cols, param_imp_obj=param_imp_obj,
                    model=histudy_num_median_imputed20cv["estimator"][0]
                    )

#%% # full data fit on optuna optimized model  fit of full training data  
subm10_path = get_data_path(folder_name="submissions", file_name="submission10.csv")   
hist_fulldata_prediction.to_csv(subm10_path, index=False)    


#%%

min_param_imp_obj = ParameterImputation(data=train_num_df, 
                                    aggregation_type="min",
                                    aggregation_var="month", 
                                    param=None
                                    )

min_pm2_5_monthly = min_param_imp_obj.get_df_for_all_params()["pm2_5"]
#%%
train_data_imputed = param_imp_obj.replace_all_missing_values()


#%% median imputation with LGBMRegressor
from lightgbm import LGBMRegressor
lg_model = LGBMRegressor()

lg_model_num_median_imputed20cv = cross_validate(estimator=lg_model, 
                                                X=X_num_median_imputed, 
                                                y=y_num_median_imputed, cv=20, n_jobs=-1, 
                                                scoring='neg_root_mean_squared_error',
                                                verbose=4,
                                                return_estimator=True,
                                                return_train_score=True
                                                )

#%%
lg_model_num_median_imputed20cv["test_score"].mean()

#%%
mean_squared_error(y_true=y_unseen, y_pred=
                    lg_model_num_median_imputed20cv["estimator"][0].predict(X_unseen),
                    squared=False
                    )

#%%
lgmed_prediction = imputed_and_predict(imputation_df=median_month_imputation,
                    test_data=test_df, params=num_cols,
                    model=lg_model_num_median_imputed20cv["estimator"][0]
                    )

#%% # baseline estimator monthly median imputed for all numberic features
# 2 decimal place conversion
lgmed_prediction.pm2_5 = lgmed_prediction.pm2_5.round(2)
subm8_path = get_data_path(folder_name="submissions", file_name="submission8.csv")   
lgmed_prediction.to_csv(subm8_path, index=False) 









    
#%%   ####### experiment with tpot  ##############

from tpot import TPOTRegressor

#%%
log = "/home/lin/codebase/air_quality_pred/log"
model_dir = os.path.dirname(model_path)
tpot_obj = TPOTRegressor(verbosity=3, population_size=100,
                        cv=20, n_jobs=-1,max_time_mins=120,
                        random_state=42, warm_start=True,
                        memory=log, use_dask=False,
                        periodic_checkpoint_folder=model_dir
                        )

#%%  ## tpot fit on all numberic data

tpot_obj.fit(features=X_num, target=y_num)

#%%  ####### simple imputer and train with best model
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="most_frequent")

#%%
imputer.fit(X_train42rm)
training_features = imputer.transform(X_train42rm)

#%%
randomsearch_num20cv_medianimput = cross_validate(estimator=random_search.best_estimator_, 
                                                    X=training_features, 
                                                    y=y_train42rm, cv=20, n_jobs=-1, 
                                                    scoring='neg_root_mean_squared_error',
                                                    verbose=4,
                                                    return_estimator=True,
                                                    return_train_score=True
                                                )
#%%
randomsearch_num20cv_medianimput["test_score"].mean()

#%%
testing_features = imputer.transform(X_test42rm)

#%%
mean_squared_error(y_true=y_test42rm, y_pred=
                    randomsearch_num20cv_medianimput["estimator"][0].predict(testing_features),
                    squared=False
                    )

#%%
test_df_submostfreq_impt = test_df.copy()

#transformed_submostfreq_impt =  imputer.transform(test_df_submostfreq_impt)

#%%
subm_feat = imputer.transform(test_df[num_cols])

#%%
test_df_submostfreq_impt["pm2_5"] = randomsearch_num20cv_medianimput["estimator"][0].predict(subm_feat)

#%%
subm4_mostfreq_impt = os.path.join(subm_dir, "submission4.csv")
test_dfcp3[["id", "pm2_5"]].to_csv(subm4_mostfreq_impt, index=False)

#%%  #################
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
#tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
#features = tpot_data.drop('target', axis=1)
#training_features, testing_features, training_target, testing_target = \
#            train_test_split(features, tpot_data['target'], random_state=42)
#%%
imputer = SimpleImputer(strategy="median")
imputer.fit(X_train42rm)
training_features = imputer.transform(X_train42rm)
testing_features = imputer.transform(X_test42rm)

#%% Average CV score on the training set was: -362.63985600630355
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=RidgeCV()),
    XGBRegressor(learning_rate=0.1, max_depth=6, min_child_weight=3, 
                 n_estimators=100, n_jobs=1, objective="reg:squarederror",
                 subsample=1.0, verbosity=3
                 )
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, y_train42rm)

#%%
results = exported_pipeline.predict(testing_features)
#%%
mean_squared_error(y_true=y_test42rm, y_pred=results, squared=False)

#%%
xgb_cv20 = cross_validate(estimator=exported_pipeline, 
                X=training_features, 
                y=y_train42rm, cv=20, n_jobs=-1, 
                scoring='neg_root_mean_squared_error',
                verbose=4,
                return_estimator=True,
                return_train_score=True
                )

#%%
xgb_cv20["test_score"].mean()

xgb_cv20est = xgb_cv20["estimator"][0]
#%% refit model on all data
full_feature = imputer.fit_transform(X_num)
xgb_cv20_full = cross_validate(estimator=exported_pipeline, 
                                X=full_feature, 
                                y=y_num, cv=20, n_jobs=-1, 
                                scoring='neg_root_mean_squared_error',
                                verbose=4,
                                return_estimator=True,
                                return_train_score=True
                                )
#%%
xgb_cv20_full["test_score"].mean()

#%%
xgb_cv20_full["train_score"].mean()

#%%
xgb_cv20_full_est = xgb_cv20_full["estimator"][0]

#%%
subm5_path = get_data_path(folder_name="submissions", file_name="submission6.csv")
sub5_xgbcv20_median = imputer.transform(test_df[num_cols].copy())
sub5 = test_df.copy()

#%%
sub5["pm2_5"] = xgb_cv20est.predict(sub5_xgbcv20_median)

#%%
sub5[["id","pm2_5"]].to_csv(subm5_path, index=False)

#%%

tpot_obj.fitted_pipeline_.predict(X_test42rm)


#%%
tpot_obj.pareto_front_fitted_pipelines_

#%%
tpot_obj.evaluated_individuals_

#%%
train_df.dropna()


#%%
"""
selected features produced a lower accuracy prediction with HistGradient
"""
#%%
"""
## Imputing values
groupby month and hour and find whether they influence PM25
Investigate how this can be used to impute missing values

# create features like day of the week, weekends,
time of the day
how does each and combinations of this relate to PM25


"""

# %%
"""
remove feature names endining with angle,altitude
'nitrogendioxide_absorbing_aerosol_index' == 'uvaerosolindex_absorbing_aerosol_index'
'ozone_cloud_fraction', 'sulphurdioxide_cloud_fraction'

# remove cloud fraction from the various predictors and use only tha of
sentinel Cloud

# use paramters ending with index


#### new params

# use difference between top - bottom
# time of day -- night, daytime
# create time features from data feature

### handling missing values
# replace with mode/ average / mean of same hour or daytime
# replace with the mean / value of other correlated values
# replace missing values with corresponding ones in other indicators
eg cloud fraction when missing should be replaced with sulphuroixde cloud 
fraction etc

selected = ['sulphurdioxide_so2_column_number_density',
       'sulphurdioxide_so2_column_number_density_amf',
       'sulphurdioxide_so2_slant_column_number_density',
       'sulphurdioxide_so2_column_number_density_15km',
       'month',
       'carbonmonoxide_co_column_number_density',
       'carbonmonoxide_h2o_column_number_density',
       'nitrogendioxide_no2_column_number_density',
       'nitrogendioxide_tropospheric_no2_column_number_density',
       'nitrogendioxide_stratospheric_no2_column_number_density',
       'nitrogendioxide_no2_slant_column_number_density',
       'nitrogendioxide_tropopause_pressure',
       'formaldehyde_tropospheric_hcho_column_number_density',
       'formaldehyde_tropospheric_hcho_column_number_density_amf',
       'formaldehyde_hcho_slant_column_number_density',
       'ozone_o3_column_number_density',
       'ozone_o3_column_number_density_amf',
       'ozone_o3_slant_column_number_density',
       'ozone_o3_effective_temperature',
       'cloud_cloud_fraction', 
       'cloud_surface_albedo',
       'uvaerosolindex_absorbing_aerosol_index',
       
       ]
       
       
# When the AAI is positive, it indicates the presence of UV-absorbing aerosols like dust and smoke
Clouds yield near-zero residual values and strongly negative residual values can be indicative of the presence of non-absorbing aerosols including sulfate aerosols.



"""