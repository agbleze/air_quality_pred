#%%
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from utils.get_path import get_data_path
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
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
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
import optuna
from optuna.integration import WeightsAndBiasesCallback
import wandb
import json
#%%
def get_missing_data_percent(data: pd.DataFrame, variable: str):
    total_missing = data[variable].isnull().sum()
    total_data = data.shape[0]
    percent_missing = (total_missing / total_data) * 100
    print(f'Percentage of data missing in {variable}: {round(percent_missing, 2)}%')
    return percent_missing


    
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


train_path = get_data_path(folder_name="zindi_pm25_pred", file_name="Train.csv")
test_path = get_data_path(folder_name="zindi_pm25_pred", file_name="Test.csv")
sample_path = get_data_path(folder_name="zindi_pm25_pred", file_name="SampleSubmission.csv")
# %%
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_df = pd.read_csv(sample_path)

# %%
mask = train_df.isnull().any()
no_missing_cl = train_df.columns[~mask]
no_missing_data_features = no_missing_cl[1:5]

#%%
selected_features = ['sulphurdioxide_so2_column_number_density',  ## not selected due to covariance
       'sulphurdioxide_so2_column_number_density_amf',
       'sulphurdioxide_so2_slant_column_number_density',  ## not selected due to covariance
       'sulphurdioxide_so2_column_number_density_15km',
       'month',
       'carbonmonoxide_co_column_number_density',
       'carbonmonoxide_h2o_column_number_density',
       'nitrogendioxide_no2_column_number_density',  ## npt selected due to covariance
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
reduced_features = ['sulphurdioxide_so2_column_number_density_15km',
                    'nitrogendioxide_tropospheric_no2_column_number_density',
                    'ozone_o3_slant_column_number_density',
                    'formaldehyde_hcho_slant_column_number_density',
                    'sulphurdioxide_so2_column_number_density_amf',
                    'carbonmonoxide_co_column_number_density', 
                    'uvaerosolindex_absorbing_aerosol_index',
                    'cloud_cloud_fraction', 
                    'cloud_surface_albedo',
                    'pm2_5'   
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
                        'pm2_5'
                        ]
#%%
train_df[reduced_features].corr()
# %%
train_df[['cloud_cloud_fraction', 
       'cloud_surface_albedo']].corr()

# %%
train_df[feat_scatted_to_pm25].corr()
# %%
all_selected_feat = ['id', 'site_id', 'site_latitude', 'site_longitude', 
        'city', 'country',
        'date', 'hour', 'month', 
        'sulphurdioxide_so2_column_number_density_15km',
                    'nitrogendioxide_tropospheric_no2_column_number_density',
                    'ozone_o3_slant_column_number_density',
                    'formaldehyde_hcho_slant_column_number_density',
                    'sulphurdioxide_so2_column_number_density_amf',
                    'carbonmonoxide_co_column_number_density', 
                    'uvaerosolindex_absorbing_aerosol_index',
                    'cloud_cloud_fraction', 
                    'cloud_surface_albedo',
        'pm2_5'
]
# %%
all_selected_train_df = train_df[all_selected_feat]
# %%
all_selected_train_df.date = pd.to_datetime(all_selected_train_df.date)

# %%
all_selected_train_df["day_name"] = all_selected_train_df.date.dt.day_name()

# %%
all_selected_train_df["is_leap_year"] = all_selected_train_df.date.dt.is_leap_year
# %%
all_selected_train_df["day_of_year"] = all_selected_train_df.date.dt.day_of_year

#%%
all_selected_train_df["quater"] = all_selected_train_df.date.dt.quarter

#%%
all_selected_train_df.date.dt.year.nunique()


# %%
all_selected_train_df.groupby(["month","day_name"])["pm2_5"].aggregate("mean").reset_index().plot()
# %%
meanpm25_perday =  all_selected_train_df.groupby(["day_name"])["pm2_5"].aggregate("mean").reset_index()
# %%
meanpm25_perday
# %%
def barplot(data_to_plot: pd.DataFrame, 
            variable_to_plot: str, 
            title: str,
            ylabel: str, y_colname: str = None
            ):
    if y_colname is None:
        bar_graph = (ggplot(data=data_to_plot, mapping=aes(x=variable_to_plot, 
                                                fill=variable_to_plot
                                            )
                                )
                                + geom_bar()  
                                + ggtitle(title) + xlab(variable_to_plot)
                                + ylab(ylabel)
                        )

        return print(bar_graph)
    else:
        bar_graph = (ggplot(data=data_to_plot, 
                            mapping=aes(x=variable_to_plot, 
                                        y=y_colname,
                                        fill=variable_to_plot
                                        )
                                )
                                + geom_col()
                                + ggtitle(title) + xlab(variable_to_plot)
                                + ylab(ylabel)
                        )

        return print(bar_graph)
    
#%%
def boxplot(data_to_plot: pd.DataFrame, x_colname: str, 
            y_colname: str,
            title: str = None
            ):
    if title is None:
        title = f'Distribution of {y_colname} among various {x_colname}'
        
    box_graph = (ggplot(data=data_to_plot, 
                        mapping=aes(x=x_colname, y=y_colname)
                        )
                    + geom_boxplot()
                    + coord_flip()
                    + ggtitle(title)
                )
    # the returned ggplot is printed to draw the graph which is not 
    # the case by default  when not printed
    return print(box_graph)
# %%
barplot(data_to_plot=meanpm25_perday, variable_to_plot="day_name",
        y_colname="pm2_5", title="Mean pm2_5 per day",
        ylabel="Mean pm2_5"
        )
# %%
all_selected_train_df.groupby("quater")["pm2_5"].aggregate("mean").reset_index()
# %%
meanpm25_perquater = all_selected_train_df.groupby("quater")["pm2_5"].aggregate("mean").reset_index()
# %%
barplot(data_to_plot=meanpm25_perquater, variable_to_plot="quater",
        y_colname="pm2_5", title="Mean pm2_5 per quarter",
        ylabel="Mean pm2_5"
        )
# %%
boxplot(data_to_plot=meanpm25_perquater, x_colname='quater', y_colname='pm2_5')
# %%

all_selected_train_df["quater_group"] = (np.where((all_selected_train_df['quater'].isin([1,4])),"first_fourth",
                                                 "second_third"
                                                 )
                                        )

#%%
def create_quater_group(data):
    data["quater_group"] =(np.where((data["quater"].isin([1,4])), "first_fourth","second_third",
                                    )
                           )
    return data
# %%
meanpm25_per_quatergrp = all_selected_train_df.groupby("quater_group")["pm2_5"].aggregate("mean").reset_index()
# %%
barplot(data_to_plot=meanpm25_per_quatergrp, variable_to_plot="quater_group",
        y_colname="pm2_5", title="Mean pm2_5 per quarter group",
        ylabel="Mean pm2_5"
        )
# %%
meanpm25_per_month = all_selected_train_df.groupby("month")["pm2_5"].aggregate("mean").reset_index()
# %%
barplot(data_to_plot=meanpm25_per_month, variable_to_plot="month",
        y_colname="pm2_5", title="Mean pm2_5 per month",
        ylabel="Mean pm2_5"
        )

# %%

#def create_month_emission_rate(data):
    
def create_month_emission_rate(data):
    data["month_pm25_rate"] =(np.where((data["month"].isin([1,2,12])), "high", 
                                       "low"
                                    )
                           )
    data["month_pm25_rate"] =(np.where((data["month"].isin([3,4,5])), "low", 
                                       data["month_pm25_rate"]
                                    )
                           )
    data["month_pm25_rate"] =(np.where((data["month"].isin([6,7,8,9,10,11])), "medium", 
                                       data["month_pm25_rate"]
                                    )
                           )
    return data   
# %%
all_selected_train_df = create_month_emission_rate(all_selected_train_df)
# %%
meanpm25_month_pm25_rate = all_selected_train_df.groupby("month_pm25_rate")["pm2_5"].aggregate("mean").reset_index()

# %%
barplot(data_to_plot=meanpm25_month_pm25_rate, variable_to_plot="month_pm25_rate", title="mean pm25",
        ylabel="mean pm2_5", y_colname="pm2_5")
# %%
## implement function for percentage of missing data per variable
for variable_name in all_selected_train_df.columns:
    get_missing_data_percent(data=all_selected_train_df, variable=variable_name)

#%%
# 0 is giving where missing value is true and 1 where false
create_parameter_is_avialable = lambda x: 0 if x is True else 1

def create_missing_value_features(data):
    data= data.copy()
    for var in data.columns:
        percent_miss = get_missing_data_percent(data=data, variable=var)
        if percent_miss > 0:
            data[f"ismiss_{var}"] = data[var].isnull().apply(create_parameter_is_avialable)
    
    return data

#%%

all_selected_train_df = create_missing_value_features(data=all_selected_train_df)


#%%
outlier_imputer = OutlierImputer(data=all_selected_train_df, colname="pm2_5")

#%%


outlier_values = outlier_imputer.get_outlier_samples.pm2_5.values

#%%  ########### create feature identifying outliers
outlier_imputer.upper_limit


#%%


all_selected_train_df["is_outlier"] = (np.where((all_selected_train_df["pm2_5"] > outlier_imputer.upper_limit), 1,
                                             np.where(all_selected_train_df["pm2_5"] < outlier_imputer.lower_limit, 1,
                                                      0)
                                    )
                           )

#%%
all_selected_train_df[all_selected_train_df["is_outlier"]==1][["pm2_5","is_outlier"]]

#%%
isoutlier_df = all_selected_train_df[all_selected_train_df["is_outlier"]==1]
all_selected_train_df.groupby("month")["is_outlier"].value_counts().reset_index()
#%%
all_selected_train_df.groupby(["month","day_name"])["is_outlier"].value_counts().reset_index()
#%%
isoutlier_df = all_selected_train_df[all_selected_train_df["is_outlier"]==1]
isoutlier_df.groupby("month")["is_outlier"].value_counts().reset_index()

#%%
mnthday_outlier = isoutlier_df.groupby(["month", "day_name"])["is_outlier"].value_counts().reset_index()


#%%
(ggplot(mnthday_outlier, aes(x='month', y='count', fill='day_name'))
 + geom_col(stat='identity', position='dodge')) + theme_dark() + ggtitle('outlier counts')

#%%
isoutlier_df.groupby("day_name")["is_outlier"].value_counts().reset_index()

#%%
# day of month features, month of year
###  ratio of outliers to non-outliers
### probability of being an outlier

#%% 

mth_outlier_ct = all_selected_train_df.groupby("month")["is_outlier"].value_counts().reset_index()

#%%
def calculate_monthly_outlier_nonoutlier_ratio(mth_outlier_ct):
    month = []
    month_out_nonout_ratio = []
    for mn in mth_outlier_ct.month.unique():
        mn_outlier_df = mth_outlier_ct[mth_outlier_ct.month==mn]
        outlier_cnt = mn_outlier_df[mn_outlier_df.is_outlier==1]["count"].values[0]
        nonoutlier_cnt = mn_outlier_df[mn_outlier_df.is_outlier==0]["count"].values[0]
        out_nonout_ratio = outlier_cnt/nonoutlier_cnt
        month.append(mn)
        month_out_nonout_ratio.append(out_nonout_ratio)

    dict_mnth_out_nonout_ratio = {"month": month, "out_nonout_ratio": month_out_nonout_ratio}
    month_out_nonout_ratio_df = pd.DataFrame(dict_mnth_out_nonout_ratio)
    return month_out_nonout_ratio_df

#%%

month_out_nonout_ratio_df = calculate_monthly_outlier_nonoutlier_ratio(mth_outlier_ct=mth_outlier_ct)
# this correlates with mean monthly_pm25 

#%%  ratio of outlier to nonoutlier per day of month
dayofmnth_outliercnt = all_selected_train_df.groupby(["month","day_name"])["is_outlier"].value_counts().reset_index()
mnths = all_selected_train_df.month.unique()
day_names = all_selected_train_df.day_name.unique()

#%%
def calculate_day_per_month_outier_non_outlier_ratio(dayofmnth_outliercnt):
    mnths = dayofmnth_outliercnt.month.unique()
    day_names = dayofmnth_outliercnt.day_name.unique()
    month_day = []
    month_day_out_nonout_ratio = []
    for mn in mnths:
        mn_outlier_df = dayofmnth_outliercnt[dayofmnth_outliercnt.month==mn]
        for day in day_names:
            if day not in day_names:
                out_nonout_ratio = 0
                mn_day = f"{str(mn)}_{day}"
                month_day.append(mn_day)
                month_day_out_nonout_ratio.append(out_nonout_ratio)
            else:
                day_mn_outlier_df = mn_outlier_df[mn_outlier_df.day_name==day]
                #print(mn, day)
                if 1 not in day_mn_outlier_df.is_outlier.values:
                    out_nonout_ratio = 0
                else:
                    outlier_cnt = day_mn_outlier_df[day_mn_outlier_df.is_outlier==1]["count"].values[0]
                    nonoutlier_cnt = day_mn_outlier_df[day_mn_outlier_df.is_outlier==0]["count"].values[0]
                    out_nonout_ratio = outlier_cnt/nonoutlier_cnt
                mn_day = f"{str(mn)}_{day}"
                month_day.append(mn_day)
                month_day_out_nonout_ratio.append(out_nonout_ratio)
    dict_month_day_out_nonout_ratio = {"month_day": month_day, "out_nonout_ratio": month_day_out_nonout_ratio}
    day_month_out_nonout_ratio_df = pd.DataFrame(dict_month_day_out_nonout_ratio)
    return day_month_out_nonout_ratio_df

#%%
day_month_out_nonout_ratio_df= calculate_day_per_month_outier_non_outlier_ratio(dayofmnth_outliercnt)

#%% probability that day of month has outliers
def calculate_proba_day_per_month_isoutier(dayofmnth_outliercnt):
    mnths = dayofmnth_outliercnt.month.unique()
    day_names = dayofmnth_outliercnt.day_name.unique()
    month_day = []
    month_day_proba = []
    for mn in mnths:
        mn_outlier_df = dayofmnth_outliercnt[dayofmnth_outliercnt.month==mn]
        for day in day_names:
            if day not in day_names:
                proba = 0
                mn_day = f"{str(mn)}_{day}"
                month_day.append(mn_day)
                month_day_proba.append(proba)
            else:
                day_mn_outlier_df = mn_outlier_df[mn_outlier_df.day_name==day]
                #print(mn, day)
                if 1 not in day_mn_outlier_df.is_outlier.values:
                    proba = 0
                else:
                    outlier_cnt = day_mn_outlier_df[day_mn_outlier_df.is_outlier==1]["count"].values[0]
                    total_daycnt = day_mn_outlier_df["count"].sum()
                    proba = outlier_cnt/total_daycnt
                mn_day = f"{str(mn)}_{day}"
                month_day.append(mn_day)
                month_day_proba.append(proba)
    dict_month_day_proba = {"month_day": month_day, "has_outlier_proba": month_day_proba}
    day_month_proba_df = pd.DataFrame(dict_month_day_proba)
    return day_month_proba_df

#%%
day_month_proba_df = calculate_proba_day_per_month_isoutier(dayofmnth_outliercnt=dayofmnth_outliercnt)


#%% probality that hwne you pick a firday in jan it is an outlier
create_month_day_parameter = lambda x: f"{str(x['month'])}_{x['day_name']}"

all_selected_train_df["month_day"] = all_selected_train_df.apply(create_month_day_parameter, 
                                                                 axis=1
                                                                 )

#%%
all_selected_train_df.groupby("month_day")["is_outlier"].value_counts().reset_index()

#%%
def calculate_global_proba_day_has_isoutier(data):
    mnths = data.month.unique()
    day_names = data.day_name.unique()
    grp_data = data.groupby("month_day")["is_outlier"].value_counts().reset_index()
    total = grp_data["count"].sum()
    month_day = []
    month_day_proba = []
    for mn in mnths:
        for day in day_names:
            mn_day = f"{str(mn)}_{day}"
            if mn_day not in data.month_day.unique():
                proba = 0
                month_day.append(mn_day)
                month_day_proba.append(proba)
                
            else:
                day_mn_outlier_df = grp_data[grp_data.month_day==mn_day]
                if 1 not in day_mn_outlier_df.is_outlier.values:
                    proba = 0
                else:
                    outlier_cnt = day_mn_outlier_df[day_mn_outlier_df.is_outlier==1]["count"].values[0]
                    proba = outlier_cnt/total
                month_day.append(mn_day)
                month_day_proba.append(proba)
    dict_month_day_proba = {"month_day": month_day, "global_has_outlier_proba": month_day_proba}
    day_month_proba_df = pd.DataFrame(dict_month_day_proba)
    return day_month_proba_df
#%%
global_proba_day_has_isoutier_df = calculate_global_proba_day_has_isoutier(data=all_selected_train_df)

#%%
global_proba_day_has_isoutier_df[global_proba_day_has_isoutier_df["month_day"]=="10.0_Monday"]


#%%  access holiday info
import calendarific
import json
import requests
cal_api = "uz4FqDqeZVeR8wUr85IlJiHiZ62DsyWc"
#%%
calapi = calendarific.v2(cal_api)
url = 'https://calendarific.com/api/v2/holidays?'

#%%
calapi.api_key
#%%
cities = ["Lagos", "Accra", "Nairobi", "Yaounde", "Bujumbura", "Kisumu", "Kampala", "Gulu"]
countries = ["ng"]
#%%
parameters = {"country": "NG",
              "year": 2023,
              "api_key": cal_api
              }

#%%
dict_country_holiday = {}
#data_response = []
for count in ["ng", "gh", "ke", "cm", "bi", "ug" ]:
    data_response = []
    for year in range(2014, 2025):
        parameters = {"country": count,
                    "year": year,
                    "api_key": cal_api
                    }
        response = requests.get(url, params=parameters)
        data = json.loads(response.text)
        data_response.append(data)
    dict_country_holiday[count] = data_response

#%%
holiname_lst = []
holidesc_lst = []
holicountry_lst = []
holidate_lst = []
holitype_lst = []
holiloc_lst = []

for countcode in dict_country_holiday:
    country_data = dict_country_holiday[countcode]
    for dt in country_data:
        dt_response = dt["response"]
        response_holidays = dt_response["holidays"]
        for holiday in response_holidays:
            holiday_name = holiday["name"]
            holiday_desc = holiday["description"]
            holiday_country = holiday["country"]["name"]
            holiday_date = holiday["date"]["iso"]
            holiday_type = holiday["type"]
            holiday_loc = holiday["locations"]
            
            holiname_lst.append(holiday_name)
            holidesc_lst.append(holiday_desc)
            holicountry_lst.append(holiday_country)
            holidate_lst.append(holiday_date)
            holitype_lst.append(holiday_type)
            holiloc_lst.append(holiday_loc)
            
#%%

holiday_dict = {"holiday_name": holiname_lst, "holiday_desc": holidesc_lst,
                "holiday_country": holicountry_lst, "holiday_date": holidate_lst,
                "holiday_type": holitype_lst, "holiday_loc": holiloc_lst
                }

holiday_df = pd.DataFrame(holiday_dict)

#%%
holiday_df = pd.read_csv("/home/lin/air_quality_pred/holiday.csv")
#%%
unlist_holidaytype_values = lambda x: x["holiday_type"][0]


holiday_df["holiday_type"] = holiday_df.apply(unlist_holidaytype_values, axis=1)

#%%
convert_holidaydate_to_dateonly = lambda x: str(x).split(" ")[0].split(":")[0].split("T")[0]
holiday_df["holiday_date"] = holiday_df["holiday_date"].apply(convert_holidaydate_to_dateonly)

#%%  
holiday_df["holiday_date"]= pd.to_datetime(holiday_df["holiday_date"])

#%%
holiday_df["month"] = holiday_df.holiday_date.dt.month
holiday_df["day_name"] = holiday_df.holiday_date.dt.day_name()

#%%
holiday_df["month_day"] = holiday_df.apply(create_month_day_parameter, axis=1)


#%%
all_selected_train_df["date"] = all_selected_train_df["date"].apply(convert_holidaydate_to_dateonly)
all_selected_train_df["date"]= pd.to_datetime(all_selected_train_df["date"])
#%%
count = 0
for val in all_selected_train_df.date.values:
    if val not in holiday_df.holiday_date.values:
        print(val)

#%%  todo 
# 1. add feature column for out_nonout ratio and proba
# add holiday 
# train model

#%%
global_proba_day_has_isoutier_df

#%%
all_selected_train_df = all_selected_train_df.merge(right=global_proba_day_has_isoutier_df, 
                            left_on="month_day", right_on="month_day",
                            )#[["month_day","global_has_outlier_proba", "month", "day_name"]]

#%%
all_selected_train_df = all_selected_train_df.merge(right=day_month_out_nonout_ratio_df,
                                                    left_on="month_day",
                                                    right_on="month_day"
                                                    )

#%%

all_selected_train_df = all_selected_train_df.merge(right=day_month_proba_df,
                                                    left_on="month_day",
                                                    right_on="month_day"
                                                    )

#%%
# all_selected_train_df["is_holiday"] = 0
# for country in holiday_df.holiday_country.values:
#     country_holidays = holiday_df[holiday_df.holiday_country==country]
#     for mnday in country_holidays.month_day.values:
#         #if country in all_selected_train_df["country"].unique()
#         all_selected_train_df["is_holiday"] = np.where((all(all_selected_train_df["country"]==country) and 
#                                                          all(all_selected_train_df["month_day"]==mnday)
#                                                          ),
#                                                        1, all_selected_train_df["is_holiday"]
#                                                         )
#%%
#cond1 = all_selected_train_df["country"]=="Nigeria"
#cond2 = all_selected_train_df["date"]=="2024-01-01"


#%%
for countr in holiday_df.holiday_country.unique():
    if countr in all_selected_train_df.country.unique():
        country_data = all_selected_train_df[all_selected_train_df["country"]==countr]
        for dateobj in holiday_df[holiday_df.holiday_country==countr].holiday_date.unique():
            if dateobj in country_data.date.unique():
                cond1 = all_selected_train_df["country"]==countr
                cond2 = all_selected_train_df["date"]==dateobj
             
                all_selected_train_df.loc[cond1 & cond2, "is_holiday"] = 1

#%%
def impute_holiday(holiday_df, data):
    for countr in holiday_df.holiday_country.unique():
        if countr in data.country.unique():
            country_data = data[data["country"]==countr]
            for dateobj in holiday_df[holiday_df.holiday_country==countr].holiday_date.unique():
                if dateobj in country_data.date.unique():
                    cond1 = data["country"]==countr
                    cond2 = data["date"]==dateobj
                
                    data.loc[cond1 & cond2, "is_holiday"] = 1
    data["is_holiday"].fillna(value=0, inplace=True)
    return data
        
#%%
all_data_with_holiday_df = impute_holiday(holiday_df=holiday_df, data=all_selected_train_df)

#all_selected_train_df.drop(columns="is_holiday", inplace=True)

#%%


#%%

all_data_with_holiday_df[(all_data_with_holiday_df.is_holiday == 1) & 
                      (all_data_with_holiday_df.country=="Kenya") &
                      (all_data_with_holiday_df.date.dt.year == 2023)&
                      (all_data_with_holiday_df.date.dt.month==12)][["date", "month_day", "country", "is_holiday"]] 

#%%
holiday_df[(holiday_df["holiday_date"].dt.year == 2023) &
           (holiday_df.holiday_date.dt.month==12) &
           (holiday_df.holiday_country=="Kenya")
           ][["holiday_country", "holiday_date", "month_day"]]

#%% ######## add ratio and proba to data and develop first model

#%%
create_is_outlier = lambda x: 0 if x is True else 1

#%%  ##### imputation  #############

# %%
columns = all_data_with_holiday_df.columns.to_list()

columns.remove("country")
# %%
columns
# %%
for col in ['id','site_id','site_latitude','site_longitude', 
            'city','date', 'month_day', "month_pm25_rate", "quater_group","day_name"]:
    columns.remove(col)

# %%
columns
# %%
param_imp_obj = ParameterImputation(data=all_data_with_holiday_df, 
                                    aggregation_type="mean",
                                    aggregation_var="month", 
                                    param=columns
                                    )

median_value_per_mnth = param_imp_obj.get_df_for_all_params()
# %%
modelling_data_imputed = param_imp_obj.replace_all_missing_values()
# %%
median_month_imputation = param_imp_obj.df_dict

#%%
y_selected_feng_median_imputed = modelling_data_imputed["pm2_5"]

X_selected_feng_median_imputed = modelling_data_imputed.drop(columns="pm2_5")


#%%  split data before impuatation
# 10% testing, 90% for 20 CV
# stratify for outliers, isholiday and month
training_df, testing_df = train_test_split(all_data_with_holiday_df, test_size=0.1, random_state=42,
                                           stratify=all_data_with_holiday_df[["is_outlier", 
                                                                               "is_holiday",
                                                                               "month"
                                                                               ]]
                                           )

train_param_imp_obj = ParameterImputation(data=training_df, 
                                        aggregation_type="median",
                                        aggregation_var="month", 
                                        param=columns
                                        )

#%%
train_median_imputed_df = train_param_imp_obj.replace_all_missing_values()
train_median_imputation = train_param_imp_obj

#%%
categorical_var = ['day_name', 'month_pm25_rate']

one = OneHotEncoder(sparse_output=False)

precprocess = make_column_transformer((one, categorical_var))
#%% columns to drop
col_todrop = ['id', 'site_id', 'site_latitude','site_longitude', 
              'city', 'country', 'date', 'hour', 
              'month_day', 'is_leap_year', "pm2_5", 
              "sulphurdioxide_so2_column_number_density_amf", "is_outlier",
              'quater_group'
              ]
scoring='neg_root_mean_squared_error'
X_features = train_median_imputed_df.drop(columns=col_todrop)
y_target = train_median_imputed_df["pm2_5"]

#%%
hist_study_bestparam = HistGradientBoostingRegressor(learning_rate=0.05404929713850879,
                                                    max_leaf_nodes=43,
                                                    min_samples_leaf=67,
                                                    l2_regularization=0.03278676565828471,
                                                    max_iter= 365,
                                                    scoring='neg_root_mean_squared_error',
                                                    verbose=1, random_state=42
                                                    )

model_pipeline = make_pipeline(precprocess,hist_study_bestparam, verbose=True)
cv = 20
#%%
histgb_cv = cross_validate(estimator=model_pipeline,
                            X=X_features, 
                            y=y_target,
                            cv=cv, 
                            scoring=scoring,
                            return_estimator=True,
                            return_train_score=True,
                            verbose=3,
                            n_jobs=-1
                        )

#%%

histgb_cv["test_score"].mean()

#%%
histgb_cv["train_score"].mean()
# %%
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
#one = OneHotEncoder(handle_unknown='ignore')
one = OneHotEncoder(sparse_output=False)
scaler = StandardScaler()
#%%
wandbc_kwargs = {"project": "airquality_pm2_5_prediction"}
wandbc = WeightsAndBiasesCallback(metric_name="rmse", 
                                  wandb_kwargs=wandbc_kwargs, 
                                  as_multirun=True
                                  )
@wandbc.track_in_wandb()
def objective(trial, X_features=X_features, 
              y_target=y_target,
              cv=20, verbose=3,
              scoring='neg_root_mean_squared_error'
              ):
    precprocess = make_column_transformer((one, categorical_var))
    #preprocess_pipeline =  make_column_transformer((precprocess, ),
    #                                                  n_jobs=-1)
    learning_rate = trial.suggest_float("learning_rate", 0.03, 1, log=True)
    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 2, 1300, log=True)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 1300, log=True)
    l2_regularization = trial.suggest_float("l2_regularization", 0.01, 1, log=True)
    max_iter = trial.suggest_int("max_iter", 200, 1700, log=True)
    loss = trial.suggest_categorical("loss", ['linear', 'square', 'exponential'])
    n_estimators = trial.suggest_int("n_estimators", 50, 1500, log=True)
    max_samples = trial.suggest_int("max_samples", 1, 1500, log=True)
    max_features = trial.suggest_int("max_features", 1, 70)
    rfcriterion = trial.suggest_categorical("criterion", ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']) #Literal['squared_error', 'absolute_error', 'friedman_mse', 'poisson'] = "squared_error",
    max_depth = trial.suggest_int("max_depth", 3, 1700, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 1300, log=True)
    min_weight_fraction_leaf = trial.suggest_float("min_weight_fraction_leaf", 0.1, 0.5, log=True)
    
    min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0.1, 0.5, log=True)
    
    ccp_alpha= trial.suggest_float("ccp_alpha", 0.3, 10, log=True)
    regressor_name = trial.suggest_categorical("regressor", ["hgbr"]) #, "gbr","etr", "hgbr"])
    
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
    elif regressor_name == "hgbr":
        regressor_obj = histgb
    
    
    model_pipeline = make_pipeline(precprocess,
                                    regressor_obj, verbose=True
                                    )
    
     
    histgb_cv = cross_validate(estimator=model_pipeline,
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

# %%
study = optuna.create_study(direction="maximize", study_name="pm2_5_prediction_hgbr")
logger.info("Start optimization.")
study.optimize(objective, n_trials=300, show_progress_bar=True, callbacks=[wandbc])
# %%
study.best_trials

#%%  train model with best parameters
cross_validate(estimator=hist_study_obj["estimator"][0], 
                                                    X=full_X_num_median_imputed, 
                                                    y=full_y_num_median_imputed, 
                                                    cv=20, n_jobs=-1, 
                                                    scoring='neg_root_mean_squared_error',
                                                    verbose=4,
                                                    return_estimator=True,
                                                    return_train_score=True
                                                )





# %%
unseen_data_NAfeatures = train_param_imp_obj.get_features_with_missing_values(data=testing_df)
test_data_median_imputed = train_param_imp_obj.replace_all_missing_values(df_dict=train_median_imputation,
                                                                    empty_feature=unseen_data_NAfeatures,
                                                                    data=testing_df
                                                                    )

#%%
pred_cols = X_features.columns
#%%
X_unseen_features = create_missing_value_features(data=test_data_median_imputed[pred_cols])

month_out_nonout_ratio_df = calculate_monthly_outlier_nonoutlier_ratio(mth_outlier_ct=mth_outlier_ct)

day_month_out_nonout_ratio_df= calculate_day_per_month_outier_non_outlier_ratio(dayofmnth_outliercnt)

day_month_proba_df = calculate_proba_day_per_month_isoutier(dayofmnth_outliercnt=dayofmnth_outliercnt)

#
global_proba_day_has_isoutier_df = calculate_global_proba_day_has_isoutier(data=all_selected_train_df)



all_data_with_holiday_df = impute_holiday(holiday_df=holiday_df, data=all_selected_train_df)


#%%
test_data_median_imputed["month_day"] = test_data_median_imputed.apply(create_month_day_parameter, axis=1)

## merge global_proba_day_has_isoutier_df, day_month_out_nonout_ratio_df, day_month_proba_df on data

test_data_median_imputed = test_data_median_imputed.merge(right=global_proba_day_has_isoutier_df, 
                                                        left_on="month_day", right_on="month_day",
                                                        )

test_data_median_imputed = test_data_median_imputed.merge(right=day_month_out_nonout_ratio_df,
                                                        left_on="month_day",
                                                        right_on="month_day"
                                                        )
test_data_median_imputed = test_data_median_imputed.merge(right=day_month_proba_df,
                                                            left_on="month_day",
                                                            right_on="month_day"
                                                            )

#%%
unseen_features = test_data_median_imputed.drop(columns=col_todrop, axis=1)
unseen_target = test_data_median_imputed["pm2_5"]
#%% fitting best param model
best_histgb = HistGradientBoostingRegressor(random_state=42,
                                           learning_rate=0.030717210588095418,
                                           max_leaf_nodes=4,
                                           min_samples_leaf=5,
                                           l2_regularization=0.08739049568488838,
                                           max_iter=206
                                           )

model_pipeline = make_pipeline(precprocess,
                                best_histgb, verbose=True
                                )
    
     
histgb_cv = cross_validate(estimator=model_pipeline,
                        X=X_features, 
                        y=y_target,
                        cv=cv, 
                        scoring=scoring,
                        return_estimator=True,
                        return_train_score=True,
                        verbose=5,
                        n_jobs=-1
                )

#%%
histgb_cv["test_score"].mean()

#%%
histgb_cv["train_score"].mean()
# 'n_estimators': 61, 'max_samples': 214, 'max_features': 7, 'criterion': 'absolute_error', 'max_depth': 12, 'min_samples_split': 7, 'min_weight_fraction_leaf': 0.12244407489221353, 'min_impurity_decrease': 0.10061620281921929, 'ccp_alpha': 5.184132738860568, 'regressor': 'hgbr'}. Best is trial 242 with value: -23.52367566026926.

#%%
unseen_pred = histgb_cv["estimator"][0].predict(unseen_features)

mean_squared_error(y_true=unseen_target, y_pred=unseen_pred, squared=False)

#%%

any(unseen_pred<0)
max(unseen_pred)
min(unseen_pred)
len(unseen_pred)
np.mean(unseen_pred)
#%% prediction on submission

def create_date_features(data):
    data["date"] = pd.to_datetime(data.date)
    data["day_name"] = data.date.dt.day_name()
    data["day_of_year"] = data.date.dt.day_of_year
    data["quater"] = data.date.dt.quarter
    data["date"] = data["date"].apply(convert_holidaydate_to_dateonly)
    data["date"] = pd.to_datetime(data["date"])
    data["month_day"] = data.apply(create_month_day_parameter, axis=1)
    return data

def add_outlier_probability_features(data, global_proba_day_has_isoutier_df, 
                                     day_month_out_nonout_ratio_df,
                                     day_month_proba_df):
    data = data.merge(right=global_proba_day_has_isoutier_df, 
                    left_on="month_day", right_on="month_day",
                    )

    data = data.merge(right=day_month_out_nonout_ratio_df,
                    left_on="month_day",
                    right_on="month_day"
                    )
    data = data.merge(right=day_month_proba_df,
                    left_on="month_day",
                    right_on="month_day"
                    )
    return data
#%%
test_selected_df = test_df[all_selected_feat[:-1]]
test_selected_df = create_missing_value_features(data=test_selected_df)
test_df_with_features = create_date_features(test_selected_df)
test_df_with_features = add_outlier_probability_features(data=test_df_with_features, 
                                                  global_proba_day_has_isoutier_df=global_proba_day_has_isoutier_df,
                                                day_month_out_nonout_ratio_df=day_month_out_nonout_ratio_df,
                                                day_month_proba_df=day_month_proba_df
                                                )
test_df_with_features = create_month_emission_rate(test_df_with_features)

test_df_with_features = impute_holiday(holiday_df=holiday_df, data=test_df_with_features)
test_df_NAfeatures = train_param_imp_obj.get_features_with_missing_values(data=test_df_with_features)
train_median_records_dict = train_param_imp_obj.df_dict
test_df_with_features = train_param_imp_obj.replace_all_missing_values(df_dict=train_median_records_dict,
                                                                        empty_feature=test_df_NAfeatures,
                                                                        data=test_df_with_features
                                                                        )

#%%
def transform_data_for_predict(data, global_proba_day_has_isoutier_df,
                               day_month_out_nonout_ratio_df, day_month_proba_df,
                               holiday_df, train_median_records_dict
                               ):
    test_selected_df = create_missing_value_features(data=data)
    test_df_with_features = create_date_features(test_selected_df)
    test_df_with_features = add_outlier_probability_features(data=test_df_with_features, 
                                                    global_proba_day_has_isoutier_df=global_proba_day_has_isoutier_df,
                                                    day_month_out_nonout_ratio_df=day_month_out_nonout_ratio_df,
                                                    day_month_proba_df=day_month_proba_df
                                                    )
    test_df_with_features = create_month_emission_rate(test_df_with_features)

    test_df_with_features = impute_holiday(holiday_df=holiday_df, data=test_df_with_features)
    test_df_NAfeatures = train_param_imp_obj.get_features_with_missing_values(data=test_df_with_features)  # change train_param_imp_obj
    train_median_records_dict = train_param_imp_obj.df_dict
    test_df_with_features = train_param_imp_obj.replace_all_missing_values(df_dict=train_median_records_dict,
                                                                            empty_feature=test_df_NAfeatures,
                                                                            data=test_df_with_features
                                                                            )
    return test_df_with_features
    
#%% 
test_df_with_features[(test_df_with_features.is_holiday == 1) & 
                      (test_df_with_features.country=="Ghana") &
                      (test_df_with_features.date.dt.year == 2023)&
                      (test_df_with_features.date.dt.month==12)][["date", "month_day", "country", "is_holiday"]] 

holiday_df[(holiday_df["holiday_date"].dt.year == 2023) &
           (holiday_df.holiday_date.dt.month==12) &
           (holiday_df.holiday_country=="Ghana")
           ][["holiday_country", "holiday_date", "month_day"]]



#%%
#test_df_with_features[pred_cols]
subm_pred = histgb_cv["estimator"][0].predict(test_df_with_features[pred_cols])
test_df_sub = test_df.copy()
test_df_sub["pm2_5"] =  subm_pred
#%%
import os
subm_dir = "/home/lin/codebase/air_quality_pred/submissions"
subm14_with_holiday_feat_path = os.path.join(subm_dir, "submission14.csv")
test_df_sub[["id", "pm2_5"]].to_csv(subm14_with_holiday_feat_path, index=False)
#%%
any(subm_pred<0)

len(subm_pred)
np.mean(subm_pred)

#%%
for var in all_selected_feat:
    get_missing_data_percent(data=train_df, variable=var)


#%%  ########## explore model with features not more than 2% of missing features  ##############
selfeat_high_missing_percent = []
for var in all_selected_feat:
    miss_percent = get_missing_data_percent(data=train_df, variable=var)
    if miss_percent > 2:
        selfeat_high_missing_percent.append(var)


#%%

low_miss_feat = [var for var in X_features.columns if var not in selfeat_high_missing_percent]


#%%
X_features_with_low_missing = X_features[low_miss_feat]

#%%

wandbc_kwargs = {"project": "airquality_pm2_5_prediction"}
wandbc = WeightsAndBiasesCallback(metric_name="rmse", 
                                  wandb_kwargs=wandbc_kwargs, 
                                  as_multirun=True
                                  )
@wandbc.track_in_wandb()
def objective(trial, X_features=X_features_with_low_missing, 
              y_target=y_target,
              cv=20, verbose=3,
              scoring='neg_root_mean_squared_error'
              ):
    precprocess = make_column_transformer((one, categorical_var))
    #preprocess_pipeline =  make_column_transformer((precprocess, ),
    #                                                  n_jobs=-1)
    #learning_rate = trial.suggest_float("learning_rate", 0.03, 1, log=True)
    #max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 2, 1300, log=True)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 370, 7000, log=True)
    #l2_regularization = trial.suggest_float("l2_regularization", 0.01, 1, log=True)
    # max_iter = trial.suggest_int("max_iter", 200, 1700, log=True)
    # loss = trial.suggest_categorical("loss", ['linear', 'square', 'exponential'])
    # n_estimators = trial.suggest_int("n_estimators", 50, 1500, log=True)
    # max_samples = trial.suggest_int("max_samples", 1, 1500, log=True)
    # max_features = trial.suggest_int("max_features", 1, 70)
    # rfcriterion = trial.suggest_categorical("criterion", ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']) #Literal['squared_error', 'absolute_error', 'friedman_mse', 'poisson'] = "squared_error",
    # max_depth = trial.suggest_int("max_depth", 3, 1700, log=True)
    # min_samples_split = trial.suggest_int("min_samples_split", 2, 1300, log=True)
    # min_weight_fraction_leaf = trial.suggest_float("min_weight_fraction_leaf", 0.1, 0.5, log=True)
    
    # min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0.1, 0.5, log=True)
    
    # ccp_alpha= trial.suggest_float("ccp_alpha", 0.3, 10, log=True)
    regressor_name = trial.suggest_categorical("regressor", ["hgbr"])#["gbr","etr", "hgbr"])
    
    # histgb = HistGradientBoostingRegressor(random_state=42,
    #                                        learning_rate=learning_rate,
    #                                        max_leaf_nodes=max_leaf_nodes,
    #                                        min_samples_leaf=min_samples_leaf,
    #                                        l2_regularization=l2_regularization,
    #                                        max_iter=max_iter
    #                                        )
    # rfr = RandomForestRegressor(random_state=42, verbose=verbose,
    #                             max_depth=max_depth, n_jobs=-1,
    #                             n_estimators=n_estimators,
    #                             criterion=rfcriterion,
    #                             min_samples_split=min_samples_split,
    #                             min_samples_leaf=min_samples_leaf,
    #                             min_weight_fraction_leaf=min_weight_fraction_leaf,
    #                             min_impurity_decrease=min_impurity_decrease,
    #                             ccp_alpha=ccp_alpha, max_features=max_features
    #                             )
    # gbr = GradientBoostingRegressor(random_state=42, verbose=verbose,
    #                                 min_samples_leaf=min_samples_split,
    #                                 min_samples_split=min_samples_split,
    #                                 n_estimators=n_estimators,
    #                                 max_depth=max_depth,
    #                                 max_features=max_features,
    #                                 min_weight_fraction_leaf=min_weight_fraction_leaf,
    #                                 min_impurity_decrease=min_impurity_decrease,
    #                                 ccp_alpha=ccp_alpha,
    #                                 warm_start=True
    #                                 )
    # etr = ExtraTreesRegressor(random_state=42, verbose=verbose,
    #                           n_estimators=n_estimators,
    #                           max_depth=max_depth,
    #                           min_impurity_decrease=min_impurity_decrease,
    #                           ccp_alpha=ccp_alpha,
    #                             warm_start=True, n_jobs=-1,
    #                           min_samples_leaf=min_samples_split,
    #                         min_samples_split=min_samples_split,
    #                                 max_features=max_features,
    #                         min_weight_fraction_leaf=min_weight_fraction_leaf,
    #                           )
    # hist_study_bestparam = HistGradientBoostingRegressor(learning_rate=0.05404929713850879,
    #                                                 max_leaf_nodes=43,
    #                                                 min_samples_leaf=67,
    #                                                 l2_regularization=0.03278676565828471,
    #                                                 max_iter= 365,
    #                                                 scoring='neg_root_mean_squared_error',
    #                                                 verbose=1, random_state=42
    #                                                 )
    
    best_param_model = HistGradientBoostingRegressor(random_state=42,
                                           learning_rate=0.04267675959305963,
                                           max_leaf_nodes=187,
                                           min_samples_leaf=420,#min_samples_leaf,#378,
                                           l2_regularization=0.25375108209118796,
                                           max_iter=216
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
    
    # if regressor_name == "rfr":
    #     regressor_obj = rfr
    # elif regressor_name == "gbr":
    #     regressor_obj = gbr
    # elif regressor_name == "etr":
    #     regressor_obj = etr
    # elif regressor_name == "hgbr":
    #     regressor_obj = histgb
    
    if regressor_name == "hgbr":
        regressor_obj = best_param_model
    
    model_pipeline = make_pipeline(precprocess,
                                    regressor_obj, verbose=True
                                    )
    
     
    histgb_cv = cross_validate(estimator=model_pipeline,
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

# %%
study = optuna.create_study(direction="maximize", study_name="lowmiss_pm2_5_prediction")
logger.info("Start optimization.")
study.optimize(objective, n_trials=300, show_progress_bar=True, callbacks=[wandbc])

#%%
study.best_params

#%% best param estimate after using features with less than 2% missing values
best_param_model = HistGradientBoostingRegressor(random_state=42,
                                           learning_rate=0.04267675959305963,
                                           max_leaf_nodes=187,
                                           min_samples_leaf=378,
                                           l2_regularization=0.25375108209118796,
                                           max_iter=216
                                           )
best_param_model_pipeline = make_pipeline(precprocess,
                                best_param_model, verbose=True
                                )
    
     
best_param_model_cv = cross_validate(estimator=best_param_model_pipeline,
                        X=X_features_with_low_missing, 
                        y=y_target,
                        cv=cv, 
                        scoring=scoring,
                        return_estimator=True,
                        return_train_score=True,
                        verbose=5,
                        n_jobs=-1
                )

best_param_model_cv["test_score"].mean()
#%%
best_param_model_cv["train_score"].mean()

#%%
testing_df[X_features_with_low_missing.columns]
#%%
unseen_pred = best_param_model_cv["estimator"][0].predict(testing_df[X_features_with_low_missing.columns])

mean_squared_error(y_true=unseen_target, y_pred=unseen_pred, squared=False)

print(any(unseen_pred<0))
print(max(unseen_pred))
print(min(unseen_pred))
print(len(unseen_pred))
print(np.mean(unseen_pred))
#%%
preddata = transform_data_for_predict(data=test_df, global_proba_day_has_isoutier_df=global_proba_day_has_isoutier_df,
                            day_month_out_nonout_ratio_df=day_month_out_nonout_ratio_df,
                            day_month_proba_df=day_month_proba_df
                            )

#%%
test_df_with_features[X_features_with_low_missing.columns]
subm_pred_lowmiss = best_param_model_cv["estimator"][0].predict(test_df_with_features[X_features_with_low_missing.columns])
test_df_sub = test_df.copy()
test_df_sub["pm2_5"] =  subm_pred_lowmiss

#%%
print(any(subm_pred_lowmiss<0))
print(max(subm_pred_lowmiss))
print(min(subm_pred_lowmiss))
print(len(subm_pred_lowmiss))
print(np.mean(subm_pred_lowmiss))


#%%  ###### using LIGHTGBM with optuna   ################################
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import optuna
from sklearn.preprocessing import LabelEncoder

#%%
cat_feats = ["day_name", "month_pm25_rate"]
lgbdata = X_features_with_low_missing.copy()
labelencoder = LabelEncoder()
for col in cat_feats:
    lgbdata[col] = labelencoder.fit_transform(lgbdata[col])

for col in cat_feats:
    lgbdata[col] = lgbdata[col].astype('int')


#%% export data
lgbdata.to_csv("lgbdata.csv")
y_target.to_csv("y_target.csv")

#%%    
def objective(trial,
              cv=5, verbose=3,
              scoring='neg_root_mean_squared_error'):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 1000,
        "verbosity": -1,
        "bagging_freq": 1,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 2**10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "random_state": 42,
        #"device": "gpu",
        #"gpu_platform_id": 0,
        #"gpu_device_id": 0
    }
    regressor_name = trial.suggest_categorical("regressor", ["lgbr"])
    #precprocess = make_column_transformer((one, categorical_var))
    if regressor_name == "lgbr":
        model = lgb.LGBMRegressor(**params)
    #model_pipeline = make_pipeline(precprocess,
    #                                model, verbose=True
    #                                )
    
     
    histgb_cv = cross_validate(estimator=model,
                                X=lgbdata, 
                                y=y_target,
                                cv=cv, 
                                scoring=scoring,
                                #return_estimator=True,
                                #return_train_score=True,
                                verbose=verbose,
                                n_jobs=-1
                        )
    neg_rmse = histgb_cv["test_score"].mean()
    return neg_rmse

#%%
study = optuna.create_study(direction="maximize", study_name="lowmiss_pm2_5_prediction_lgbr")
logger.info("Start optimization.")
study.optimize(objective, n_trials=300, show_progress_bar=True, callbacks=[wandbc])



#%%
import lightgbm as lgb
params = {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 1000,
        "verbosity": -1,
        "bagging_freq": 1,
        "learning_rate": 0.1,
        "num_leaves": 200,
        "subsample": 0.5,
        "colsample_bytree": 0.5,
        "min_data_in_leaf":100,
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0
    }



#%%
#X_features_with_low_missing["day_name"] = X_features_with_low_missing["day_name"].astype(str)
from sklearn.preprocessing import LabelEncoder

cat_feats = ["day_name", "month_pm25_rate"]
lgbdata = X_features_with_low_missing.copy()
labelencoder = LabelEncoder()

for col in cat_feats:
    lgbdata[col] = labelencoder.fit_transform(lgbdata[col])

for col in cat_feats:
    lgbdata[col] = lgbdata[col].astype('int')
#%%

train_set = lgb.Dataset(data=lgbdata, 
                        label=y_target, 
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
                                                        ],
            feature_name=['ozone_o3_slant_column_number_density',
                            'uvaerosolindex_absorbing_aerosol_index','global_has_outlier_proba', 'out_nonout_ratio', 
                            'has_outlier_proba', 'day_of_year',
                            ]
            )


lgb.cv(params=params, train_set=train_set.astype("int32"), num_boost_round=500,
       folds=20)

#%%
from lightgbm import LGBMRegressor
model = LGBMRegressor()
model.fit(lgbdata, y_target)

#%%
lgbr_cv = cross_validate(estimator=model,
                                X=lgbdata, 
                                y=y_target,
                                cv=cv, 
                                scoring=scoring,
                                return_estimator=True,
                                return_train_score=True,
                                verbose=3,
                                n_jobs=-1
                        )
#%%
test_df["month_day"] = test_df.apply(create_month_day_parameter, 
                                                                 axis=1
                                                                 )


#%%
test_data_median_imputed.drop()


#%%  ########## train model by imputing outliers    ##########

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

#%%
#X = ["a", "b", "c", "d"]
kf = KFold(n_splits=20)
stratified_kf = StratifiedKFold(n_splits=20)

splits = kf.split(lgbdata)
#%%
for train, test in kf.split(lgbdata):
    print("%s %s" % (train, test))
# %%
lgbdata["pm2_5"] = y_target

train_lgbdata, test_lgbdata = train_test_split(lgbdata, random_state=42)
train_lgbdata.to_csv("train_lgbdata.csv", header=True, index=False)
test_lgbdata.to_csv("test_lgbdata.csv", header=True, index=False)

# %%
pd.read_csv("/home/lin/air_quality_pred/train_lgbdata.csv")
# %%   ##### use all data for cv  ##########
training_df, testing_df = train_test_split(all_data_with_holiday_df, test_size=0.1, random_state=42,
                                           stratify=all_data_with_holiday_df[["is_outlier", 
                                                                               "is_holiday",
                                                                               "month"
                                                                               ]]
                                           )


#%%
alldata_param_imp_obj = ParameterImputation(data=all_data_with_holiday_df, 
                                        aggregation_type="median",
                                        aggregation_var="month", 
                                        param=columns
                                        )
alldata_median_imputed_df = alldata_param_imp_obj.replace_all_missing_values()

#%%
alldata_median_dict = alldata_param_imp_obj.df_dict

# %%
stratified_splits = stratified_kf.split(X=alldata_median_imputed_df, 
                                        y=alldata_median_imputed_df["is_outlier"]
                                        )

train_cv_idx = []
test_cv_idx = []
#%%
for train_idx, test_idx in stratified_splits:
    train_cv_idx.append(train_idx)
    test_cv_idx.append(test_cv_idx)


#%%    
all_data_cv = alldata_median_imputed_df.copy()
# %%
len(train_cv_idx)
# %%
cat_feats = ["day_name", "month_pm25_rate"]
labelencoder = LabelEncoder()

day_name_labelencoder = LabelEncoder()
month_pm25_rate_labelencoder = LabelEncoder()

day_name_labelencoder.fit(all_data_cv["day_name"])
month_pm25_rate_labelencoder.fit(all_data_cv["month_pm25_rate"])
#for col in cat_feats:
all_data_cv["day_name"] = day_name_labelencoder.transform(all_data_cv["day_name"]) 
all_data_cv["month_pm25_rate"] = month_pm25_rate_labelencoder.transform(all_data_cv["month_pm25_rate"]) 

for col in cat_feats:
    all_data_cv[col] = all_data_cv[col].astype('int')

#%%

submission_test_df_date_feat = create_date_features(data=test_df)
submission_test_df_with_missing_feat = create_missing_value_features(data=submission_test_df_date_feat)
submission_test_df_with_features = create_month_emission_rate(submission_test_df_with_missing_feat)
submission_test_df_with_features = impute_holiday(holiday_df=holiday_df, data=submission_test_df_with_features)

#%% selected features for missing value imputation median value in every month
alldata_median_dict['ozone_o3_slant_column_number_density'].to_csv("ozone_o3_slant_column_number_density.csv")
alldata_median_dict['uvaerosolindex_absorbing_aerosol_index'].to_csv("uvaerosolindex_absorbing_aerosol_index.csv")


#%%
monthly_median_ozone_o3_slant_column_number_density = pd.read_csv("ozone_o3_slant_column_number_density.csv")
monthly_median_uvaerosolindex_absorbing_aerosol_index = pd.read_csv("uvaerosolindex_absorbing_aerosol_index.csv")

#%%
median_imput_dict = {"ozone_o3_slant_column_number_density": monthly_median_ozone_o3_slant_column_number_density,
   "uvaerosolindex_absorbing_aerosol_index": monthly_median_uvaerosolindex_absorbing_aerosol_index
}

empty_feature = ["ozone_o3_slant_column_number_density", "uvaerosolindex_absorbing_aerosol_index"]
#%%

def replace_all_missing_values(empty_feature, df_dict,
                                data
                                ):
    if not isinstance(empty_feature, list):
        empty_feature = [empty_feature]
    
    for column in empty_feature:
        for index, row in data.iterrows():
            if pd.isnull(row[column]):
                month = row.month
                param_imput = df_dict[column]
                nafill = param_imput[param_imput.month==month][column].values[0]
                data.loc[index, column] = nafill
    return data

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
test_set_cv1_df = pd.read_csv('/home/lin/LightGBM/cv_splitdata__/cv_1_test.csv')

prediction_cols = test_set_cv1_df.columns.to_list()
prediction_cols.pop()

#%%
submission_test_df_with_outlier_features[prediction_cols]

#%%
cat_feats = ["day_name", "month_pm25_rate"]

for col in cat_feats:
    submission_test_df_with_outlier_features[col] = labelencoder.transform(submission_test_df_with_outlier_features[col])

#%%


#submission_test_df_with_outlier_features["day_name"] = day_name_labelencoder.transform(submission_test_df_with_outlier_features["day_name"])
submission_test_df_with_outlier_features["month_pm25_rate"] = month_pm25_rate_labelencoder.transform(submission_test_df_with_outlier_features["month_pm25_rate"])

#%%
for col in cat_feats:
    submission_test_df_with_outlier_features[col] = submission_test_df_with_outlier_features[col].astype('int')
# %%
#low_miss_feat.append("pm2_5")
#all_data_cv = all_data_cv[low_miss_feat]
# %%
def export_stratified_cv_data(data, stratify_col, n_splits=20, save_dir=None,
                              selected_export_cols=None
                              ):
    stratified_kf = StratifiedKFold(n_splits=n_splits)
    stratified_splits = stratified_kf.split(X=data, 
                                        y=data[stratify_col]
                                        )
    data_paths = []
    if not save_dir:
        save_dir = os.getcwd()
        
    for i, (train_idx, test_idx) in enumerate(stratified_splits):
        trn = data.iloc[train_idx]
        tsn = data.iloc[test_idx]
        trn_path = os.path.join(save_dir, f"cv_{i+1}_train.csv")
        tsn_path = os.path.join(save_dir, f"cv_{i+1}_test.csv")
        if not selected_export_cols:
            trn.to_csv(trn_path, header=True, index=False)
            tsn.to_csv(tsn_path, header=True, index=False)
            data_paths.append((trn_path, tsn_path))
        else:
            trn[selected_export_cols].to_csv(trn_path, header=True, index=False)
            tsn[selected_export_cols].to_csv(tsn_path, header=True, index=False)
            data_paths.append((trn_path, tsn_path))
        
    return data_paths


# %%
cv_split_paths = export_stratified_cv_data(data=all_data_cv, 
                                           stratify_col="is_outlier",
                                            n_splits=20,
                                            save_dir="/home/lin/LightGBM/cv_splitdata__",
                                            selected_export_cols=low_miss_feat
                                            )
# %%

with open("/home/lin/LightGBM/cv_splitdata_path.json", "w") as file:
    dp = {"cv_paths": cv_split_paths}
    json.dump(dp, file)
# %%

with open("/home/lin/LightGBM/cv_splitdata_path.json", "r") as file:
    cv_pathdata = json.load(file)
# %%
for cvpath in cv_pathdata["cv_paths"]:
    #for trnpath, tsnpath in zip(cvpath):
    #print(cvpath[0], tsnpath[1])
    trn_path, tsn_path = cvpath
    print(trn_path)
    #print(tsn_path)
    pd.read_csv(trn_path).describe()
    #print(pd.read_csv(trn_path).describe())
# %%
test_set_cv1_df = pd.read_csv('/home/lin/LightGBM/cv_splitdata__/cv_1_test.csv')
# %%
test_set_cv1_df.columns
# %%
params_df = pd.read_csv("/home/lin/LightGBM/wandb_export_2024-06-11T23_48_01.141+02_00.csv")
# %%
params_df.sort_values(by="rmse", axis=0, inplace=True)
# %%
colsample_bytree = params_df.head(2)["colsample_bytree"].to_list()
learning_rate = params_df.head(2)["learning_rate"].to_list()
min_samples_leaf = params_df.head(2)["min_samples_leaf"].to_list()
num_leaves = params_df.head(2)["num_leaves"].to_list()
subsample = params_df.head(2)["subsample"].to_list()
#learning_rate = params_df.head(10)["learning_rate"].to_list()
#learning_rate = params_df.head(10)["learning_rate"].to_list()



# %%
import itertools

param_combine = list(itertools.product(colsample_bytree, learning_rate, min_samples_leaf, num_leaves, subsample))
# %%

for i in param_combine:
    print(i)
len(param_combine)
# %%






