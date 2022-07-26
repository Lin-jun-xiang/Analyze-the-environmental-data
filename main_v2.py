import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import category_encoders as ce
import re

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

# Read file
df_train = pd.read_excel(".//train_data.xlsx")
df_test = pd.read_excel(".//test_data.xlsx")

# Drop the first column - Unnamed
df_train, df_test = df_train.iloc[:, 1:], df_test.iloc[:, 1:]

# Drop the non-feature colums - 縣市、測站、日期、時間
df_train, df_test = df_train.iloc[:, 4:], df_test.iloc[:, 4:]

# Convert "-" to nan
df_train.replace("-", np.nan, inplace=True)
df_test.replace("-", np.nan, inplace=True)

# Check the data type : convert the string to float type if the data can be change (ignore)
print("Data_type:", df_train.info())
for col in df_train.columns:
    try:
        df_train[col] = df_train[col].astype(float)
        df_test[col] = df_test[col].astype(float)

    except:
        print(col, " cannot be convert to float type")

# EDA
def number_null(df):
    n_null = df.isnull().sum()/df.shape[0]
    fig = px.histogram(x=df.columns, y=n_null)
    fig.show()

number_null(df_train)

# Insight from the number of null
# 1. We will drop "PSI值" which have 100% null
# 2. Check the relation between "污染物" and pm2.5 - box plot
# -> In the box plot: we will keep the "污染物" and encode this feature later.
df_train.drop("PSI值", axis=1, inplace=True)
df_test.drop("PSI值", axis=1, inplace=True)

fig = px.box(df_train.sort_values(by=["細懸浮微粒(μg/m<sup>3</sup>)"]), x="污染物", y="細懸浮微粒(μg/m<sup>3</sup>)")
fig.show()

# Correlation
fig = px.imshow(df_train.corr())
fig.update_layout(
    font_size=8
)
fig.show()
print("Corr with pm2.5", df_train.corr()["細懸浮微粒(μg/m<sup>3</sup>)"])

# Feature Engineering

# We find the "風向" have lower correlation with pm2.5
# Let's check out the feature
# We convert the "風向degree" to "東西南北", then see the corr with pm2.5 through box plot
# When wind_direction = (1, 2) -> (45-135 degree) -> (東北風-東南風) , will have higher pm2.5
# Therefore, we can suppose the "東風無法將pm2.5吹散, 因為中央山脈的阻隔"
wind_direction = np.arange(0, 8) # 東、西、東南、...
for i, degree in enumerate(np.linspace(45, 360, 8)):
    filter_ = ((df_train["風向(degrees)"] <= degree) & (df_train["風向(degrees)"] > degree-45))
    df_train.loc[filter_, "風向(degrees)"] = wind_direction[i]
    filter_ = ((df_test["風向(degrees)"] <= degree) & (df_test["風向(degrees)"] > degree-45))
    df_test.loc[filter_, "風向(degrees)"] = wind_direction[i]

fig = px.box(x=df_train["風向(degrees)"], y=df_train["細懸浮微粒(μg/m<sup>3</sup>)"])
fig.show()

# Standard deviation - drop features which have lower std(non-infomation data)
print("STD: ", df_train.std())

df_train.drop(["一氧化碳(ppm)", "二氧化硫(ppb)", "二氧化氮(ppb)", "一氧化碳8小時移動平均(ppb)", "一氧化氮(ppb)(NO)"], axis=1, inplace=True)
df_test.drop(["一氧化碳(ppm)", "二氧化硫(ppb)", "二氧化氮(ppb)", "一氧化碳8小時移動平均(ppb)", "一氧化氮(ppb)(NO)"], axis=1, inplace=True)

# Encoding - Target encoding
X_train = df_train.drop(["細懸浮微粒(μg/m<sup>3</sup>)"], axis=1)
y_train = df_train["細懸浮微粒(μg/m<sup>3</sup>)"]

ce_target = ce.TargetEncoder(cols=df_train.select_dtypes(object))
ce_target.fit(X_train, y_train)

X_train = ce_target.transform(X_train, y_train)
X_test = ce_target.transform(df_test.drop("細懸浮微粒(μg/m<sup>3</sup>)", axis=1))

# Missing value - interpolation for time series data
X_train = X_train.astype(float).interpolate(method="linear")
y_train = df_train["細懸浮微粒(μg/m<sup>3</sup>)"].astype(float).interpolate(method="linear")
X_test = X_test.astype(float).interpolate(method="linear")
y_test = df_test["細懸浮微粒(μg/m<sup>3</sup>)"].astype(float).interpolate(method="linear")

# Modeling

# rmse metric
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def find_best_model(X_train, X_test, y_train, y_test):
    # RandomForestRegression
    rfg = RandomForestRegressor(random_state=42)
    rfg.fit(X_train, y_train)
    y_pred = rfg.predict(X_test)
    rfg_rmse = rmse(y_test, y_pred)

    # GBM
    gbm = GradientBoostingRegressor(random_state=42, loss='huber')
    gbm.fit(X_train, y_train)
    y_pred = gbm.predict(X_test)
    gbm_rmse = rmse(y_test, y_pred)

    # lgb
    lightgbm = LGBMRegressor(random_state=42, objective='regression')
    lightgbm.fit(X_train, y_train)
    y_pred = lightgbm.predict(X_test)
    lightgbm_rmse = rmse(y_test, y_pred)

    # xgb
    xgb = XGBRegressor(random_state=42, objective='reg:linear')
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    xgb_rmse = rmse(y_test, y_pred)

    df_model = pd.DataFrame({"Model":["RandomForest", "GBM", "LGB", "XGBoost"],
                           "RMSE":[rfg_rmse, gbm_rmse, lightgbm_rmse, xgb_rmse]})
    print(df_model)


# Avoid the xgboost bug : feature_names must be string, and may not contain [, ] or <
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
X_train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_train.columns.values]

dataset = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

find_best_model(*dataset)

# XGB
xgb = XGBRegressor(random_state=42, n_jobs=-1, objective='reg:linear') # n_job - use all cores on machine
xgb.fit(X_train, y_train)

# Prediction
y_pred = xgb.predict(X_test)

xgb.score(X_test, y_test) # return 86%

# Show the important feature
fig = px.histogram(x=X_train.columns, y=xgb.feature_importances_)
fig.show()

# Results Plot
fig = go.Figure()

scatter1 = go.Scatter(x=[i for i in range(len(y_test))], y=y_test,
                      name='true value',
                      mode='markers',
                      # setting marker (points) style
                      marker = {"color":"#D16BA5", "size": 5}, opacity=0.8)
fig.add_trace(scatter1)

line1 = go.Scatter(x=[i for i in range(len(y_pred))], y=y_pred,
                        mode='lines',
                        name='predict value',
                        # setting line style
                        line={"color":"#86A8E7"}, opacity=0.6)
fig.add_trace(line1)

# Update layout
fig.update_layout(title="Prediction of PM2.5",
                  xaxis_title="times",
                  yaxis_title="pm2.5 (μg/m3)",
                  font_family="Courier New",
                  font_color="blue",
                  title_font_family="Times New Roman",
                  title_font_color="red",
                  legend_title_font_color="green",
                  legend_font_color="green",
                  xaxis={"color":"black"},
                  yaxis={"color":"black"})
