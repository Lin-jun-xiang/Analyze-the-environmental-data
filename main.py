# HW08_406235002
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
# ---------------------------------------------------------- DataSet
file = 'data.xlsx'
df = pd.read_excel(file)
headList = list(df.columns[1:])
df2 = df[headList[7:23]].apply(pd.to_numeric, errors='coerce')  # ALL data (type=float), if data have '-', change to 'nan'
data = np.array(df2, dtype='float64')  # ALL features
# let 'nan' be the mean of data
def nan_to_mean(input_data):
    n = []
    n_index = []
    for y, x in enumerate(input_data):
        if np.isnan(x):
            n_index.append(y)
        else:
            n.append(x)
    for i in n_index:
        input_data[i] = np.mean(n)
    return input_data

list(map(nan_to_mean, data.T))
# ---------------------------------------------------------- PCA 找出那些features與pm2.5有類似成因
X_std = StandardScaler().fit_transform(data)  # 標準化
dfX_std = pd.DataFrame(X_std)

nPC = 10  # set PC number (would not affect the results)
model_pca = PCA(n_components=nPC)  # unsupervised model for reduction
model_pca.fit(dfX_std)
cmat = np.corrcoef(X_std.T)  # use correlation matrix
cmat = np.cov(X_std.T)  # use covariance matrix; feature之間的關聯性; 可利用此數據推論哪些features與pm2.5有相關性
eig_vals, eig_vecs = np.linalg.eig(cmat)
# print(eig_vals[0:nPC].real)
print('pc1 Var(%):', model_pca.explained_variance_ratio_[0] * 100)
print('pc2 Var(%):', model_pca.explained_variance_ratio_[1] * 100)
# print(model_pca.transform(X_std))

pca_components = np.zeros(shape = (data.shape[1], 10))
for i in range(len(model_pca.components_)):
    pca_components[:, i] = model_pca.components_.T[:, i] * np.sqrt(eig_vals)[i]

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
mpl.rcParams['font.family'] = 'Microsoft Yahei'
i = 0
for x, y in zip(pca_components[:, 0], pca_components[:, 1]):  # xy分別為 pc1, pc2
    ax1.scatter(x, y)
    plt.annotate("(%s)" % df2[headList[7:22]].columns[i], xy=[x, y], xytext=(-20, 10), textcoords='offset points')
    i += 1

ax1.set_xlabel("pc1")
ax1.set_ylabel("pc2")
plt.show()
# ----------------------------------------------------------  Model 訓練與評估-linear regression
X = np.hstack((data[:, :4], data[:, 5:15]))  # features without pm2.5
y = data[:, 4]  # target pm2.5
kf = KFold(n_splits=5, shuffle=False)
kf.get_n_splits(X)
avg = []
for i in range(1, 3):
    poly = PolynomialFeatures(i)
    X = poly.fit_transform(X)
    model = LinearRegression()
    scr = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X[train_index], y[train_index])
        scr.append(model.score(X[test_index], y[test_index]))
    print('when the PolynomialFeatures=', i, ', Return the coefficient of determination R^2 of prediction:', np.mean(scr))

# ---------------------------------------------------------- Start predict
file = 'data.xlsx'
df_2019 = pd.read_excel(file)
headList_2019 = list(df_2019.columns[1:])
df2_2019 = df_2019[headList_2019[7:23]].apply(pd.to_numeric, errors='coerce')  # ALL data (type=float), if data have '-', change to 'nan'
data_2019 = np.array(df2_2019, dtype='float64')  # ALL features
# let 'nan' be the mean of data
list(map(nan_to_mean, data_2019.T))

X_2019 = np.hstack((data_2019[:, :4], data_2019[:, 5:15]))  # features without pm2.5
y_2019 = data_2019[:, 4]

X = np.hstack((data[:, :4], data[:, 5:15]))
y = data[:, 4]
poly = PolynomialFeatures(1)
X = poly.fit_transform(X)
X_2019 = poly.fit_transform(X_2019)
model = LinearRegression()
model.fit(X, y)
y_pred_2019 = model.predict(X_2019)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(range(len(y_2019)), y_2019, color='b', label='true value')
ax2.plot(y_pred_2019, color='r', label='predict value')
ax2.legend(loc = 'upper right')
ax2.set_xlabel("Time")
ax2.set_ylabel("μg/m3")
xrange = list(np.arange(0, 8500, 2150))
ax2.set_xticks(xrange)
ax2.set_xticklabels(['2019-12', '2019-9', '2019-6', '2019-3'], rotation=45)

# ---------------------------------------------------------- Add the plotly visualization
import plotly.graph_objects as go
fig = go.Figure()

# Add scatter traces
scatter1 = go.Scatter(x=[i for i in range(len(y_2019))], y=y_2019,
                      name='true value',
                      mode='markers',
                      # setting marker (points) style
                      marker = {"color":"#D16BA5", "size": 5}, opacity=0.8)
fig.add_trace(scatter1)

# Add line traces
line1 = go.Scatter(x=[i for i in range(len(y_2019))], y=y_pred_2019,
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

# Rotational x-axis tick
fig.update_xaxes(tickangle = 45)

# Set custom x-axis labels
fig.update_xaxes(
    ticktext=['2019-12', '2019-9', '2019-6', '2019-3'],
    tickvals=xrange
)
fig.show()
