import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import plotly.express as px

def missingValue():
    """
    Solve missing value of pm2.5
    """
    # Convert "-"" to nan 
    df.replace("-", np.nan, inplace=True)

    # Using interpolation method
    df["細懸浮微粒(μg/m<sup>3</sup>)"] = df["細懸浮微粒(μg/m<sup>3</sup>)"].astype(float).interpolate(method="linear")

def boxplotor():
    """
    Visualize the outlier of each month
    """
    # Convert the string data to date data type
    df_date = pd.to_datetime(df['監測日期'], format='%Y-%m-%d')

    # Get month of each data
    df_months = df_date.dt.month

    # Merge df_months to df
    df["month"] = df_months

    # sns.boxplot(x="month", y="細懸浮微粒(μg/m<sup>3</sup>)", data=df)
    # plt.xlabel("month")
    # plt.ylabel("pm2.5")
    # plt.title("BoxPlot(2018~2019)")
    fig = px.box(df, x="month", y="細懸浮微粒(μg/m<sup>3</sup>)",
                 labels={'細懸浮微粒(μg/m<sup>3</sup>)':'pm2.5'},
                 color="month",
                 title="BoxPlot(2018~2019)")
    fig.show()

def findOutlier():
    """
    Analyze the outlier of each month
    Visualize the counts of outlier at each times
    """
    # months = [str(i).rjust(2, '0') for i in range(1, 13)]
    months = [month for month in range(1, 13)]

    # Get all of outlier according to each month
    outlier = pd.DataFrame()
    for month in months:
        q1, q3 = np.percentile(df[df["month"]==month]["細懸浮微粒(μg/m<sup>3</sup>)"], [25, 75])
        above = q3 + 1.5 * (q3 - q1)
        below = q1 - 1.5 * (q3 - q1)
        filters_outlier = (df["細懸浮微粒(μg/m<sup>3</sup>)"] > above) | (df["細懸浮微粒(μg/m<sup>3</sup>)"] < below)
        filters_month = (df["month"]==month)
        if outlier.empty:
            outlier = df[filters_outlier & filters_month]
        else:
            outlier = pd.concat([outlier, df[filters_outlier & filters_month]])

    # Sort data according to 監測時間 (hr)
    outlier.sort_values(by=["監測時間"], inplace=True)

    # Groupby aggregation
    pdf = outlier.groupby(by=["監測時間"]).count()
    pdf.rename(columns={'Unnamed: 0':'counts'}, inplace=True)
    pdf["監測時間"] = [str(time) for time in range(24)]

    # Visualize the outlier
    color_continuous_scale=[[0, '#5ee7df'], [1, '#b490ca']]
    # plt.barh([str(time) for time in range(24)], pdf, color="hotpink")
    fig = px.bar(pdf, y='監測時間', x='counts',
                 color='counts',
                 labels={'counts':'counts of outlier'},
                #  template = "plotly_dark",
                 color_continuous_scale=color_continuous_scale,
                 orientation='h')
    fig.layout.coloraxis.colorbar.title = 'counts'
    fig.update_traces(textfont_size=12,
                      textangle=0,
                      textposition="outside",
                      cliponaxis=False,
                      opacity=0.9)

    fig.show()


if __name__ == "__main__":
    file = 'data.xlsx'
    df = pd.read_excel(file)
    missingValue()

    times = df["監測時間"].drop_duplicates()
    times = times[::-1]

    winter_months = df["監測日期"].str.contains('/11/|/12/|/01/|/02/|/03/')
    winter_data = df[winter_months]

    res = []
    for time in times:
        res.append(winter_data["細懸浮微粒(μg/m<sup>3</sup>)"][winter_data["監測時間"]==time].mean())

    times_tick = [i for i in range(24)]
    plt.bar(times_tick, res, alpha=0.5)
    plt.ylabel('mean of pm2.5', fontsize=16)
    plt.xlabel('times(hr)', fontsize=16)
    plt.title("pm2.5 of each time")

    boxplotor()
    findOutlier()
