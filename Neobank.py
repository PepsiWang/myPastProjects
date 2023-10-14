# myPastProjects


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
from pandas.plotting import autocorrelation_plot
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller

"""
author: YI WANG Z5387053
date: last edited on 06/08/2023
purpose: created for FINS5545 Project assignment part B option 2 NeoBank
"""


def Station1_loadData(client):
    """
    :param: str: client name e. g 12345 in this project
    data load, data cleansing and formatting
    :return: dict cleansed dataset
    """
    # load by client name
    # current directory: pythonProject1
    # data folder: pythonProject1\data
    # result folder: pythonProject1\data\result
    filepath = "./data/"
    df_cash = pd.read_excel(filepath + "Client_Cash_Accounts (2).xlsx", sheet_name=client)
    df_weather = pd.read_csv(filepath + "Weather (2).csv", engine='python')
    df_corn = pd.read_csv(filepath + "CORN_PriceHistory (4).csv", skiprows=15,
                          usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], engine='python')
    df_wheat = pd.read_csv(filepath + "WHEAT_PriceHistory (2).csv", skiprows=15,
                           usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], engine='python')
    # for df in [df_cash, df_weather, df_corn, df_wheat]:
    #     print(df.head())
    #     print(df.tail())
    #     print(df.info())

    # clean dataset to be ready for handover to Station #2, we do this by each dataframe
    # cash account:
    # rename columns "Transaction Date" to be "Date", consistent with other files
    # sort dataframe by datetime index
    df_cash.rename(columns={"Transaction Date": "Date"}, inplace=True)
    df_cash.index = df_cash["Date"]
    df_cash = df_cash.sort_index()
    # control for any duplicates
    index = df_cash.index
    df_cash = df_cash[~index.duplicated(keep="last")]
    # add missing dates to be more suitable for time series forecast
    # reindex() conform to new datetime index
    # fill holes in reindex DataFrame with last valid observation
    # reset index
    start = df_cash["Date"].iloc[0]
    end = df_cash["Date"].iloc[-1]
    idx = pd.date_range(start, end)
    df_cash = df_cash.reindex(idx, method="ffill")
    df_cash["Date"] = df_cash.index
    df_cash = df_cash.reset_index(drop=True)
    # print(df_cash)

    # weather:
    # convert time column into readable format
    # rename column contains temperature (°C) to be temperature (C)
    # sort values by date and reset index
    df_weather.dropna(axis=0, subset=None, inplace=True, how='any', )
    df_weather["Date"] = df_weather["Date"].apply(pd.to_datetime, format="%d/%m/%Y")
    df_weather.rename(columns={"Minimum temperature (°C)": "Minimum temperature (C)",
                               "Maximum temperature (°C)": "Maximum temperature (C)",
                               "9am Temperature (°C)": "9am Temperature (C)",
                               "3pm Temperature (°C)": "3pm Temperature (C)"},
                      inplace=True)
    df_weather = df_weather.sort_values(by="Date")
    df_weather = df_weather.reset_index(drop=True)

    # corn and wheat price:
    # convert time column into readable format
    # sort values by date and reset index
    df_corn.dropna(axis=0, subset=None, inplace=True, how='any', )
    df_corn["Date"] = df_corn["Date"].apply(pd.to_datetime, format="%m/%d/%y")
    df_corn = df_corn.sort_values(by="Date")
    df_corn = df_corn.reset_index(drop=True)

    df_wheat.dropna(axis=0, subset=None, inplace=True, how='any', )
    df_wheat["Date"] = df_wheat["Date"].apply(pd.to_datetime, format="%m/%d/%y")
    df_wheat = df_wheat.sort_values(by="Date")
    df_wheat = df_wheat.reset_index(drop=True)

    # for df in [df_cash, df_weather, df_corn, df_wheat]:
    #     print(df.head())

    # create a dictionary for all dataframes to return
    df_set = {'cash account': df_cash, 'corn price': df_corn,
              'wheat price': df_wheat, 'weather': df_weather}

    return df_set


def Station2_featuresEngineeringBase(df_set):
    """
        received cleaned DataFrames from Station #1 process all relevant features
        :param: dict: dictionary contains clean dataframes passed by station1
        Receive cleaned data from Station #1 process all relevant features
        :return dict cleansed dataset
        """

    # create a dict to store relevant features to return
    dataset = {}

    # cash balance: RESET DATA INPUTS INTO ADDITIONAL FEATURES: Balance
    df_cash = df_set['cash account']
    features_cash_considered = ["Date", "Balance"]
    features_cash = df_cash[features_cash_considered]
    features_cash.set_index("Date", inplace=True)
    dataset["cash"] = features_cash
    # print(features_cash.head())

    # weather: RESET DATA INPUTS INTO ADDITIONAL FEATURES: AvgTemp(average temperature), Rainfall
    df_weather = df_set['weather']
    df_weather["AvgTemp"] = (df_weather["Minimum temperature (C)"] + df_weather["Maximum temperature (C)"]) / 2
    features_weather_considered = ["Date", "AvgTemp"]
    features_weather = df_weather[features_weather_considered]
    features_weather.set_index("Date", inplace=True)
    dataset["weather"] = features_weather
    # print(features_weather.head())

    # corn price: RESET DATA INPUTS INTO ADDITIONAL FEATURES: last price
    df_corn = df_set['corn price']
    features_corn_considered = ["Date", 'Last']
    features_corn = df_corn[features_corn_considered]
    features_corn.set_index("Date", inplace=True)
    dataset["corn price"] = features_corn
    # print(features_corn.head())

    # wheat price: RESET DATA INPUTS INTO ADDITIONAL FEATURES: last price
    df_wheat = df_set['wheat price']
    features_wheat_considered = ["Date", 'Last']
    features_wheat = df_wheat[features_wheat_considered]
    features_wheat.set_index("Date", inplace=True)
    dataset["wheat price"] = features_wheat
    # print(features_wheat.head())

    # visualization: view features in visual form
    plt.rcParams["figure.figsize"] = (25, 10)
    features_cash.plot(label="Cash Balance")
    plt.savefig("./data/feature_cash.png")
    plt.show()
    features_weather.plot(label="Average Temp")
    plt.savefig("./data/feature_weather.png")
    plt.show()
    features_corn.plot(label="Corn last price")
    plt.savefig("./data/feature_corn.png")
    plt.show()
    features_wheat.plot(label="Wheat last price")
    plt.savefig("./data/feature_wheat.png")
    plt.show()

    return dataset


class Station3_modelDesign:
    # decompose the time series data
    # to find out p, d, q values for ARIMA model
    def adfuller_test(dataset):
        result = adfuller(dataset)
        labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations']
        for value, label in zip(result, labels):
            print(label + ' : ' + str(value))
        if result[1] <= 0.05:
            print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
        else:
            print("weak evidence against null hypothesis,indicating it is non-stationary ")

    def decomposeData(dataset):
        df = pd.DataFrame()
        df['value First Difference'] = dataset - dataset.shift(1)
        df['Seasonal First Difference'] = dataset - dataset.shift(12)
        Station3_modelDesign.adfuller_test(df['Seasonal First Difference'].dropna())
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(df['Seasonal First Difference'].dropna(), lags=40, ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(df['Seasonal First Difference'].dropna(), lags=40, ax=ax2)
        plt.savefig("./data/result/pqValues.png")
        plt.show()

    # ARIMA MODEL TEST
    def ARIMA_ts(data, tt_ratio):
        # ARIMA
        X = data.values
        size = int(len(X) * tt_ratio)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        predictions = list()
        msg = ""
        for t in range(len(test)):
            model = ARIMA(history, order=(1, 1, 2))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            msg = 'progress:%', round(100 * (t / len(test))), 'predicted=%f, expected=%f' % (yhat, obs),
        error = mean_squared_error(test, predictions)
        print(msg)
        print('Test MSE: %.3f' % error)
        # with open("./data/result/ARMIA_ts_result.txt", "a") as file:
        #     file.write("\n ARIMA_ts output for cash account:")
        #     file.write(str(msg))
        #     file.write('Test MSE: %.3f' % error)
        preds = np.append(train, predictions)
        # visualizations
        plt.rcParams["figure.figsize"] = (25, 10)
        plt.plot(list(preds), color='green', linewidth=3, label="Predicted Data")
        plt.plot(list(data), color='blue', linewidth=2, label="Original Data")
        plt.axvline(x=int(len(data) * tt_ratio) - 1, linewidth=5, color='red')
        plt.legend()
        plt.savefig("./data/result/predict.png")
        plt.show()

    # create a differenced series
    def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
        diff.append(value)
        return np.array(diff)

    # invert differenced value
    def inverse_difference(history, yhat, interval=1):
        return yhat + history[-interval]

    # forecast futures target with ARIMA model
    # multi-Step ***out-of-sample forecast
    def ARIMA(dataset, future_target):
        X = dataset.values
        days_in_year = 365
        differenced = Station3_modelDesign.difference(X, days_in_year)
        # fit model
        model = ARIMA(differenced, order=(1, 1, 2))
        model_fit = model.fit()
        # multi-step out-of-sample forecast
        start_index = len(differenced)
        end_index = start_index + future_target - 1
        forecast = model_fit.predict(start=start_index, end=end_index)
        # invert the differenced forecast to something usable
        history = [x for x in X]
        predict = []
        for yhat in forecast:
            inverted = Station3_modelDesign.inverse_difference(history, yhat, days_in_year)
            inverted = inverted.round(4)
            history.append(inverted)
            predict.append(inverted)
        start_date = dataset.index[-1] + datetime.timedelta(days=1)
        idx = pd.date_range(start=start_date, periods=future_target, freq="D")
        # print predictions and write it to csv file
        df_predictions = pd.DataFrame({"Date": idx, "Prediction": predict})
        print(df_predictions)
        df_predictions.to_csv("./data/result/predict.csv", mode="a")
        # visualizations
        plt.rcParams["figure.figsize"] = (25, 10)
        preds = np.append(list(dataset), predict)
        plt.plot(list(preds), color='green', linewidth=3, label="Predicted Data")
        plt.plot(list(dataset), color='blue', linewidth=2, label="Original Data")
        plt.axvline(x=int(len(dataset) - 1), linewidth=3, color='red')
        plt.legend()
        plt.show()
        plt.close()


def main():
    # Initial client
    client = str(1)
    # Station #1
    df1 = Station1_loadData(client)
    # Station#2
    data = Station2_featuresEngineeringBase(df1)

    # Station #4 Product Implementation (Implementing Station #3)
    data_cash = data["cash"]["Balance"]
    data_weather = data["weather"]["AvgTemp"]
    data_corn = data["corn price"]["Last"]
    data_wheat = data["wheat price"]["Last"]
    dataset = [data_cash, data_weather, data_corn, data_wheat]

    # decide model parameters
    # for data in dataset:
    Station3_modelDesign.adfuller_test(data_cash)
    #     Station3_modelDesign.decomposeData(data)
    #
    # # ARIMA model testing
    # for data in dataset:
    #     tt_ratio = 0.7
    #     print(data.tail())
    #     Station3_modelDesign.ARIMA_ts(data, tt_ratio)
    #
    # # predict future 25 days (out-of-sample forecasts)
    # for data in dataset:
    #     future_target = 25
    #     Station3_modelDesign.ARIMA(data, future_target)


if __name__ == '__main__':
    main()
