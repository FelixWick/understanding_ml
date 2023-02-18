import sys

import pandas as pd
import numpy as np
import datetime

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from IPython import embed


def plot_timeseries(df, suffix):
    plt.figure()
    df.index = df['DATE']
    df['y'].plot(style='r', label="sales")
    df['yhat'].plot(style='b-.', label="prediction")
    plt.legend(fontsize=15)
    plt.ylabel("sum", fontsize=15)
    plt.tight_layout()
    plt.savefig('plots/ts_{}.pdf'.format(suffix))


def plotting(df, suffix=''):
    df = df[['y', 'yhat', 'P_ID', 'PG_ID_1', 'PG_ID_2', 'PG_ID_3', 'L_ID', 'DATE']]

    ts_data = df.groupby(['DATE'])[['y', 'yhat']].sum().reset_index()
    plot_timeseries(ts_data, 'full' + suffix)

    predictions_grouped = df.groupby('PG_ID_3')
    for name, group in predictions_grouped:
        ts_data = group.groupby(['DATE'])[['y', 'yhat']].sum().reset_index()
        plot_timeseries(ts_data, 'pg3_' + str(name) + suffix)

    predictions_grouped = df.groupby('L_ID')
    for name, group in predictions_grouped:
        ts_data = group.groupby(['DATE'])[['y', 'yhat']].sum().reset_index()
        plot_timeseries(ts_data, 'l_' + str(name) + suffix)


def eval_results(yhat, y):
    mad = np.nanmean(np.abs(y - yhat))
    print('MAD: ', mad)
    mse = np.nanmean(np.square(y - yhat))
    print('MSE: ', mse)
    wmape = np.nansum(np.abs(y - yhat)) / np.nansum(y)
    print('WMAPE: ', wmape)
    smape = 100. * np.nanmean(np.abs(y - yhat) / ((np.abs(y) + np.abs(yhat)) / 2.))
    print('SMAPE: ', smape)
    md = np.nanmean(y - yhat)
    print('MD: ', md)

    mean_y = np.nanmean(y)
    print('mean(y): ', mean_y)


def get_events(df):
    for event in [
        'Christmas',
        'Easter',
        'Labour_Day',
        'German_Unity',
        'Other_Holiday',
        'Local_Holiday_0',
        'Local_Holiday_1',
        'Local_Holiday_2'
    ]:
        for event_date in df['DATE'][df['EVENT'] == event].unique():
            for event_days in range(-10, 11):
                df.loc[df['DATE'] == pd.to_datetime(event_date) + datetime.timedelta(days=event_days), event] = event_days

    return df


def prepare_data(df):
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['dayofweek'] = df['DATE'].dt.dayofweek

    df = get_events(df)

    y = np.asarray(df['SALES'])
    X = df.drop(columns='SALES')

    return X, y


def eval_static_train_test(X_test, y_test):
    print('static test:')
    eval_results(X_test['yhat'], y_test)

    mask = (X_test['PROMOTION_TYPE'] > 0)
    print('static test promotion:')
    eval_results(X_test.loc[mask, 'yhat'], y_test[mask])

    for event in ['Easter', 'Labour_Day', 'Other_Holiday', 'Local_Holiday_1', 'Local_Holiday_2']:
        mask = (X_test[event] >= -7) & (X_test[event] <= 7)
        print('static test {}:'.format(event))
        eval_results(X_test.loc[mask, 'yhat'], y_test[mask])

    X_test['y'] = y_test
    plotting(X_test, "_static")


def eval_horizon_1(X):
    X_train = X.loc[X['DATE'] <= '2022-03-31']
    print('dynamic train:')
    eval_results(X_train['yhat'], X_train['y'])
    X_test = X.loc[X['DATE']>'2022-03-31']
    print('dynamic test:')
    eval_results(X_test['yhat'], X_test['y'])

    mask = (X_train['PROMOTION_TYPE'] > 0)
    print('dynamic train promotion:')
    eval_results(X_train.loc[mask, 'yhat'], X_train.loc[mask, 'y'])
    mask = (X_test['PROMOTION_TYPE'] > 0)
    print('dynamic test promotion:')
    eval_results(X_test.loc[mask, 'yhat'], X_test.loc[mask, 'y'])

    for event in ['Christmas', 'Easter', 'Labour_Day', 'German_Unity', 'Other_Holiday', 'Local_Holiday_0', 'Local_Holiday_1', 'Local_Holiday_2']:
        mask = (X_train[event] >= -7) & (X_train[event] <= 7)
        print('dynamic train {}:'.format(event))
        eval_results(X_train.loc[mask, 'yhat'], X_train.loc[mask, 'y'])
    for event in ['Easter', 'Labour_Day', 'Other_Holiday', 'Local_Holiday_1', 'Local_Holiday_2']:
        mask = (X_test[event] >= -7) & (X_test[event] <= 7)
        print('dynamic test {}:'.format(event))
        eval_results(X_test.loc[mask, 'yhat'], X_test.loc[mask, 'y'])

    plotting(X_test, "_hor_1")


def ewma_expertise(df, group_cols, alpha):
    df.sort_values(['DATE'], inplace=True)
    df_grouped = df.groupby(group_cols, group_keys=False)

    def latest_ewma_value(s):
        s = s.ewm(alpha=alpha, ignore_na=True).mean()
        return s.iloc[-1]

    return df_grouped['y'].apply(latest_ewma_value).to_frame(name="yhat")


def static_train_test(df_train, df_test):
    X_train, y = prepare_data(df_train)

    X_train['y'] = y
    ewma_est_df = ewma_expertise(X_train, ['P_ID', 'L_ID', 'dayofweek', 'PROMOTION_TYPE'], 0.01)

    X_test, y_test = prepare_data(df_test)
    X_test = X_test.merge(ewma_est_df, how='left', on=['P_ID', 'L_ID', 'dayofweek', 'PROMOTION_TYPE'])
    return X_test, y_test


def ewma_prediction(df, group_cols, alpha, horizon):
    df.sort_values(['DATE'], inplace=True)
    df_grouped = df.groupby(group_cols, group_keys=False)

    def emov(s):
        s = s.shift(horizon).ewm(alpha=alpha, ignore_na=True).mean()
        return s

    df['yhat'] = df_grouped['y'].apply(emov)
    return df


def horizon_forecast(df_train, df_test, horizon):
    df = pd.concat([df_train, df_test], ignore_index=True)

    X, y = prepare_data(df)

    X['y'] = y
    X = ewma_prediction(X, ['P_ID', 'L_ID', 'dayofweek', 'PROMOTION_TYPE'], 0.01, horizon)
    return X


def main(args):
    df_train = pd.read_parquet("../train.parquet.gzip")
    df_test = pd.read_parquet("../test.parquet.gzip")
    df_test_results = pd.read_parquet("../test_results.parquet.gzip")
    df_test = df_test.merge(df_test_results, how='inner', on=['P_ID', 'L_ID', 'DATE'])

    # exercise 1
    df_train["SALES"].hist(bins=200, log=True)
    plt.savefig("plots/sales.pdf")
    plt.clf()

    # exercise 2 a
    X_test, y_test = static_train_test(df_train, df_test)
    # exercise 3
    eval_static_train_test(X_test, y_test)

    # exercise 2 b
    X = horizon_forecast(df_train, df_test, 1)
    # exercise 3
    eval_horizon_1(X)

    # naive: mean(y) for all
    print('naive model train:')
    eval_results(np.mean(df_train['SALES']), df_train['SALES'])
    print('naive model test:')
    eval_results(np.mean(df_train['SALES']), df_test['SALES'])

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
