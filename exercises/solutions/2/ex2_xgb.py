import sys

import pandas as pd
import numpy as np
import datetime

from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

from IPython import embed


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


def ewma_prediction(df, group_cols, alpha):
    df.sort_values(['DATE'], inplace=True)
    df_grouped = df.groupby(group_cols, group_keys=False)
    return df_grouped['SALES'].apply(lambda x: x.shift(1).ewm(alpha=alpha, ignore_na=True).mean())


def prepare_data(df):
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['dayofweek'] = df['DATE'].dt.dayofweek
    df['dayofyear'] = df['DATE'].dt.dayofyear
    df['month'] = df['DATE'].dt.month
    df['dayofmonth'] = df['DATE'].dt.day

    df['td'] = (df['DATE'] - df['DATE'].min()).dt.days

    df['price_ratio'] = df['SALES_PRICE'] / df['NORMAL_PRICE']
    df['price_ratio'].fillna(1, inplace=True)
    df['price_ratio'].clip(0, 1, inplace=True)

    df = get_events(df)
    df.fillna(-11, inplace=True)

    # df['ewma'] = ewma_prediction(df, ['P_ID', 'L_ID', 'dayofweek', 'PROMOTION_TYPE'], 0.01)
    # df['ewma'].fillna(df['SALES'].mean(), inplace=True)

    y = np.asarray(df['SALES'])
    X = df.drop(columns='SALES')

    return X, y


def eval_static_train_test(X_train, y_train, X_test, y_test):
    print('train:')
    eval_results(X_train['yhat'], y_train)

    print('test:')
    eval_results(X_test['yhat'], y_test)

    mask = (X_train['PROMOTION_TYPE'] > 0)
    print('train promotion:')
    eval_results(X_train.loc[mask, 'yhat'], y_train[mask])
    mask = (X_test['PROMOTION_TYPE'] > 0)
    print('test promotion:')
    eval_results(X_test.loc[mask, 'yhat'], y_test[mask])

    for event in ['Easter', 'Labour_Day', 'Other_Holiday', 'Local_Holiday_1', 'Local_Holiday_2']:
        mask = (X_train[event] >= -7) & (X_train[event] <= 7)
        print('train {}:'.format(event))
        eval_results(X_train.loc[mask, 'yhat'], y_train[mask])
    for event in ['Easter', 'Labour_Day', 'Other_Holiday', 'Local_Holiday_1', 'Local_Holiday_2']:
        mask = (X_test[event] >= -7) & (X_test[event] <= 7)
        print('test {}:'.format(event))
        eval_results(X_test.loc[mask, 'yhat'], y_test[mask])


def training(X, y):
    # ml_est = GradientBoostingRegressor(random_state=0)
    ml_est = XGBRegressor(random_state=0)
    ml_est.fit(X, y)

    del X
    return ml_est


def prediction(X, ml_est):
    yhat = ml_est.predict(X)

    del X
    return yhat


def main(args):
    df_train = pd.read_parquet("../../train.parquet.gzip")
    df_test = pd.read_parquet("../../test.parquet.gzip")
    df_test_results = pd.read_parquet("../../test_results.parquet.gzip")
    df_test = df_test.merge(df_test_results, how='inner', on=['P_ID', 'L_ID', 'DATE'])
    df = pd.concat([df_train, df_test], ignore_index=True)

    X, y = prepare_data(df)

    X_train = X.loc[X['DATE']<='2022-03-31']
    y_train = y[X['DATE']<='2022-03-31']
    X_test = X.loc[X['DATE']>'2022-03-31']
    y_test = y[X['DATE']>'2022-03-31']

    features = [
        'L_ID', 'PG_ID_1', 'PG_ID_2', 'PG_ID_3', 'P_ID',
        'NORMAL_PRICE',
        'SALES_AREA',
        'SCHOOL_HOLIDAY',
        'PROMOTION_TYPE',
        'dayofweek',
        'dayofyear',
        'month',
        'dayofmonth',
        'td',
        'price_ratio',
        'Christmas',
        'Easter',
        'Labour_Day',
        'German_Unity',
        'Other_Holiday',
        'Local_Holiday_0',
        'Local_Holiday_1',
        'Local_Holiday_2',
        # 'ewma'
    ]

    X_train = X_train[features]
    X_test = X_test[features]

    ml_est = training(X_train.copy(), y_train)

    # in-sample
    X_train['yhat'] = prediction(X_train.copy(), ml_est)
    # out-of-sample
    X_test['yhat'] = prediction(X_test.copy(), ml_est)

    # quasi-dynamic if ewma is included as feature with horizon (e.g., 1) like above
    eval_static_train_test(X_train, y_train, X_test, y_test)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
