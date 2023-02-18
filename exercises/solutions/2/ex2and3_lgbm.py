import sys

import pandas as pd
import numpy as np
import datetime

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor

import shap

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
            for event_days in range(-10+10, 11+10):
                df.loc[df['DATE'] == pd.to_datetime(event_date) + datetime.timedelta(days=event_days), event] = event_days

    return df


def ewma_prediction(df, ewma_column, group_cols, alpha, horizon):
    df.sort_values(['DATE'], inplace=True)
    df_grouped = df.groupby(group_cols, group_keys=False)
    return df_grouped[ewma_column].apply(lambda x: x.shift(horizon).ewm(alpha=alpha, ignore_na=True).mean())


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
    df.fillna(66, inplace=True)

    # df['ewma'] = ewma_prediction(df, 'SALES', ['P_ID', 'L_ID', 'dayofweek', 'PROMOTION_TYPE'], 0.01, 1)
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
    # 'L_ID', 'PG_ID_1', 'PG_ID_2', 'PG_ID_3', 'P_ID',
    # 'NORMAL_PRICE', 'SALES_AREA',
    # 'SCHOOL_HOLIDAY', 'PROMOTION_TYPE',
    # 'dayofweek',
    # 'dayofyear', 'month', 'dayofmonth',
    # 'td',
    # 'price_ratio',
    # 'Christmas', 'Easter', 'Labour_Day', 'German_Unity', 'Other_Holiday', 'Local_Holiday_0', 'Local_Holiday_1', 'Local_Holiday_2',
    # 'ewma'
    ml_est = HistGradientBoostingRegressor(
        categorical_features=[
            True, True, True, True, True,
            False, False,
            True, True,
            True,
            False, False, False,
            False,
            False,
            True, True, True, True, True, True, True, True,
            # False
        ],
        monotonic_cst=[
            0, 0, 0, 0, 0,
            -1, 1,
            0, 0,
            0,
            0, 0, 0,
            0,
            -1,
            0, 0, 0, 0, 0, 0, 0, 0,
            # 0
        ],
        # loss='absolute_error',
        # loss='poisson',
        # loss='quantile', quantile=0.95,
        max_bins=X['P_ID'].max()+1, random_state=0)
    ml_est.fit(X, y)

    del X
    return ml_est


def prediction(X, ml_est):
    yhat = ml_est.predict(X)

    del X
    return yhat


def evaluate_quantile(X, y):
    quantile_acc = (y <= X['yhat']).mean()
    print('fraction of actuals lower than quantile prediction: {}'.format(quantile_acc))


def shap_explanation(X_test, ml_est, sample):
    X_test = X_test.drop(columns=['yhat'])
    feature_names = X_test.columns.to_list()
    X10000 = shap.utils.sample(X_test, 10000)
    explainer = shap.Explainer(ml_est.predict, X10000, feature_names=feature_names)
    shap_values = explainer(X_test[1000:2000])

    shap.plots.bar(shap_values, show=False)
    plt.savefig('plots/shap_bar_global.png')
    plt.clf()
    shap.plots.bar(shap_values[sample], show=False)
    plt.savefig('plots/shap_bar_local.png')
    plt.clf()
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig('plots/shap_beeswarm.png')
    plt.clf()
    shap.plots.waterfall(shap_values[sample], show=False)
    plt.savefig('plots/shap_waterfall.png')
    plt.clf()


def residual_correction(X):
    group_cols = ['P_ID', 'L_ID']
    alpha = 0.01
    horizon = 1
    truth_emov = ewma_prediction(X, 'y', group_cols, alpha, horizon)
    pred_emov = ewma_prediction(X, 'yhat', group_cols, alpha, horizon)
    X['yhat'] = (truth_emov / pred_emov) * X['yhat']

    X_train = X.loc[X['DATE']<='2022-03-31']
    y_train = X.loc[X['DATE']<='2022-03-31', 'y']
    X_test = X.loc[X['DATE']>'2022-03-31']
    y_test = X.loc[X['DATE']>'2022-03-31', 'y']
    eval_static_train_test(X_train, y_train, X_test, y_test)


def main(args):
    df_train = pd.read_parquet("../train.parquet.gzip")
    df_test = pd.read_parquet("../test.parquet.gzip")
    df_test_results = pd.read_parquet("../test_results.parquet.gzip")
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

    # evaluate_quantile(X_train, y_train)
    # evaluate_quantile(X_test, y_test)

    # exercise 3
    shap_explanation(X_test, ml_est, 666)

    # X['yhat'] = pd.concat([X_train['yhat'], X_test['yhat']])
    # X['y'] = y
    # residual_correction(X)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
