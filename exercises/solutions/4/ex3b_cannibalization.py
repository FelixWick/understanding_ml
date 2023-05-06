import sys

import pandas as pd
import numpy as np
import datetime

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge

from cyclic_boosting import binning, flags, CBPoissonRegressor, observers, common_smoothers
from cyclic_boosting.smoothing.onedim import SeasonalSmoother, IsotonicRegressor

from IPython import embed


def plot_timeseries(df, suffix):
    plt.figure()
    df.index = df['DATE']
    df['y'].plot(style='r', label="sales")
    df['yhat'].plot(style='b-.', label="prediction")
    df['yhat_canni'].plot(style='g--', label="cannibalization")
    plt.legend(fontsize=15)
    plt.ylabel("sum", fontsize=15)
    plt.tight_layout()
    plt.savefig('plots/ts_canni_{}.pdf'.format(suffix))


def eval_results(yhat_mean, y):
    mad = np.nanmean(np.abs(y - yhat_mean))
    print('MAD: {}'.format(mad))
    mse = np.nanmean(np.square(y - yhat_mean))
    print('MSE: {}'.format(mse))
    mape = np.nansum(np.abs(y - yhat_mean)) / np.nansum(y)
    print('MAPE: {}'.format(mape))
    smape = 100. * np.nanmean(np.abs(y - yhat_mean) / ((np.abs(y) + np.abs(yhat_mean)) / 2.))
    print('SMAPE: {}'.format(smape))
    md = np.nanmean(y - yhat_mean)
    print('MD: {}'.format(md))

    mean_y = np.nanmean(y)
    print('mean(y): {}'.format(mean_y))


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
    df['dayofyear'] = df['DATE'].dt.dayofyear
    df['month'] = df['DATE'].dt.month
    df['dayofmonth'] = df['DATE'].dt.day

    df['td'] = (df['DATE'] - df['DATE'].min()).dt.days

    df['price_ratio'] = df['SALES_PRICE'] / df['NORMAL_PRICE']
    df['price_ratio'].fillna(1, inplace=True)
    df['price_ratio'].clip(0, 1, inplace=True)
    df.loc[df['price_ratio'] == 1., 'price_ratio'] = np.nan

    df = get_events(df)

    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    df[['L_ID', 'P_ID', 'PG_ID_1', 'PG_ID_2', 'PG_ID_3']] = enc.fit_transform(df[['L_ID', 'P_ID', 'PG_ID_1', 'PG_ID_2', 'PG_ID_3']])

    y = np.asarray(df['SALES'])
    X = df.drop(columns='SALES')
    return X, y


def feature_properties():
    fp = {}
    fp['P_ID'] = flags.IS_UNORDERED
    fp['PG_ID_1'] = flags.IS_UNORDERED
    fp['PG_ID_2'] = flags.IS_UNORDERED
    fp['PG_ID_3'] = flags.IS_UNORDERED
    fp['L_ID'] = flags.IS_UNORDERED
    fp['dayofweek'] = flags.IS_ORDERED
    fp['month'] = flags.IS_ORDERED
    fp['dayofyear'] = flags.IS_CONTINUOUS | flags.IS_LINEAR
    fp['dayofmonth'] = flags.IS_CONTINUOUS
    fp['price_ratio'] = flags.IS_CONTINUOUS | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['PROMOTION_TYPE'] = flags.IS_ORDERED
    fp['SCHOOL_HOLIDAY'] = flags.IS_ORDERED
    fp['Christmas'] = flags.IS_ORDERED  | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['Easter'] = flags.IS_ORDERED  | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['Labour_Day'] = flags.IS_ORDERED  | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['German_Unity'] = flags.IS_ORDERED  | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['Other_Holiday'] = flags.IS_ORDERED  | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['Local_Holiday_0'] = flags.IS_ORDERED  | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['Local_Holiday_1'] = flags.IS_ORDERED  | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['Local_Holiday_2'] = flags.IS_ORDERED  | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['NORMAL_PRICE'] = flags.IS_CONTINUOUS
    fp['ewma'] = flags.IS_CONTINUOUS | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['td'] = flags.IS_CONTINUOUS | flags.IS_LINEAR
    return fp


def cb_model():
    fp = feature_properties()
    explicit_smoothers = {('dayofyear',): SeasonalSmoother(order=3),
                          ('price_ratio',): IsotonicRegressor(increasing=False),
                          ('NORMAL_PRICE',): IsotonicRegressor(increasing=False),
                          ('ewma',): IsotonicRegressor(increasing=True),
                         }

    features = [
        'dayofweek',
        'L_ID',
        'PG_ID_1',
        'PG_ID_2',
        'PG_ID_3',
        'P_ID',
        'PROMOTION_TYPE',
        'price_ratio',
        'dayofyear',
        'month',
        'dayofmonth',
        'SCHOOL_HOLIDAY',
        'Christmas',
        'Easter',
        'Labour_Day',
        'German_Unity',
        'Other_Holiday',
        'Local_Holiday_0',
        'Local_Holiday_1',
        'Local_Holiday_2',
        ('L_ID', 'td'),
        ('P_ID', 'td'),
        ('P_ID', 'L_ID'),
        ('L_ID', 'dayofweek'),
        ('PG_ID_1', 'dayofweek'),
        ('PG_ID_2', 'dayofweek'),
        ('PG_ID_3', 'dayofweek'),
        ('P_ID', 'dayofweek'),
        ('L_ID', 'PG_ID_1', 'dayofweek'),
        ('L_ID', 'PG_ID_2', 'dayofweek'),
        ('L_ID', 'PG_ID_3', 'dayofweek'),
        ('SCHOOL_HOLIDAY', 'dayofweek'),
        ('SCHOOL_HOLIDAY', 'L_ID', 'dayofweek'),
        ('SCHOOL_HOLIDAY', 'PG_ID_3', 'dayofweek'),
        ('SCHOOL_HOLIDAY', 'L_ID', 'PG_ID_3', 'dayofweek'),
        ('L_ID', 'dayofmonth'),
        ('PG_ID_3', 'dayofmonth'),
        ('L_ID', 'PG_ID_3', 'dayofmonth'),
        ('L_ID', 'dayofyear'),
        ('PG_ID_3', 'dayofyear'),
        ('P_ID', 'dayofyear'),
        ('L_ID', 'PG_ID_3', 'dayofyear'),
        ('L_ID', 'Christmas'),
        ('L_ID', 'Easter'),
        ('L_ID', 'Labour_Day'),
        ('L_ID', 'German_Unity'),
        ('L_ID', 'Local_Holiday_0'),
        ('L_ID', 'Local_Holiday_1'),
        ('PG_ID_3', 'Christmas'),
        ('PG_ID_3', 'Easter'),
        ('PG_ID_3', 'Labour_Day'),
        ('PG_ID_3', 'German_Unity'),
        ('PG_ID_3', 'Local_Holiday_0'),
        ('PG_ID_3', 'Local_Holiday_1'),
        ('P_ID', 'Christmas'),
        ('P_ID', 'Easter'),
        ('P_ID', 'Labour_Day'),
        ('P_ID', 'German_Unity'),
        ('P_ID', 'Local_Holiday_0'),
        ('P_ID', 'Local_Holiday_1'),
        ('L_ID', 'PG_ID_3', 'Christmas'),
        ('L_ID', 'PG_ID_3', 'Easter'),
        ('L_ID', 'PG_ID_3', 'Labour_Day'),
        ('L_ID', 'PG_ID_3', 'German_Unity'),
        ('L_ID', 'PG_ID_3', 'Local_Holiday_0'),
        ('L_ID', 'PG_ID_3', 'Local_Holiday_1'),
        ('PROMOTION_TYPE', 'dayofweek'),
        ('price_ratio', 'dayofweek'),
        ('P_ID', 'PROMOTION_TYPE'),
        ('P_ID', 'price_ratio'),
        'NORMAL_PRICE',
    ]

    plobs = [observers.PlottingObserver(iteration=-1)]

    est = CBPoissonRegressor(
        feature_properties=fp,
        feature_groups=features,
        observers=plobs,
        maximal_iterations=50,
        smoother_choice=common_smoothers.SmootherChoiceGroupBy(
            use_regression_type=True,
            use_normalization=False,
            explicit_smoothers=explicit_smoothers),
    )

    binner = binning.BinNumberTransformer(n_bins=100, feature_properties=fp)

    ml_est = Pipeline([("binning", binner), ("CB", est)])
    return ml_est


def cb_fit(X, y):
    ml_est = cb_model()
    ml_est.fit(X, y)

    del X
    return ml_est


def cb_predict(X, ml_est):
    yhat = ml_est.predict(X)

    del X
    return yhat


def canni_prep(X):
    X_p_id_matrix = X[['P_ID', 'L_ID', 'DATE', 'PROMOTION_TYPE']]
    X_p_id_matrix = X_p_id_matrix.pivot(index=['L_ID', 'DATE'], columns='P_ID', values='PROMOTION_TYPE')
    X_p_id_matrix.reset_index(inplace=True)

    X = X.merge(X_p_id_matrix, on=['L_ID', 'DATE'], how='left')
    X.fillna(0, inplace=True)
    return X


def canni_fit_predict(X):
    X = canni_prep(X)

    ml_est = LinearRegression()
    # ml_est = Ridge()

    X_train = X.loc[X['DATE'] <= '2022-03-31']
    X_train['res_target'] = X_train['y'] / X_train['yhat']
    ml_est.fit(X_train.filter(regex='\.0'), X_train['res_target'])
    X_train['yhat_canni'] = ml_est.predict(X_train.filter(regex='\.0')) * X_train['yhat']

    X_test = X.loc[X['DATE'] > '2022-03-31']
    X_test['yhat_canni'] = ml_est.predict(X_test.filter(regex='\.0')) * X_test['yhat']

    X = pd.concat([X_train, X_test])

    X = X[['P_ID', 'L_ID', 'DATE', 'yhat', 'yhat_canni', 'y']]
    return X


def canni_model(X_train, X_test):
    X = pd.concat([X_train, X_test])
    X = X.groupby('PG_ID_3').apply(canni_fit_predict).reset_index()
    X_train = X.loc[X['DATE'] <= '2022-03-31']
    X_test = X.loc[X['DATE'] > '2022-03-31']
    return X_train, X_test


def plot_cannibalization(df):
    cannibalized_pg_3 = df.loc[df['cannibalizing'] == 1, 'PG_ID_3'].unique()
    df = df[['y', 'yhat', 'yhat_canni', 'P_ID', 'PG_ID_3', 'DATE', 'cannibalizing']]

    df_p_id = df.loc[df['cannibalizing'] == 1]
    predictions_grouped = df_p_id.groupby(['P_ID'])
    for name, group in predictions_grouped:
        ts_data = group.groupby(['DATE'])['y', 'yhat', 'yhat_canni'].sum().reset_index()
        plot_timeseries(ts_data, 'cannibalizing_p_id_' + str(name))

    df_pg_3 = df.loc[(df['PG_ID_3'].isin(cannibalized_pg_3)) & (df['cannibalizing'] == 0)]
    predictions_grouped = df_pg_3.groupby(['PG_ID_3'])
    for name, group in predictions_grouped:
        ts_data = group.groupby(['DATE'])['y', 'yhat', 'yhat_canni'].sum().reset_index()
        plot_timeseries(ts_data, 'cannibalized_pg_3_' + str(name))


def plot_price_elasticity(df):
    x = df['price_ratio']
    y = df['y']
    sns.regplot(x=x, y=y, x_bins=7, fit_reg=True)
    y = df['yhat']
    sns.regplot(x=x, y=y, x_bins=7, fit_reg=False, color='red')
    plt.savefig("plots/elasticity.pdf")
    plt.clf()


def main(args):
    np.random.seed(42)

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

    # not necessarily CB, could use any other strong model as pre-cannibalization model
    ml_est = cb_fit(X_train.copy(), y_train)

    X_train['yhat'] = cb_predict(X_train.copy(), ml_est)
    X_test['yhat'] = cb_predict(X_test.copy(), ml_est)

    X_train['y'] = y_train
    X_test['y'] = y_test

    # not relevant for this exercise, just interesting
    plot_price_elasticity(X_test[X_test['price_ratio']!=1])

    X_train, X_test = canni_model(X_train.copy(), X_test.copy())

    # in-sample
    eval_results(X_train['yhat'], X_train['y'])
    eval_results(X_train['yhat_canni'], X_train['y'])
    # out-of-sample
    eval_results(X_test['yhat'], X_test['y'])
    eval_results(X_test['yhat_canni'], X_test['y'])

    # cannibalizing products: [191, 115, 97, 158, 163, 39, 19, 90, 89, 95, 148, 136, 198, 21, 86]
    X_train['cannibalizing'] = 0
    X_train.loc[X_train['P_ID'].isin([191, 115, 97, 158, 163, 39, 19, 90, 89, 95, 148, 136, 198, 21, 86]), 'cannibalizing'] = 1
    plot_cannibalization(X_train[(X_train['DATE'] >= '2019-10-01') & (X_train['DATE'] < '2019-12-01')])

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
