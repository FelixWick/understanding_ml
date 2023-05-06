import sys

import pandas as pd
import numpy as np
import datetime

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from scipy.stats import nbinom, poisson

from cyclic_boosting import binning, flags, CBPoissonRegressor, CBNBinomC, CBExponential, observers, common_smoothers
from cyclic_boosting.smoothing.onedim import SeasonalSmoother, IsotonicRegressor
from cyclic_boosting.plots import plot_analysis

from IPython import embed


def plot_CB(filename, plobs, binner):
    for i, p in enumerate(plobs):
        plot_analysis(
            plot_observer=p,
            file_obj="plots/" + filename + "_{}".format(i), use_tightlayout=False,
            binners=[binner]
        )


def plot_timeseries(df, suffix):
    plt.figure()
    df.index = df['DATE']
    df['y'].plot(style='r', label="sales")
    df['yhat_mean'].plot(style='b-.', label="prediction")
    plt.legend(fontsize=15)
    plt.ylabel("sum", fontsize=15)
    plt.tight_layout()
    plt.savefig('plots/ts_{}.pdf'.format(suffix))


def plotting(df, suffix=''):
    df = df[['y', 'yhat_mean', 'P_ID', 'PG_ID_1', 'PG_ID_2', 'PG_ID_3', 'L_ID', 'DATE']]

    ts_data = df.groupby(['DATE'])[['y', 'yhat_mean']].sum().reset_index()
    plot_timeseries(ts_data, 'full' + suffix)

    predictions_grouped = df.groupby(['PG_ID_3'])
    for name, group in predictions_grouped:
        ts_data = group.groupby(['DATE'])['y', 'yhat_mean'].sum().reset_index()
        plot_timeseries(ts_data, 'pg3_' + str(name) + suffix)

    predictions_grouped = df.groupby(['L_ID'])
    for name, group in predictions_grouped:
        ts_data = group.groupby(['DATE'])['y', 'yhat_mean'].sum().reset_index()
        plot_timeseries(ts_data, 'l_' + str(name) + suffix)

    # predictions_grouped = df.loc[df['PG_ID_3'] == 16].groupby(['P_ID'])
    # for name, group in predictions_grouped:
    #     ts_data = group.groupby(['DATE'])['y', 'yhat_mean'].sum().reset_index()
    #     plot_timeseries(ts_data, 'p_' + str(name) + suffix)
    # predictions_grouped = df.loc[df['PG_ID_3'] == 16].groupby(['P_ID', 'L_ID'])
    # for name, group in predictions_grouped:
    #     ts_data = group.groupby(['DATE'])['y', 'yhat_mean'].sum().reset_index()
    #     plot_timeseries(ts_data, 'pl_' + str(name) + suffix)


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
    # comment out for CBexp
    df.loc[df['price_ratio'] == 1., 'price_ratio'] = np.nan

    df = get_events(df)

    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    df[['L_ID', 'P_ID', 'PG_ID_1', 'PG_ID_2', 'PG_ID_3']] = enc.fit_transform(df[['L_ID', 'P_ID', 'PG_ID_1', 'PG_ID_2', 'PG_ID_3']])

    # df['ewma'] = ewma_prediction(df, ['P_ID', 'L_ID', 'dayofweek', 'PROMOTION_TYPE'], 0.01)
    # df['ewma'].fillna(df['SALES'].mean(), inplace=True)

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


def cb_mean_model():
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
        # 'ewma',
    ]

    price_features = [
        'L_ID',
        'PG_ID_1',
        'PG_ID_2',
        'PG_ID_3',
        'P_ID',
        'dayofweek',
    ]

    plobs = [observers.PlottingObserver(iteration=1), observers.PlottingObserver(iteration=-1)]

    est=CBPoissonRegressor(
        feature_properties=fp,
        feature_groups=features,
        observers=plobs,
        maximal_iterations=50,
        smoother_choice=common_smoothers.SmootherChoiceGroupBy(
            use_regression_type=True,
            use_normalization=False,
            explicit_smoothers=explicit_smoothers),
    )
    # est = CBExponential(
    #     feature_properties=fp,
    #     standard_feature_groups=features,
    #     external_feature_groups=price_features,
    #     external_colname='price_ratio',
    #     observers=plobs,
    #     maximal_iterations=50,
    #     smoother_choice=common_smoothers.SmootherChoiceGroupBy(
    #         use_regression_type=True,
    #         use_normalization=False,
    #         explicit_smoothers=explicit_smoothers),
    # )

    binner = binning.BinNumberTransformer(n_bins=100, feature_properties=fp)

    ml_est = Pipeline([("binning", binner), ("CB", est)])
    return ml_est


def mean_fit(X, y):
    ml_est_mean = cb_mean_model()
    ml_est_mean.fit(X, y)

    plot_CB('analysis_CB_mean_iterlast', [ml_est_mean[-1].observers[0], ml_est_mean[-1].observers[-1]], ml_est_mean[-2])

    del X
    return ml_est_mean


def mean_predict(X, ml_est_mean):
    yhat_mean = ml_est_mean.predict(X)

    del X
    return yhat_mean


def cb_width_model():
    fp = feature_properties()
    fp['yhat_mean_feature'] = flags.IS_CONTINUOUS | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    explicit_smoothers = {}

    features = [
        'yhat_mean_feature',
        'dayofweek',
        'L_ID',
        'PG_ID_1',
        'PG_ID_2',
        'PG_ID_3',
        'P_ID',
        'PROMOTION_TYPE',
    ]

    plobs = [observers.PlottingObserver(iteration=1), observers.PlottingObserver(iteration=-1)]

    est = CBNBinomC(
        mean_prediction_column='yhat_mean',
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


def width_fit(X, y):
    ml_est_width = cb_width_model()
    ml_est_width.fit(X, y)

    plot_CB('analysis_CB_width_iterlast', [ml_est_width[-1].observers[0], ml_est_width[-1].observers[-1]], ml_est_width[-2])

    del X
    return ml_est_width


def width_predict(X, ml_est_width):
    c = ml_est_width.predict(X)

    variance = X['yhat_mean'] + c * X['yhat_mean'] * X['yhat_mean']

    del X
    return variance, c


def transform_nbinom(mean, var):
    p = np.minimum(np.where(var > 0, mean / var, 1 - 1e-8), 1 - 1e-8)
    n = np.where(var > 0, mean * p / (1 - p), 1)
    return n, p


def random_from_cdf_interval(X, mode='nbinom'):
    if mode == 'nbinom':
        cdf_high = nbinom.cdf(X['y'], X['n'], X['p'])
        cdf_low = nbinom.cdf(np.where(X['y'] >= 1., X['y'] - 1., 0.), X['n'], X['p'])
    elif mode == 'poisson':
        cdf_high = poisson.cdf(X['y'], X['yhat_mean'])
        cdf_low = poisson.cdf(np.where(X['y'] >= 1., X['y'] - 1., 0.), X['yhat_mean'])
    cdf_low = np.where(X['y'] == 0., 0., cdf_low)
    return cdf_low + np.random.uniform(0, 1, len(cdf_high)) * (cdf_high - cdf_low)


def cdf_truth(X):
    X['n'], X['p'] = transform_nbinom(X['yhat_mean'], X['yhat_var'])
    X['cdf_truth'] = random_from_cdf_interval(X, mode='nbinom')
    X['cdf_truth_poisson'] = random_from_cdf_interval(X, mode='poisson')
    return X


def plot_cdf_truth(cdf_truth, suffix):
    plt.figure()
    plt.hist(cdf_truth, bins=30)
    if suffix == 'nbinom':
        plt.title('NBD', fontsize=15)
    elif suffix == 'poisson':
        plt.title('Poisson', fontsize=15)
    else:
        plt.title(suffix, fontsize=20)
    plt.xlabel("CDF values", fontsize=15)
    plt.ylabel("count", fontsize=15)
    plt.hlines(100000./30, xmin=0, xmax=1, linestyles ="dashed")
    plt.tight_layout(rect=(0,0,1,0.99))
    plt.savefig('plots/cdf_truth_' + suffix + '.pdf')
    plt.clf()


def wasserstein(p, q, n_bins):
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    # scale by 1/0.5 (0.5 is maximum)
    wasser_distance = 2.0 * np.sum(np.abs(cdf_p - cdf_q)) / len(cdf_p)
    return wasser_distance * n_bins / (n_bins - 1.0)


def cdf_accuracy(cdf_truth):
    counts = cdf_truth.value_counts(bins=100)
    n_cdf_bins = len(counts)
    pmf = counts / np.sum(counts)  # relative frequencies for each bin
    unif = np.full_like(pmf, 1.0 / len(pmf))  # uniform distribution
    divergence = wasserstein(unif, pmf, n_cdf_bins)
    cdf_acc = np.float(1.0 - np.clip(divergence, a_min=None, a_max=1.0))
    print('cdf accuracy: {}'.format(cdf_acc))


def ewma_prediction(df, group_cols, alpha):
    df.sort_values(['DATE'], inplace=True)
    df_grouped = df.groupby(group_cols, group_keys=False)
    return df_grouped['SALES'].apply(lambda x: x.shift(1).ewm(alpha=alpha, ignore_na=True).mean())


def evaluate_quantile(X, quantile):
    quantile_acc = (nbinom.cdf(X['y'], X['n'], X['p']) <= quantile).mean()
    print('fraction of actuals lower than quantile prediction: {}'.format(quantile_acc))


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

    ml_est_mean = mean_fit(X_train.copy(), y_train)

    X_train['yhat_mean'] = mean_predict(X_train.copy(), ml_est_mean)
    X_test['yhat_mean'] = mean_predict(X_test.copy(), ml_est_mean)

    X_train['yhat_mean_feature'] = X_train['yhat_mean']
    ml_est_width = width_fit(X_train.copy(), np.float64(y_train))
    X_train['yhat_var'], X_train['c'] = width_predict(X_train.copy(), ml_est_width)
    X_test['yhat_mean_feature'] = X_test['yhat_mean']
    X_test['yhat_var'], X_test['c'] = width_predict(X_test.copy(), ml_est_width)

    # in-sample
    X_train['y'] = y_train
    eval_results(X_train['yhat_mean'], X_train['y'])
    # out-of-sample
    X_test['y'] = y_test
    eval_results(X_test['yhat_mean'], X_test['y'])

    # naive: mean(y) for all
    print('naive model train:')
    eval_results(np.mean(y_train), X_train['y'])
    print('naive model test:')
    eval_results(np.mean(y_test), X_test['y'])
    # Bayes error: use simulated mean
    # print('Bayes error train:')
    # eval_results(X_train['LAMBDA'], X_train['y'])
    # print('Bayes error test:')
    # eval_results(X_test['LAMBDA'], X_test['y'])

    X_train = cdf_truth(X_train)
    plot_cdf_truth(X_train['cdf_truth'], 'nbinom_train')
    plot_cdf_truth(X_train['cdf_truth_poisson'], 'poisson_train')
    cdf_accuracy(X_train['cdf_truth'])

    X_test = cdf_truth(X_test)
    plot_cdf_truth(X_test['cdf_truth'], 'nbinom')
    plot_cdf_truth(X_test['cdf_truth_poisson'], 'poisson')
    cdf_accuracy(X_test['cdf_truth'])

    evaluate_quantile(X_train, 0.5)
    evaluate_quantile(X_test, 0.5)
    evaluate_quantile(X_train, 0.95)
    evaluate_quantile(X_test, 0.95)

    # plotting(X_train[X_train['DATE'] >= '2021-09-01'])
    plotting(X_test)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
