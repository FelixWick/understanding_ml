import sys

import pandas as pd
import numpy as np
import datetime

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from cyclic_boosting import binning, flags, CBPoissonRegressor, observers, common_smoothers
from cyclic_boosting.plots import plot_analysis

from IPython import embed


def plot_CB(filename, plobs, binner):
    for i, p in enumerate(plobs):
        plot_analysis(
            plot_observer=p,
            file_obj="plots/" + filename + "_{}".format(i), use_tightlayout=False,
            binners=[binner]
        )


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
    df['dayofmonth'] = df['DATE'].dt.day
    df["WEEK_OF_YEAR"] = df["DATE"].dt.isocalendar().week.astype(np.int16)
    df['month'] = df['DATE'].dt.month

    df['td'] = (df['DATE'] - df['DATE'].min()).dt.days

    df = get_events(df)
    df.fillna(-11, inplace=True)

    y = np.asarray(df['PROMOTION_TYPE'] > 0, dtype=int)

    return df, y


def feature_properties():
    fp = {}
    fp['P_ID'] = flags.IS_UNORDERED
    fp['PG_ID_1'] = flags.IS_UNORDERED
    fp['PG_ID_2'] = flags.IS_UNORDERED
    fp['PG_ID_3'] = flags.IS_UNORDERED
    fp['L_ID'] = flags.IS_UNORDERED
    fp['dayofweek'] = flags.IS_ORDERED
    fp['WEEK_OF_YEAR'] = flags.IS_CONTINUOUS
    fp['dayofmonth'] = flags.IS_CONTINUOUS
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
    fp['td'] = flags.IS_CONTINUOUS | flags.IS_LINEAR
    return fp


def cb_mean_model():
    fp = feature_properties()
    explicit_smoothers = {}

    features = [
        'dayofweek',
        'L_ID',
        'PG_ID_1',
        'PG_ID_2',
        'PG_ID_3',
        'P_ID',
        'WEEK_OF_YEAR',
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


def train_RF(X, y):
    features = [
        'L_ID', 'PG_ID_1', 'PG_ID_2', 'PG_ID_3', 'P_ID',
        'NORMAL_PRICE',
        'SALES_AREA',
        'SCHOOL_HOLIDAY',
        'dayofweek',
        'WEEK_OF_YEAR',
        'dayofmonth',
        'td',
        'Christmas',
        'Easter',
        'Labour_Day',
        'German_Unity',
        'Other_Holiday',
        'Local_Holiday_0',
        'Local_Holiday_1',
        'Local_Holiday_2',
    ]

    X = X[features]

    ml_est = RandomForestClassifier(max_depth=8, random_state=0, n_jobs=5)
    ml_est.fit(X, y)

    feature_names = X.columns.to_list()
    plot_feature_importances(ml_est, feature_names)

    del X
    return ml_est


def train_CB(X, y):
    ml_est = cb_mean_model()
    ml_est.fit(X, y)

    plot_CB('analysis_CB_promo_mean_iterlast', [ml_est[-1].observers[-1]], ml_est[-2])

    del X
    return ml_est


def plot_feature_importances(forest, feature_names):
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig('plots/feature_importances_promo.pdf')
    plt.clf()


def calculate_ace(df):
    print('naive:')
    ace = df.loc[df['PROMOTION_TYPE'] > 0, 'SALES'].mean() - df.loc[df['PROMOTION_TYPE'] == 0, 'SALES'].mean()
    print("absolute average causal effect of promotions: {}".format(round(ace, 3)))
    print("relative average causal effect of promotions: {}%".format(round(ace / df['SALES'].mean() * 100, 1)))

    print('adjusted:')
    ace = 0

    # for i in range(1, 54):
    #     mask = (df['WEEK_OF_YEAR'] == i)
    #     ace_week = df.loc[(df['PROMOTION_TYPE'] > 0) & mask, 'SALES'].mean() - df.loc[(df['PROMOTION_TYPE'] == 0) & mask, 'SALES'].mean()
    #     ratio_week =  len(df.loc[mask]) / len(df)
    #     ace += ace_week * ratio_week
    # for i in range(1, 13):
    #     mask = (df['month'] == i)
    #     ace_week = df.loc[(df['PROMOTION_TYPE'] > 0) & mask, 'SALES'].mean() - df.loc[(df['PROMOTION_TYPE'] == 0) & mask, 'SALES'].mean()
    #     ratio_week =  len(df.loc[mask]) / len(df)
    #     ace += ace_week * ratio_week

    # for j in range(1, 21):
    #     mask = (df['PG_ID_3'] == j)
    #     ace_week = df.loc[(df['PROMOTION_TYPE'] > 0) & mask, 'SALES'].mean() - df.loc[(df['PROMOTION_TYPE'] == 0) & mask, 'SALES'].mean()
    #     ratio_week =  len(df.loc[mask]) / len(df)
    #     ace += ace_week * ratio_week
    # for j in range(1, 4):
    #     mask = (df['PG_ID_2'] == j)
    #     ace_week = df.loc[(df['PROMOTION_TYPE'] > 0) & mask, 'SALES'].mean() - df.loc[(df['PROMOTION_TYPE'] == 0) & mask, 'SALES'].mean()
    #     ratio_week =  len(df.loc[mask]) / len(df)
    #     ace += ace_week * ratio_week

    for i in range(1, 13):
        for j in range(1, 4):
            mask = (df['month'] == i) & (df['PG_ID_2'] == j)
            ace_week = df.loc[(df['PROMOTION_TYPE'] > 0) & mask, 'SALES'].mean() - df.loc[(df['PROMOTION_TYPE'] == 0) & mask, 'SALES'].mean()
            ratio_week = len(df.loc[mask]) / len(df)
            ace += ace_week * ratio_week

    print("absolute average causal effect of promotions: {}".format(round(ace, 3)))
    print("relative average causal effect of promotions: {}%".format(round(ace / df['SALES'].mean() * 100, 1)))

    # df.groupby('PG_ID_3')['SALES'].mean()
    # PG_ID_3
    # 1 5.699302
    # 2 2.794592
    # 3 5.824225
    # 4 2.854730
    # 5 4.750665
    # 6 8.416651
    # 7 6.583932
    # 8 2.452734
    # 9 4.648171
    # 10 5.207618
    # 11 6.969130
    # 12 2.440983
    # 13 3.870397
    # 14 3.679386
    # 15 3.662141
    # 16 3.655426
    # 17 1.037350
    # 18 2.433259
    # 19 2.263190
    # 20 1.154342
    #
    # df.groupby('PG_ID_3')['PROMOTION_TYPE'].mean()
    # PG_ID_3
    # 1 0.049668
    # 2 0.071654
    # 3 0.093861
    # 4 0.068372
    # 5 0.072862
    # 6 0.084582
    # 7 0.080694
    # 8 0.105846
    # 9 0.087133
    # 10 0.112163
    # 11 0.116350
    # 12 0.138293
    # 13 0.147666
    # 14 0.132169
    # 15 0.118116
    # 16 0.129253
    # 17 0.147711
    # 18 0.135347
    # 19 0.163675
    # 20 0.155213


def main(args):
    df_train = pd.read_parquet("../../train.parquet.gzip")
    df_test = pd.read_parquet("../../test.parquet.gzip")
    df_test_results = pd.read_parquet("../../test_results.parquet.gzip")
    df_test = df_test.merge(df_test_results, how='inner', on=['P_ID', 'L_ID', 'DATE'])
    df = pd.concat([df_train, df_test], ignore_index=True)

    X, y = prepare_data(df)

    calculate_ace(X)

    # rest is just for identifying confounders
    _ = train_RF(X.copy(), y)
    _ = train_CB(X.copy(), y)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
