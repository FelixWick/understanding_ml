import sys

import pandas as pd
import numpy as np
import datetime

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

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


def prepare_data(df):
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['dayofweek'] = df['DATE'].dt.dayofweek
    df['dayofyear'] = df['DATE'].dt.dayofyear
    df['month'] = df['DATE'].dt.month
    df['dayofmonth'] = df['DATE'].dt.day

    df['price_ratio'] = df['SALES_PRICE'] / df['NORMAL_PRICE']
    df['price_ratio'].fillna(1, inplace=True)
    df['price_ratio'].clip(0, 1, inplace=True)

    df = get_events(df)
    df.fillna(66, inplace=True)

    # add time index
    df['time_idx'] = (df['DATE'] - df['DATE'].min()).dt.days

    # add additional features
    df["avg_sales_by_item"] = df.groupby(["time_idx", "P_ID"], observed=True).SALES.transform("mean")
    df["avg_sales_by_store"] = df.groupby(["time_idx", "L_ID"], observed=True).SALES.transform("mean")

    df['P_ID'] = df['P_ID'].astype(str)
    df['L_ID'] = df['L_ID'].astype(str)
    df['dayofweek'] = df['dayofweek'].astype(str)
    df['dayofyear'] = df['dayofyear'].astype(str)
    df['dayofmonth'] = df['dayofmonth'].astype(str)

    df['SALES'] = df['SALES'].astype(float)

    return df


def main(args):
    df_train = pd.read_parquet("../../train.parquet.gzip")
    df_test = pd.read_parquet("../../test.parquet.gzip")
    df_test_results = pd.read_parquet("../../test_results.parquet.gzip")
    df_test = df_test.merge(df_test_results, how='inner', on=['P_ID', 'L_ID', 'DATE'])
    df = pd.concat([df_train, df_test], ignore_index=True)

    df = prepare_data(df)

    X_train = df.loc[df['DATE']<='2022-03-31']
    X_test = df.loc[df['DATE']>'2022-03-31']

    # max_prediction_length = 28
    max_prediction_length = 132
    max_encoder_length = 90
    # training_cutoff = X_train["time_idx"].max() - max_prediction_length
    training_cutoff = X_train["time_idx"].max()

    training = TimeSeriesDataSet(
        X_train[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="SALES",
        group_ids=["P_ID", "L_ID"],
        min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["P_ID", "L_ID"],
        static_reals=[],
        time_varying_known_categoricals=[
            "dayofweek", "dayofyear", "dayofmonth"
        ],
        #        variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
        time_varying_known_reals=["time_idx", "NORMAL_PRICE", "price_ratio"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "SALES",
            "avg_sales_by_store",
            "avg_sales_by_item",
        ],
        target_normalizer=GroupNormalizer(
            groups=["L_ID", "P_ID"], transformation="softplus"
        ),  # use softplus and normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
        #        categorical_encoders={name_of_the_column: NaNLabelEncoder(add_nan=True)},
    )

    # create validation set (predict=True) which means to predict the last max_prediction_length points in time for each series
    validation = TimeSeriesDataSet.from_dataset(training, X_test, predict=True, stop_randomization=True)

    # create dataloaders for model
    batch_size = 128  # set this between 32 to 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    # configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    #    logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard
    logger = CSVLogger("lightning_logs")

    trainer = pl.Trainer(
        max_epochs=30,
        gpus=0,
        gradient_clip_val=0.1,
        limit_train_batches=30,  # coment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that network or dataset has no serious bugs
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,  # 7 quantiles by default
        loss=QuantileLoss(),
        #        log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

    # fit network
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # load the best model according to the validation loss
    # (given that we use early stopping, this is not necessarily the last epoch)
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # calcualte mean absolute error on validation set
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    predictions = best_tft.predict(val_dataloader)
    mad = (actuals - predictions).abs().mean()
    print('MAD: {}'.format(mad))

    # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
    raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)

    #    for idx in range(10):  # plot 10 examples
    for idx in [9, 16, 17, 41, 50]:
        best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True).savefig('plots/ts_' + str(idx) + '.pdf')

    interpretation = best_tft.interpret_output(raw_predictions, reduction="sum")
    interpretation_dict = best_tft.plot_interpretation(interpretation)
    interpretation_dict['attention'].savefig('plots/attention.pdf')
    interpretation_dict['encoder_variables'].savefig('plots/encoder_variables.pdf')
    interpretation_dict['decoder_variables'].savefig('plots/decoder_variables.pdf')

    eval_results(predictions.numpy().flatten(), actuals.numpy().flatten())

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
