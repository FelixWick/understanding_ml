import sys
import math

import pandas as pd
import numpy as np
import datetime

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

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
        'Easter'
    ]:
        df[event] = 0
        df.loc[df['EVENT'] == event, event] = 1

    for event in [
        'Labour_Day',
        'German_Unity'
    ]:
        df['event_type_1'] = 0
        df.loc[df['EVENT'] == event, 'event_type_1'] = 1

    for event in [
        'Local_Holiday_0',
        'Local_Holiday_1'
    ]:
        df['event_type_2'] = 0
        df.loc[df['EVENT'] == event, 'event_type_2'] = 1

    return df


def split_sequence(df, sales_scaler):
    df = df[['past_sales', 'dayofweek', 'price_ratio', 'Christmas', 'Easter', 'event_type_1', 'event_type_2', 'SALES']]

    df['past_sales'].fillna(df['SALES'].mean(), inplace=True)
    df[['past_sales']] = sales_scaler.transform(df[['past_sales']])

    df['dayofweek'].fillna(-1., inplace=True)
    df['price_ratio'].fillna(-1., inplace=True)
    df['Christmas'].fillna(-1., inplace=True)
    df['Easter'].fillna(-1., inplace=True)
    df['event_type_1'].fillna(-1., inplace=True)
    df['event_type_2'].fillna(-1., inplace=True)

    # split in sub-sequences (7 days)
    samples = list()
    length = 7
    # overlapping sequence samples, stride 1
    for i in range(0, len(df)-length, 1):
        sample = df[i:i+length].reset_index(drop=True)
        # drop whole sequence sample if nan in target
        # if sample['SALES'].notnull().values.all():
        if np.isfinite(sample['SALES'][6]):
            samples.append(sample)

    samples_concat = pd.DataFrame({0: [], 1: [], 2: [], 3: [], 4: [],5: [], 6: []})
    for df in samples:
        # transpose
        df = df.transpose()
        samples_concat = pd.concat([samples_concat, df])
    return samples_concat


def create_sequence_samples(df, date_from, date_upto, sales_scaler):
    # fill gaps by merge with full time range
    date_df = pd.DataFrame()
    # one week overlap needed
    date_df['DATE'] = pd.date_range(start=date_from, end=date_upto)
    df = date_df.merge(df, how='left')

    # shift for last day's sales
    df = df.sort_values('DATE')
    df['past_sales'] = df['SALES'].shift(1)

    return split_sequence(df, sales_scaler)


def prepare_data(df):
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['dayofweek'] = df['DATE'].dt.dayofweek

    df['price_ratio'] = df['SALES_PRICE'] / df['NORMAL_PRICE']
    df['price_ratio'].fillna(1, inplace=True)
    df['price_ratio'].clip(0, 1, inplace=True)

    df = get_events(df)

    df['past_sales'] = df['SALES']
    sales_scaler = MinMaxScaler().fit(df[['past_sales']])

    # use overlapping sequence samples
    df_train = df.groupby(['P_ID', 'L_ID'], group_keys=False).apply(create_sequence_samples, '09/25/2019', '03/31/2022', sales_scaler)
    df_test = df.groupby(['P_ID', 'L_ID'], group_keys=False).apply(create_sequence_samples, '03/25/2022', '09/30/2022', sales_scaler)

    # channels
    X1_train = df_train.loc[['past_sales']].reset_index(drop=True)
    X2_train = df_train.loc[['dayofweek']].reset_index(drop=True)
    X3_train = df_train.loc[['price_ratio']].reset_index(drop=True)
    X4_train = df_train.loc[['Christmas']].reset_index(drop=True)
    X5_train = df_train.loc[['Easter']].reset_index(drop=True)
    X6_train = df_train.loc[['event_type_1']].reset_index(drop=True)
    X7_train = df_train.loc[['event_type_2']].reset_index(drop=True)
    X1_test = df_test.loc[['past_sales']].reset_index(drop=True)
    X2_test = df_test.loc[['dayofweek']].reset_index(drop=True)
    X3_test = df_test.loc[['price_ratio']].reset_index(drop=True)
    X4_test = df_test.loc[['Christmas']].reset_index(drop=True)
    X5_test = df_test.loc[['Easter']].reset_index(drop=True)
    X6_test = df_test.loc[['event_type_1']].reset_index(drop=True)
    X7_test = df_test.loc[['event_type_2']].reset_index(drop=True)

    y_train = df_train.loc[['SALES']].reset_index(drop=True)
    y_test = df_test.loc[['SALES']].reset_index(drop=True)

    return X1_train, X2_train, X3_train, X4_train, X5_train, X6_train, X7_train, y_train, X1_test, X2_test, X3_test, X4_test, X5_test, X6_test, X7_test, y_test


class RNN(nn.Module):
    def __init__(self):
        super().__init__()

        # instead of (rows/sequence_samples, features, time_steps)
        # input = torch.randn(rows/sequence_samples, time_steps, features)
        # nn.LSTM(input features, output features, batch_first=True)

        self.lstm = nn.LSTM(7, 15, batch_first=True)

        # self.dense = nn.Linear(105, 7)
        self.dense = nn.Linear(105, 1)

    def forward(self, X):
        X, _ = self.lstm(X)
        X = torch.reshape(X, (X.size(0), -1))  # flatten the tensor
        return self.dense(X)


def get_model():
    model = RNN()
    # optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    optimizer = optim.Adam(model.parameters())
    return model, optimizer


def fit(epochs, model, optimizer, train_dl, valid_dl):
    loss_func = F.mse_loss

    # loop over epochs
    for epoch in range(epochs):
        model.train()

        # loop over mini-batches
        for X_mb, y_mb in train_dl:
            y_hat = model(X_mb)

            loss = loss_func(y_hat, y_mb)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        print('epoch {}, loss {}'.format(epoch, loss.item()))

        model.eval()

        with torch.no_grad():
            valid_loss = sum(loss_func(model(X_mb), y_mb) for X_mb, y_mb in valid_dl)

        print(epoch, valid_loss / len(valid_dl))

    print('Finished training')


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


def main(args):
    if args[0]=='prepro':
        df_train = pd.read_parquet("../../train.parquet.gzip")
        df_test = pd.read_parquet("../../test.parquet.gzip")
        df_test_results = pd.read_parquet("../../test_results.parquet.gzip")
        df_test = df_test.merge(df_test_results, how='inner', on=['P_ID', 'L_ID', 'DATE'])
        df = pd.concat([df_train, df_test], ignore_index=True)
        X1_train, X2_train, X3_train, X4_train, X5_train, X6_train, X7_train, y_train, X1_test, X2_test, X3_test, X4_test, X5_test, X6_test, X7_test, y_test = prepare_data(df)
        X1_train.to_csv("prepro_data_X1_train.csv", index=False)
        X2_train.to_csv("prepro_data_X2_train.csv", index=False)
        X3_train.to_csv("prepro_data_X3_train.csv", index=False)
        X4_train.to_csv("prepro_data_X4_train.csv", index=False)
        X5_train.to_csv("prepro_data_X5_train.csv", index=False)
        X6_train.to_csv("prepro_data_X6_train.csv", index=False)
        X7_train.to_csv("prepro_data_X7_train.csv", index=False)
        y_train.to_csv("prepro_data_y_train.csv", index=False)
        X1_test.to_csv("prepro_data_X1_test.csv", index=False)
        X2_test.to_csv("prepro_data_X2_test.csv", index=False)
        X3_test.to_csv("prepro_data_X3_test.csv", index=False)
        X4_test.to_csv("prepro_data_X4_test.csv", index=False)
        X5_test.to_csv("prepro_data_X5_test.csv", index=False)
        X6_test.to_csv("prepro_data_X6_test.csv", index=False)
        X7_test.to_csv("prepro_data_X7_test.csv", index=False)
        y_test.to_csv("prepro_data_y_test.csv", index=False)
        print('Finished pre-processing')
        embed()
    else:
        X1_train = pd.read_csv("prepro_data_X1_train.csv").to_numpy()
        X2_train = pd.read_csv("prepro_data_X2_train.csv").to_numpy()
        X3_train = pd.read_csv("prepro_data_X3_train.csv").to_numpy()
        X4_train = pd.read_csv("prepro_data_X4_train.csv").to_numpy()
        X5_train = pd.read_csv("prepro_data_X5_train.csv").to_numpy()
        X6_train = pd.read_csv("prepro_data_X6_train.csv").to_numpy()
        X7_train = pd.read_csv("prepro_data_X7_train.csv").to_numpy()
        y_train = pd.read_csv("prepro_data_y_train.csv").to_numpy()
        X1_test = pd.read_csv("prepro_data_X1_test.csv").to_numpy()
        X2_test = pd.read_csv("prepro_data_X2_test.csv").to_numpy()
        X3_test = pd.read_csv("prepro_data_X3_test.csv").to_numpy()
        X4_test = pd.read_csv("prepro_data_X4_test.csv").to_numpy()
        X5_test = pd.read_csv("prepro_data_X5_test.csv").to_numpy()
        X6_test = pd.read_csv("prepro_data_X6_test.csv").to_numpy()
        X7_test = pd.read_csv("prepro_data_X7_test.csv").to_numpy()
        y_test = pd.read_csv("prepro_data_y_test.csv").to_numpy()

    torch.manual_seed(3)

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    # train_target = torch.tensor(y_train.astype(np.float32))
    train_target = torch.tensor(y_train[:, 6].astype(np.float32)).unsqueeze(1).to(device)

    # test_target = torch.tensor(y_test.astype(np.float32))
    test_target = torch.tensor(y_test[:, 6].astype(np.float32)).unsqueeze(1).to(device)

    X_train = np.stack([X1_train, X2_train, X3_train, X4_train, X5_train, X6_train, X7_train], axis=2)
    train = torch.tensor(X_train.astype(np.float32)).to(device)

    X_test = np.stack([X1_test, X2_test, X3_test, X4_test, X5_test, X6_test, X7_test], axis=2)
    test = torch.tensor(X_test.astype(np.float32)).to(device)

    train_ds = TensorDataset(train, train_target)
    test_ds = TensorDataset(test, test_target)

    mini_batch_size = 1024
    train_dl, valid_dl = get_data(train_ds, test_ds, mini_batch_size)

    model, optimizer = get_model()
    model = model.to(device)

    epochs = 20
    fit(epochs, model, optimizer, train_dl, valid_dl)

    # in-sample
    train_preds = model(train_ds[:][0]).cpu().detach().numpy().flatten()
    # eval_results(train_preds, y_train.flatten())
    eval_results(train_preds, y_train[:, 6])

    # out-of-sample
    test_preds = model(test_ds[:][0]).cpu().detach().numpy().flatten()
    # eval_results(test_preds, y_test.flatten())
    eval_results(test_preds, y_test[:, 6])

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
