import sys
import math

import pandas as pd
import numpy as np
import datetime

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from IPython import embed


def plot_t_sne(X, suffix, red=[], yellow=[], black=[], blue=[], green=[]):
    plt.figure()
    for col in red:
        plt.scatter(X[col, 0], X[col, 1], c='r')
    for col in yellow:
        plt.scatter(X[col, 0], X[col, 1], c='y')
    for col in black:
        plt.scatter(X[col, 0], X[col, 1], c='k')
    for col in blue:
        plt.scatter(X[col, 0], X[col, 1], c='b')
    for col in green:
        plt.scatter(X[col, 0], X[col, 1], c='g')
    plt.tight_layout()
    plt.savefig('plots/t-SNE_{}.pdf'.format(suffix))


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
    df['dayofyear'] = df['DATE'].dt.dayofyear
    df['month'] = df['DATE'].dt.month
    df['dayofmonth'] = df['DATE'].dt.day

    df['price_ratio'] = df['SALES_PRICE'] / df['NORMAL_PRICE']
    df['price_ratio'].fillna(1, inplace=True)
    df['price_ratio'].clip(0, 1, inplace=True)

    df = get_events(df)
    df.fillna(-11, inplace=True)

    df['ewma'] = ewma_prediction(df, ['P_ID', 'L_ID', 'dayofweek', 'PROMOTION_TYPE'], 0.01)
    df['ewma'].fillna(df['SALES'].mean(), inplace=True)
    print('Finished ewma')

    normalized_features = [
        'SALES_AREA',
        'NORMAL_PRICE',
        'SALES_PRICE',
        'dayofweek',
        'dayofyear',
        'month',
        'dayofmonth',
        'Christmas',
        'Easter',
        'Labour_Day',
        'German_Unity',
        'Other_Holiday',
        'Local_Holiday_0',
        'Local_Holiday_1',
        'Local_Holiday_2'
    ]
    df[normalized_features] = MinMaxScaler().fit_transform(df[normalized_features])
    print('Finished scaling')

    enc = make_column_transformer(
        (OneHotEncoder(), ['L_ID', 'PG_ID_1', 'PG_ID_2', 'PG_ID_3']),
        remainder='passthrough'
    )
    df_onehot = pd.DataFrame(enc.fit_transform(df), columns=enc.get_feature_names_out())
    df_onehot = df_onehot.filter(regex='onehotencoder')

    enc = make_column_transformer(
        (OneHotEncoder(max_categories=50), ['P_ID']),
        remainder='passthrough'
    )
    df_onehot_pid = pd.DataFrame(enc.fit_transform(df), columns=enc.get_feature_names_out())
    df_onehot_pid = df_onehot_pid.filter(regex='onehotencoder')
    df_onehot = df_onehot.merge(df_onehot_pid, left_index=True, right_index=True)

    df = df.merge(df_onehot, left_index=True, right_index=True, how='left')
    print('Finished one-hot encoding')

    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    df_enc = enc.fit_transform(df[['P_ID', 'L_ID', 'PG_ID_3']])
    df['prod_enc'] = df_enc[: ,0]
    df['loc_enc'] = df_enc[:, 1]
    df['pg3_enc'] = df_enc[:, 2]
    print('Finished ordinal encoding')

    return df


def ewma_prediction(df, group_cols, alpha):
    df.sort_values(['DATE'], inplace=True)
    df_grouped = df.groupby(group_cols, group_keys=False)
    return df_grouped['SALES'].apply(lambda x: x.shift(1).ewm(alpha=alpha, ignore_na=True).mean())


class FF_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(113, 50),
            # nn.Linear(114, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, 30),
            nn.ReLU(),
            nn.BatchNorm1d(30),
            nn.Linear(30, 1)
        )

    def forward(self, X):
        return self.mlp(X)


class FF_NN_emb(nn.Module):
    def __init__(self):
        super().__init__()

        self.product_embedding = nn.Embedding(154, 20)
        self.location_embedding = nn.Embedding(20, 10)
        self.pg3_embedding = nn.Embedding(20, 10)

        self.mlp = nn.Sequential(
            nn.Linear(63, 50),
            # nn.Linear(64, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, 30),
            nn.ReLU(),
            nn.BatchNorm1d(30),
            nn.Linear(30, 1)
        )

    def forward(self, X):
        X_prod_embed = self.product_embedding(X[:, -3].type(torch.LongTensor))
        X_loc_embed = self.location_embedding(X[:, -2].type(torch.LongTensor))
        X_pg3_embed = self.pg3_embedding(X[:, -1].type(torch.LongTensor))
        X = torch.cat([X[:, :-3], X_prod_embed.squeeze(), X_loc_embed.squeeze(), X_pg3_embed.squeeze()], dim=1)
        return self.mlp(X)


def get_model(mode):
    if mode == 'embeddings':
        model = FF_NN_emb()
    else:
        model = FF_NN()
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

    return model


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
        df = prepare_data(df)
        df.to_csv("prepro_data.csv", index=False)
        print('Finished pre-processing')
        embed()
    else:
        mode = args[0]
        df = pd.read_csv("prepro_data.csv")

    if mode == 'embeddings':
        df.drop(columns=df.filter(regex='onehotencoder__P_ID_').columns.values, inplace=True)
        df.drop(columns=df.filter(regex='onehotencoder__L_ID_').columns.values, inplace=True)
        df.drop(columns=df.filter(regex='onehotencoder__PG_ID_3_').columns.values, inplace=True)
    else:
        df.drop(columns=['prod_enc', 'loc_enc', 'pg3_enc'], inplace=True)

    y = np.asarray(df['SALES'])
    X = df.drop(columns='SALES')

    X_train = X.loc[X['DATE'] <= '2022-03-31']
    y_train = y[X['DATE'] <= '2022-03-31']
    X_test = X.loc[X['DATE'] > '2022-03-31']
    y_test = y[X['DATE'] > '2022-03-31']

    # X_train.drop(columns=['DATE', 'EVENT', 'P_ID', 'PG_ID_3' ,'PG_ID_2' ,'PG_ID_1', 'L_ID'], inplace=True)
    # X_test.drop(columns=['DATE', 'EVENT', 'P_ID', 'PG_ID_3' ,'PG_ID_2' ,'PG_ID_1', 'L_ID'], inplace=True)
    X_train.drop(columns=['ewma', 'DATE', 'EVENT', 'P_ID' ,'PG_ID_3' ,'PG_ID_2' ,'PG_ID_1', 'L_ID'], inplace=True)
    X_test.drop(columns=['ewma', 'DATE', 'EVENT', 'P_ID' ,'PG_ID_3' ,'PG_ID_2' ,'PG_ID_1', 'L_ID'], inplace=True)

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    train_target = torch.tensor(y_train.astype(np.float32)).unsqueeze(1).to(device)
    train = torch.tensor(X_train.values.astype(np.float32)).to(device)
    test_target = torch.tensor(y_test.astype(np.float32)).unsqueeze(1).to(device)
    test = torch.tensor(X_test.values.astype(np.float32)).to(device)

    train_ds = TensorDataset(train, train_target)
    test_ds = TensorDataset(test, test_target)

    mini_batch_size = 1024
    # 2671227 samples
    # mini_batch_size = train_ds[:][0].shape[0] // 10
    train_dl, valid_dl = get_data(train_ds, test_ds, mini_batch_size)

    model, optimizer = get_model(mode)
    model = model.to(device)

    epochs = 20
    trained_model = fit(epochs, model, optimizer, train_dl, valid_dl)

    # in-sample
    train_preds = trained_model(train_ds[:][0]).cpu().detach().numpy().flatten()
    eval_results(train_preds, y_train)

    # out-of-sample
    test_preds = trained_model(test_ds[:][0]).cpu().detach().numpy().flatten()
    eval_results(test_preds, y_test)

    if mode == 'embeddings':
        X_train["SALES"] = y_train

        product_embeddings = model.product_embedding(test_ds[:][0][:, -3].type(torch.LongTensor).unique()).detach().numpy()
        product_t_sne = TSNE(learning_rate='auto').fit_transform(product_embeddings)
        sales_sorted = X_train.groupby("prod_enc")["SALES"].mean().sort_values().reset_index()
        plot_t_sne(product_t_sne, "products", red=sales_sorted[131:153].prod_enc.values.astype(int), yellow=sales_sorted[101:131].prod_enc.values.astype(int), black=sales_sorted[51:101].prod_enc.values.astype(int), blue=sales_sorted[21:51].prod_enc.values.astype(int), green=sales_sorted[0:21].prod_enc.values.astype(int))

        location_embeddings = model.location_embedding(test_ds[:][0][:, -2].type(torch.LongTensor).unique()).detach().numpy()
        location_t_sne = TSNE(learning_rate='auto', perplexity=5.).fit_transform(location_embeddings)
        sales_sorted = X_train.groupby("loc_enc")["SALES"].mean().sort_values().reset_index()
        plot_t_sne(location_t_sne, "locations", red=sales_sorted[17:20].loc_enc.values.astype(int), yellow=sales_sorted[13:17].loc_enc.values.astype(int), black=sales_sorted[7:13].loc_enc.values.astype(int), blue=sales_sorted[3:7].loc_enc.values.astype(int), green=sales_sorted[0:3].loc_enc.values.astype(int))

        pg3_embeddings = model.pg3_embedding(test_ds[:][0][:, -1].type(torch.LongTensor).unique()).detach().numpy()
        pg3_t_sne = TSNE(learning_rate='auto', perplexity=5.).fit_transform(pg3_embeddings)
        sales_sorted = X_train.groupby("pg3_enc")["SALES"].mean().sort_values().reset_index()
        plot_t_sne(pg3_t_sne, "pg3s", red=sales_sorted[17:20].pg3_enc.values.astype(int), yellow=sales_sorted[13:17].pg3_enc.values.astype(int), black=sales_sorted[7:13].pg3_enc.values.astype(int), blue=sales_sorted[3:7].pg3_enc.values.astype(int), green=sales_sorted[0:3].pg3_enc.values.astype(int))

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
