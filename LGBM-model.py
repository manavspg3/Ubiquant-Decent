import os
import gc
import joblib
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from argparse import Namespace
from collections import defaultdict

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, train_test_split


import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')
pd.set_option('max_columns', 64)

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
    
    args = Namespace(
    debug=False,
    seed=21,
    folds=5,
    workers=4,
    min_time_id=None, 
    holdout=True,
    num_bins=16,
    data_path=Path("../input/ubiquant-parquet/"),
)
seed_everything(args.seed)

if args.debug:
    setattr(args, 'min_time_id', 1100)
    
%%time
train = pd.read_parquet(args.data_path.joinpath("train_low_mem.parquet"))
assert train.isnull().any().sum() == 0, "null exists."
assert train.row_id.str.extract(r"(?P<time_id>\d+)_(?P<investment_id>\d+)").astype(train.time_id.dtype).equals(train[["time_id", "investment_id"]]), "row_id!=time_id_investment_id"

if args.min_time_id is not None:
    train = train.query("time_id>=@args.min_time_id").reset_index(drop=True)
    gc.collect()
train.shape

time_id_df = (
    train.filter(regex=r"^(?!f_).*")
    .groupby("investment_id")
    .agg({"time_id": ["min", "max"]})
    .reset_index()
)
time_id_df["time_span"] = time_id_df["time_id"].diff(axis=1)["max"]
time_id_df.head(6)

train = train.merge(time_id_df.drop(columns="time_id").droplevel(level=1, axis=1), on="investment_id")
train.time_span.hist(bins=args.num_bins, figsize=(16,8)
del time_id_df
gc.collect()

if args.holdout:
    _target = pd.cut(train.time_span, args.num_bins, labels=False)
    _train, _valid = train_test_split(_target, stratify=_target)
    print(f"train length: {len(_train)}", f"holdout length: {len(_valid)}")
    valid = train.iloc[_valid.index].sort_values(by=["investment_id", "time_id"]).reset_index(drop=True)
    train = train.iloc[_train.index].sort_values(by=["investment_id", "time_id"]).reset_index(drop=True)
    train.time_span.hist(bins=args.num_bins, figsize=(16,8), alpha=0.8)
    valid.time_span.hist(bins=args.num_bins, figsize=(16,8), alpha=0.8)
    valid.drop(columns="time_span").to_parquet("valid.parquet")
    del valid, _train, _valid, _target
    gc.collect()
    
train["fold"] = -1
_target = pd.cut(train.time_span, args.num_bins, labels=False)
skf = StratifiedKFold(n_splits=args.folds)
for fold, (train_index, valid_index) in enumerate(skf.split(_target, _target)):
    train.loc[valid_index, 'fold'] = fold
    
fig, axs = plt.subplots(nrows=args.folds, ncols=1, sharex=True, figsize=(16,8), tight_layout=True)
for ax, (fold, df) in zip(axs, train[["fold", "time_span"]].groupby("fold")):
    ax.hist(df.time_span, bins=args.num_bins)
    ax.text(0, 40000, f"fold: {fold}, count: {len(df)}", fontsize=16)
plt.show()
del _target, train_index, valid_index
_=gc.collect()

cat_features = ["investment_id"]
num_features = list(train.filter(like="f_").columns)
features = num_features + cat_features

train = reduce_mem_usage(train.drop(columns="time_span"))
train[["investment_id", "time_id"]] = train[["investment_id", "time_id"]].astype(np.uint16)
train["fold"] = train["fold"].astype(np.uint8)
gc.collect()
features += ["time_id"] 
len(features)

corr_matrix = train.filter(like="f_").corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# # Find features with correlation greater than 0.9745656
to_drop = [column for column in upper.columns if any(upper[column] >= 0.9745656)]
sorted(to_drop)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def feval_rmse(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'rmse', rmse(y_true, y_pred), False

def feval_pearsonr(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'pearsonr', pearsonr(y_true, y_pred)[0], True

def run():    
    params = {
        'learning_rate':0.5,
        "objective": "regression",
        "metric": "rmse",
        'boosting_type': "gbdt",
        'verbosity': -1,
        'n_jobs': -1, 
        'seed': args.seed,
        'lambda_l1': 2.7223413643193285e-08, 
        'lambda_l2': 0.009462714717237544, 
        'num_leaves': 108, 
        'feature_fraction': 0.5298125662824026, 
        'bagging_fraction': 0.7279540797730281, 
        'bagging_freq': 9, 
        'max_depth': 40, 
        'max_bin': 487, 
        'min_data_in_leaf': 158,
        'n_estimators': 1000, 
    }
    
    y = train['target']
    train['preds'] = -1000
    scores = defaultdict(list)
    features_importance= pd.DataFrame()
    
    for fold in range(args.folds):
        print(f"=====================fold: {fold}=====================")
        trn_ind, val_ind = train.fold!=fold, train.fold==fold
        print(f"train length: {trn_ind.sum()}, valid length: {val_ind.sum()}")
        train_dataset = lgb.Dataset(train.loc[trn_ind, features], y.loc[trn_ind], categorical_feature=cat_features)
        valid_dataset = lgb.Dataset(train.loc[val_ind, features], y.loc[val_ind], categorical_feature=cat_features)

        model = lgb.train(
            params,
            train_set = train_dataset, 
            valid_sets = [train_dataset, valid_dataset], 
            verbose_eval=100,
            early_stopping_rounds=50,
            feval = feval_pearsonr
        )
        joblib.dump(model, f'lgbm_seed{args.seed}_{fold}.pkl')

        preds = model.predict(train.loc[val_ind, features])
        train.loc[val_ind, "preds"] = preds
        
        scores["rmse"].append(rmse(y.loc[val_ind], preds))
        scores["pearsonr"].append(pearsonr(y.loc[val_ind], preds)[0])
        
        fold_importance_df= pd.DataFrame({'feature': features, 'importance': model.feature_importance(), 'fold': fold})
        features_importance = pd.concat([features_importance, fold_importance_df], axis=0)
        
        del train_dataset, valid_dataset, model
        gc.collect()
    print(f"lgbm {args.folds} folds mean rmse: {np.mean(scores['rmse'])}, mean pearsonr: {np.mean(scores['pearsonr'])}")
    train.filter(regex=r"^(?!f_).*").to_csv("preds.csv", index=False)
    return features_importance
    
features_importance = run()
df = train[["target", "preds"]].query("preds!=-1000")
print(f"lgbm {args.folds} folds mean rmse: {rmse(df.target, df.preds)}, mean pearsonr: {pearsonr(df.target, df.preds)[0]}")
del df, train
gc.collect()

import seaborn as sns
import matplotlib.pyplot as plt

folds_mean_importance = (
    features_importance.groupby("feature")
    .importance.mean()
    .reset_index()
    .sort_values(by="importance", ascending=False)
)
features_importance.to_csv("features_importance.csv", index=False)
folds_mean_importance.to_csv("folds_mean_feature_importance.csv", index=False)

plt.figure(figsize=(16, 10))
plt.subplot(1,2,1)
sns.barplot(x="importance", y="feature", data=folds_mean_importance.head(50))
plt.title(f'Head LightGBM Features (avg over {args.folds} folds)')
plt.subplot(1,2,2)
sns.barplot(x="importance", y="feature", data=folds_mean_importance.tail(50))
plt.title(f'Tail LightGBM Features (avg over {args.folds} folds)')
plt.tight_layout()
plt.show()

import ubiquant
env = ubiquant.make_env()  
iter_test = env.iter_test()

models = [joblib.load(f'lgbm_seed{args.seed}_{fold}.pkl') for fold in range(args.folds)]
if args.holdout:
    valid = pd.read_parquet("valid.parquet")
    valid_pred = np.mean(np.stack([models[fold].predict(valid[features]) for fold in range(args.folds)]), axis=0)
    print(f"lgbm {args.folds} folds holdout rmse: {rmse(valid.target, valid_pred)}, holdout pearsonr: {pearsonr(valid.target, valid_pred)[0]}")
    del valid, valid_pred
    gc.collect()

for (test_df, sample_prediction_df) in iter_test:
    test_df["time_id"] = test_df.row_id.str.extract(r"(\d+)_.*").astype(np.uint16) 
    final_pred = [models[fold].predict(test_df[features]) for fold in range(args.folds)]
    sample_prediction_df['target'] = np.mean(np.stack(final_pred), axis=0)
    env.predict(sample_prediction_df) 
    display(sample_prediction_df)
