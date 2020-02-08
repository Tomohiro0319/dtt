import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sklearn

import time

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn.metrics

import gc
gc.collect()

print('データ読み込み中')
taxi_data_dummies = pd.read_csv('./bunsekiyo_data_dummied_distsum_20200201.gz')
print('データ読み込み完了')

print('説明変数と目的変数')
X = taxi_data_dummies[taxi_data_dummies.columns[taxi_data_dummies.columns != 'dist_scale']]
Y = taxi_data_dummies['dist_scale']
print(X.info())
X.drop('dist_sum', axis=1, inplace=True)
# del taxi_data_dummies
# gc.collect()
# print('drop na in X')
# X = X.drop(X.columns[np.isnan(X).any()], axis=1)

print('学習データとテストデータに分ける')
X_train, X_test, y_train, y_test = train_test_split(X,Y,random_state=0,test_size=0.1)

#!/usr/bin/env python
# -*- coding: utf-8 -*-



"""XGBoost で特徴量の重要度を可視化するサンプルコード"""

def main():

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'error',
        'scale_pos_weight':(len(y_train)-sum(y_train)) / sum(y_train)
    }

    # 学習時に用いる検証用データ
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    # 学習過程を記録するための辞書
    evals_result = {}
    bst = xgb.train(xgb_params,
                    dtrain,
                    num_boost_round=30,  # ラウンド数を増やしておく
                    evals=evals,
                    evals_result=evals_result
                    )

    y_pred_proba = bst.predict(dtest)
    y_pred = np.where(y_pred_proba > 0.5, 1, 0)
    acc = accuracy_score(y_test, y_pred)
    print('Accuracy:', acc)

    # 学習の課程を折れ線グラフとしてプロットする
    train_metric = evals_result['train']['error']
    plt.plot(train_metric, label='train error')
    eval_metric = evals_result['eval']['error']
    plt.plot(eval_metric, label='eval error')
    plt.grid()
    plt.legend()
    plt.xlabel('rounds')
    plt.ylabel('error')
    plt.show()


    # FPR, TPR(, しきい値) を算出
#     fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

    # ついでにAUCも
#     auc = metrics.auc(fpr, tpr)

    # ROC曲線をプロット
#     plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
#     plt.legend()
#     plt.title('ROC curve')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.grid(True)

#     print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
# 性能向上に寄与する度合いで重要度をプロットする
    _, ax = plt.subplots(figsize=(30, 30))
    xgb.plot_importance(bst,
                        ax=ax,
                        importance_type='gain',
                        show_values=True)
    plt.show()


if __name__ == '__main__':
    main()
