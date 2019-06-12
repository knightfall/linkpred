#12786442 Sazid Banna

from collections import Counter
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def _df(df):
    dataframe = df.copy()
    zero = Counter(dataframe.label.values)[0]
    un = Counter(dataframe.label.values)[1]
    n = zero - un
    dataframe['SC'] = dataframe['SC'].astype('category')
    dataframe['label'] = dataframe['label'].astype('category')
    dataframe = pd.get_dummies(dataframe, columns=['SC'])
    dataframe = dataframe.drop(
        dataframe[dataframe.label == 0].sample(n=n, random_state=1).index)
    return dataframe.sample(frac=1)


def get_x_y(df):
    X = df.drop(['label', 'nodes'], axis=1)
    y = df['label']
    return X, y


def main():
    train = _df(pd.read_csv('train.csv', sep=";", decimal="."))
    test = _df(pd.read_csv('test.csv', sep=";", decimal="."))

    x_train, y_train = get_x_y(train)
    x_test, y_test = get_x_y(test)
    # find best features for XGBoost
    # learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
    # train_results = []
    # test_results = []
    # for eta in learning_rates:
    #     model = GradientBoostingClassifier(learning_rate=eta)
    #     model.fit(x_train, y_train)
    #
    #     train_pred = model.predict(x_train)
    #
    #     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    #     roc_auc = auc(false_positive_rate, true_positive_rate)
    #     train_results.append(roc_auc)
    #
    #     y_pred = model.predict(x_test)
    #
    #     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    #     roc_auc = auc(false_positive_rate, true_positive_rate)
    #     test_results.append(roc_auc)
    # from matplotlib.legend_handler import HandlerLine2D
    #
    # line1, = plt.plot(learning_rates, train_results, 'b', label="Train AUC")
    # line2, = plt.plot(learning_rates, test_results, 'r', label="Test AUC")
    #
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    #
    # plt.ylabel('AUC score')
    # plt.xlabel('learning rate')
    # plt.show()
    # n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
    # train_results = []
    # test_results = []
    # for estimator in n_estimators:
    #     model = GradientBoostingClassifier(n_estimators=estimator)
    #     model.fit(x_train, y_train)
    #
    #     train_pred = model.predict(x_train)
    #
    #     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    #     roc_auc = auc(false_positive_rate, true_positive_rate)
    #     train_results.append(roc_auc)
    #
    #     y_pred = model.predict(x_test)
    #
    #     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    #     roc_auc = auc(false_positive_rate, true_positive_rate)
    #     test_results.append(roc_auc)
    # from matplotlib.legend_handler import HandlerLine2D
    #
    # line1, = plt.plot(n_estimators, train_results, 'b', label="Train AUC")
    # line2, = plt.plot(n_estimators, test_results, 'r', label="Test AUC")
    #
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    #
    # plt.ylabel('AUC score')
    # plt.xlabel('n_estimators')
    # plt.show()
    # max_depths = np.linspace(1, 32, 32, endpoint=True)
    # train_results = []
    # test_results = []
    # for max_depth in max_depths:
    #     model = GradientBoostingClassifier(max_depth=max_depth)
    #     model.fit(x_train, y_train)
    #
    #     train_pred = model.predict(x_train)
    #
    #     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    #     roc_auc = auc(false_positive_rate, true_positive_rate)
    #     train_results.append(roc_auc)
    #
    #     y_pred = model.predict(x_test)
    #
    #     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    #     roc_auc = auc(false_positive_rate, true_positive_rate)
    #     test_results.append(roc_auc)
    # from matplotlib.legend_handler import HandlerLine2D
    #
    # line1, = plt.plot(max_depths, train_results, 'b', label="Train AUC")
    # line2, = plt.plot(max_depths, test_results, 'r', label="Test AUC")
    #
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    #
    # plt.ylabel('AUC score')
    # plt.xlabel('Tree depth')
    # plt.show()
    #
    # min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
    # train_results = []
    # test_results = []
    # for min_samples_split in min_samples_splits:
    #     model = GradientBoostingClassifier(min_samples_split=min_samples_split)
    #     model.fit(x_train, y_train)
    #
    #     train_pred = model.predict(x_train)
    #
    #     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    #     roc_auc = auc(false_positive_rate, true_positive_rate)
    #     train_results.append(roc_auc)
    #
    #     y_pred = model.predict(x_test)
    #
    #     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    #     roc_auc = auc(false_positive_rate, true_positive_rate)
    #     test_results.append(roc_auc)
    # from matplotlib.legend_handler import HandlerLine2D
    #
    # line1, = plt.plot(min_samples_splits, train_results, 'b', label="Train AUC")
    # line2, = plt.plot(min_samples_splits, test_results, 'r', label="Test AUC")
    #
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    #
    # plt.ylabel('AUC score')
    # plt.xlabel('min samples split')
    # plt.show()
    #
    # min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
    # train_results = []
    # test_results = []
    # for min_samples_leaf in min_samples_leafs:
    #     model = GradientBoostingClassifier(min_samples_leaf=min_samples_leaf)
    #     model.fit(x_train, y_train)
    #
    #     train_pred = model.predict(x_train)
    #
    #     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    #     roc_auc = auc(false_positive_rate, true_positive_rate)
    #     train_results.append(roc_auc)
    #
    #     y_pred = model.predict(x_test)
    #
    #     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    #     roc_auc = auc(false_positive_rate, true_positive_rate)
    #     test_results.append(roc_auc)
    # from matplotlib.legend_handler import HandlerLine2D
    #
    # line1, = plt.plot(min_samples_leafs, train_results, 'b', label="Train AUC")
    # line2, = plt.plot(min_samples_leafs, test_results, 'r', label="Test AUC")
    #
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    #
    # plt.ylabel('AUC score')
    # plt.xlabel('min samples leaf')
    # plt.show()

    #############################################################################################
    # Prediction

    clf = GradientBoostingClassifier(criterion='friedman_mse', init=None,
                                     learning_rate=0.2, loss='deviance', max_depth=5, min_samples_leaf=0.20,
                                     min_samples_split=0.6,
                                     n_estimators=25)
    clf.fit(x_train, y_train)

    xgb_pred = clf.predict_proba(x_test)
    feature_importance = np.array([x_train.columns, clf.feature_importances_])

    # neural

    print('fin')

    importance = feature_importance[1]
    features = feature_importance[0]
    indices = np.argsort(feature_importance[1])[::-1]

    fig = plt.figure(figsize=(13, 8))
    plt.bar(range(feature_importance.shape[1]), height=importance[indices], align='center', color='r')
    plt.title('Features importance (XGBoost)')
    plt.xticks(range(feature_importance.shape[1]), features[indices])
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.ylim((0, 0.40))
    fig.savefig('feature_importance_xgb.png')
    plt.show()

    false_positive_xgb, true_positive_xgb, threshold = roc_curve(y_test, xgb_pred[:, 1])
    roc_xgb = auc(false_positive_xgb, true_positive_xgb)
    false_positive_ra, true_positive_ra, threshold = roc_curve(y_test, x_test['RA'].values)
    roc_ra = auc(false_positive_ra, true_positive_ra)
    false_positive_aa, true_positive_aa, threshold = roc_curve(y_test, x_test['AA'].values)
    roc_aa = auc(false_positive_aa, true_positive_aa)
    false_positive_jc, true_positive_jc, threshold = roc_curve(y_test, x_test['JC'].values)
    roc_jc = auc(false_positive_jc, true_positive_jc)
    false_positive_cn, true_positive_cn, threshold = roc_curve(y_test, x_test['CN'].values)
    roc_cn = auc(false_positive_cn, true_positive_cn)
    false_positive_ud, true_positive_ud, threshold = roc_curve(y_test, x_test['UD'].values)
    roc_ud = auc(false_positive_ud, true_positive_ud)
    false_positive_pa, true_positive_pa, threshold = roc_curve(y_test, x_test['PA'].values)
    roc_pa = auc(false_positive_pa, true_positive_pa)

    fig = plt.figure(figsize=(13, 8))
    plt.title('ROC Curves')
    plt.plot(false_positive_xgb, true_positive_xgb, 'r', label='XGBoost (AUC = %0.2f)' % roc_xgb)
    plt.plot(false_positive_ra, true_positive_ra, 'g', label='Resource Allocation (AUC = %0.2f)' % roc_ra)
    plt.plot(false_positive_aa, true_positive_aa, 'xkcd:aqua blue', label='Adamic-Adar Index (AUC = %0.2f)' % roc_aa)
    plt.plot(false_positive_jc, true_positive_jc, 'b', label='Jaccard Coefficient (AUC = %0.2f)' % roc_jc)
    plt.plot(false_positive_cn, true_positive_cn, 'y', label='Common Neighbors (AUC = %0.2f)' % roc_cn)
    plt.plot(false_positive_pa, true_positive_pa, '#4ee617', label='PA (AUC = %0.2f)' % roc_pa)
    plt.plot(false_positive_ud, true_positive_ud, '#ff8b38', label='UD (AUC = %0.2f)' % roc_ud)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random score (AUC = 0.50)')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='lower right')
    fig.savefig('roc_curves.png')
    plt.show()


main()
