import pandas as pd
import random
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, roc_curve, auc
import numpy as np
from sklearn import ensemble
import itertools
import xgboost as xgb
from sklearn import tree
from subprocess import check_call
# import pydotplus
import codecs
import itertools
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号#有中文出现的情况，需要u'内容'

dict_name = {1: '电池容量', 2: '是否安装雷达', 3: '处理器运行速度', 4: '有无后置影像', 5: '传感器像素',
              6: '是否支持5G', 7: '存储空间', 8: '移动深度', 9: '重量', 10: '处理器核心数', 11: '主要相机百万像素',
             12: '像素分辨率高度', 13: '像素分辨率宽度', 14: '以兆字节为单位的随机存取存储器',
             15: '屏幕高度（以cm为单位）', 16: '屏幕宽度（以cm为单位）', 17: '充电时间', 18: '是否有4G',
             19: '是否有触摸屏', 20: '是否有wifi'}


def plot_weight(acc_train, recall_train, f1_train, acc_test, recall_test, f1_test):

    # weight_x = [i for i in np.arange(0.01, 0.3, 0.01)]
    weight_x = [i for i in np.arange(0.05, 0.8, 0.05)]
    plt.plot(weight_x, acc_train, color='r', linestyle=':', marker='o', markerfacecolor='r', markersize=5,
             label='precision_train')
    plt.plot(weight_x, recall_train, color='g', linestyle='-', marker='v', markerfacecolor='g', markersize=5,
             label='recall_train')
    plt.plot(weight_x, f1_train, color='b', linestyle='-.', marker='^', markerfacecolor='b', markersize=5,
             label='f1_train')
    # plt.plot(weight_x, acc_test, color='y', linestyle=':', marker='p', markerfacecolor='y', markersize=5,
    #          label='precision_test')
    # plt.plot(weight_x, recall_test, color='k', linestyle='-', marker='o', markerfacecolor='k', markersize=5,
    #          label='recall_test')
    # plt.plot(weight_x, f1_test, color='c', linestyle='-.', marker='v', markerfacecolor='c', markersize=5,
    #          label='f1_test')
    plt.xticks(np.arange(0.05, 0.8, 0.05), fontsize=12.5)
    plt.xlim((0.05, 0.8))
    plt.yticks(np.arange(0.86, 1.02, 0.02), fontsize=12.5)
    plt.ylim((0.86, 1.02))
    plt.grid(True)
    plt.legend(loc=1, fontsize=12.5)
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 12.5}
    plt.xlabel('learning rate', font2)
    plt.ylabel('evaluation', font2)
    plt.show()


def data_split(raw_data):
    a = [i for i in range(len(raw_data))]
    random.shuffle(a)
    train_data = raw_data.iloc[a[0:int(0.9*len(a))]]
    test_data = raw_data.iloc[a[int(0.9*len(a)):]]
    train_data.to_csv(r'../../../train.csv', encoding='utf_8_sig', index=False)
    test_data.to_csv(r'../../../test.csv', encoding='utf_8_sig', index=False)


def DT(data_train, label_train, data_test, label_test):
    acc_train = []
    recall_train = []
    f1_train = []
    aucs_train = []
    acc_test = []
    recall_test = []
    f1_test = []
    aucs_test = []

    # for i in range(10, 101, 10):
    for temp in range(10, 40, 5):
        clf = tree.DecisionTreeClassifier(max_depth=temp, min_impurity_split=0.1)
        decision_tree = clf.fit(data_train, label_train)
        train_predict = decision_tree.predict(data_train)

        acc_train.append(accuracy_score(label_train, train_predict))
        recall_train.append(recall_score(label_train, train_predict, average='weighted'))
        f1_train.append(f1_score(label_train, train_predict, average='weighted'))

        test_predict = decision_tree.predict(data_test)
        test_predict_prob = decision_tree.predict_proba(data_test)
        acc_test.append(accuracy_score(label_test, test_predict))
        recall_test.append(recall_score(label_test, test_predict, average='weighted'))
        f1_test.append(f1_score(label_test, test_predict, average='weighted'))
        print(test_predict_prob)
        one_hot = []
        for i in label_test:
            temp = [0] * 4
            temp[i] = 1
            one_hot += temp
        fpr, tpr, _ = roc_curve(one_hot, list(itertools.chain.from_iterable(test_predict_prob)))
        aucs_test.append(auc(fpr, tpr))

    plot_weight(acc_train, recall_train, f1_train, acc_test, recall_test, f1_test)

    print("Train accuracy: %.4f" % (sum(acc_train) / 10))
    print("Train recall: %.4f" % (sum(recall_train) / 10))
    print("Train F1: %.4f" % (sum(f1_train) / 10))
    print("Test accuracy: %.4f" % (sum(acc_test) / 10))
    print("Test recall: %.4f" % (sum(recall_test) / 10))
    print("Test F1: %.4f" % (sum(f1_test) / 10))
    print("Test AUC Score: %.4f" % (sum(aucs_test) / 10))
    return decision_tree


def decision_tree_show(decision_tree, name_list):
    class_temp = [str(i) for i in range(4)]
    dot_data = tree.export_graphviz(decision_tree, out_file=None,
                                    feature_names=name_list,
                                    class_names=class_temp,
                                    filled=True, rounded=True)

    print(type(dot_data))

    with open('./dot_data.dot', 'w', encoding='utf-8') as f:
        class_temp = [str(i) for i in range(18)]
        dot_data = tree.export_graphviz(decision_tree, out_file=f,
                                        feature_names=name_list,
                                        class_names=class_temp,
                                        filled=True, rounded=True)

    check_call([r'G:\py_env\Lib\site-packages\graphviz-2.38\release\bin\dot', '-v', '-Tpng', './dot_data.dot', '-o', 'test.png'])


def RF(data_train, label_train, data_test, label_test):
    acc_train = []
    recall_train = []
    f1_train = []
    aucs_train = []
    acc_test = []
    recall_test = []
    f1_test = []
    aucs_test = []
    for iTrees in range(10, 310, 20):
    # for iTrees in range(250, 251):
        # iTrees = 100
        # depth = 50
        # 训练
        RFModel = ensemble.RandomForestClassifier(n_estimators=iTrees, min_impurity_split=0.1, n_jobs=-1, oob_score=False, random_state=531)
        RFModel.fit(data_train, label_train)

        # Accumulate auc on test set
        train_predict = RFModel.predict(data_train)

        print("Train accuracy: %.4f" % accuracy_score(label_train, train_predict))
        acc_train.append(accuracy_score(label_train, train_predict))
        print("Train recall: %.4f" % recall_score(label_train, train_predict, average='weighted'))
        recall_train.append(recall_score(label_train, train_predict, average='weighted'))
        print("Train F1: %.4f" % f1_score(label_train, train_predict, average='weighted'))
        f1_train.append(f1_score(label_train, train_predict, average='weighted'))
        test_predict = RFModel.predict(data_test)

        print("Test accuracy: %.4f" % accuracy_score(label_test, test_predict))
        acc_test.append(accuracy_score(label_test, test_predict))
        print("Test recall: %.4f" % recall_score(label_test, test_predict, average='weighted'))
        recall_test.append(recall_score(label_test, test_predict, average='weighted'))
        print("Test F1: %.4f" % f1_score(label_test, test_predict, average='weighted'))
        f1_test.append(f1_score(label_test, test_predict, average='weighted'))
        confusion_mat = confusion_matrix(label_test, test_predict)

        test_predict_prob = RFModel.predict_proba(data_test)
        one_hot = []
        for i in label_test:
            temp = [0] * 4
            temp[i] = 1
            one_hot += temp
        fpr, tpr, _ = roc_curve(one_hot, list(itertools.chain.from_iterable(test_predict_prob)))
        print("Auc score: %.4f" %auc(fpr, tpr))

        # pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(r'../results/classification/RF_data_enhance.csv',
        #                                               encoding='utf_8_sig', index=False)

        # print("Test confusion matrix:")
        # print(confusion_mat)
    plot_weight(acc_train, recall_train, f1_train, acc_test, recall_test, f1_test)


def XGB(data_train, label_train, data_test, label_test):
    acc_train = []
    recall_train = []
    f1_train = []
    aucs_train = []
    acc_test = []
    recall_test = []
    f1_test = []
    aucs_test = []
    # for estimator_num in range(50, 500, 20):
    for lr in np.arange(0.05, 0.8, 0.05):
    # for lr in np.arange(0.15, 0.16):
        estimator_num = 550
        m_class = xgb.XGBClassifier(learning_rate=lr, n_estimators=estimator_num, objective='multi:softmax', num_class=4)
        # m_class = xgb.XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=50, min_child_weight=6, gamma=0,
        #                             subsample=0.5, n_jobs=-1, reg_alpha=0.05, reg_lambda=0.05,
        #                             colsample_bytree=0.8, objective='multi:softmax', num_class=4, seed=127)
        # 训练
        decision_tree = m_class.fit(data_train, label_train)
        train_predict = m_class.predict(data_train)

        print("Train accuracy: %.4f" % accuracy_score(label_train, train_predict))
        acc_train.append(accuracy_score(label_train, train_predict))
        print("Train recall: %.4f" % recall_score(label_train, train_predict, average='weighted'))
        recall_train.append(recall_score(label_train, train_predict, average='weighted'))
        print("Train F1: %.4f" % f1_score(label_train, train_predict, average='weighted'))
        f1_train.append(f1_score(label_train, train_predict, average='weighted'))

        test_predict = m_class.predict(data_test)
        print("Test accuracy: %.4f" % accuracy_score(label_test, test_predict))
        acc_test.append(accuracy_score(label_test, test_predict))
        print("Test recall: %.4f" % recall_score(label_test, test_predict, average='weighted'))
        recall_test.append(recall_score(label_test, test_predict, average='weighted'))
        print("Test F1: %.4f" % f1_score(label_test, test_predict, average='weighted'))
        f1_test.append(f1_score(label_test, test_predict, average='weighted'))

        test_predict_prob = m_class.predict_proba(data_test)
        one_hot = []
        for i in label_test:
            temp = [0] * 4
            temp[i] = 1
            one_hot += temp
        fpr, tpr, _ = roc_curve(one_hot, list(itertools.chain.from_iterable(test_predict_prob)))
        print("Test auc score: %.4f" % auc(fpr, tpr))
        # pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(r'../results/classification/XGB_data_enhance.csv',
        #                                               encoding='utf_8_sig', index=False)

        # confusion_mat = confusion_matrix(label_test, test_predict)
        # print("Test confusion matrix:")
        # print(confusion_mat)
    plot_weight(acc_train, recall_train, f1_train, acc_test, recall_test, f1_test)


def label_creat(list_level):
    labels = []
    for i in list_level:
        if i == "A":
            labels.append(0)
        elif i == "B":
            labels.append(1)
        elif i == "C":
            labels.append(2)
        elif i == "D":
            labels.append(3)
    return labels


def creat_level(lists):
    labels = []
    for i in lists:
        if i == 0:
            labels.append('A')
        elif i == 1:
            labels.append('B')
        elif i == 2:
            labels.append('C')
        elif i == 3:
            labels.append('D')
    return labels


def main():
    # path_data = r'../../../index.csv'
    # data_label = pd.read_csv(path_data, encoding='utf_8_sig')
    # data_split(data_label)
    train_data = pd.read_csv(r'../../../train.csv', encoding='utf_8_sig')
    test_data = pd.read_csv(r'../../../test.csv', encoding='utf_8_sig')
    index_temp = ['销售规模（对数）', '销项开票数量（对数）', '平均销项开票金额（对数）', '进购规模（对数）', '进项开票数量（对数）',
     '税价比', '发票平均税率', '下游客户数量', '指定周期负数发票金额比率', '指定周期作废发票金额比率', '固定周期下游客户数量',
    '销方基尼系数', '销方HIHI指数']
    train_sp = train_data[index_temp]
    # test_sp = test_data[index_temp]
    train_label = label_creat(list(train_data['信誉评级']))
    # test_label = label_creat(list(test_data['信誉评级']))
    # dt = DT(train_sp, train_label, test_sp, test_label)
    # decision_tree_show(dt, list(dict_name.keys()))
    # RF(train_sp, train_label, test_sp, test_label)
    # XGB(train_sp, train_label, test_sp, test_label)

    # train_data = pd.read_excel(r'../../problem  C  附件1.xlsx')
    test_data = pd.read_csv(r'../../../index_2.csv', encoding='utf_8_sig')
    # train_sp = train_data[[i for i in range(1, 21)]]
    test_sp = test_data[index_temp]
    # train_label = list(train_data['编队'])
    m_class = xgb.XGBClassifier(learning_rate=0.15, n_estimators=350, objective='multi:softmax', num_class=4)
    m_class.fit(train_sp, train_label)
    test_predict = m_class.predict(test_sp)
    train_predict = m_class.predict(train_sp)
    print("Train F1: %.4f" % f1_score(train_label, train_predict, average='weighted'))
    test_data['信誉评级'] = creat_level(test_predict)
    test_data.to_csv(r'../../../附件二企业信誉评级识别结果.csv', encoding='utf_8_sig', index=False)


if __name__ == '__main__':
    main()





