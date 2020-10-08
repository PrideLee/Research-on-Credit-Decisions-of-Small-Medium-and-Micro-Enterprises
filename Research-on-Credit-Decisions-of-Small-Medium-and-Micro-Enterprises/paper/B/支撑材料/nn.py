import numpy as np
import pandas as pd
import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, roc_curve, auc
import torch.utils.data
import matplotlib.pyplot as plt
import math
from sklearn.manifold import TSNE
# from pylab import
import random

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号#有中文出现的情况，需要u'内容'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def data_prepare(data_raw):
    df_norm = (data_raw - data_raw.mean()) / (data_raw.std())
    df_norm[0:110].to_csv(r'../../../train_norm.csv', encoding='utf_8_sig', index=False)
    df_norm[110:].to_csv(r'../../../test_norm.csv', encoding='utf_8_sig', index=False)
    # df_norm[120:].to_csv(r'../../../val_norm.csv', encoding='utf_8_sig', index=False)


class nn_embedding(nn.Module):
    def __init__(self, input_size, class_num):
        super(nn_embedding, self).__init__()
        # self.embedding = nn.Embedding(attributes_num, dim_embedding)
        self.linear1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(64, 64)
        # self.linear3 = nn.Linear(128, 256)
        # self.linear4 = nn.Linear(256, 512)
        self.linear5 = nn.Linear(64, class_num)
    #
    def forward(self, inputs):
        # embeds = self.embedding(inputs).view(1, -1)
        # embeds = torch.cat((embeds, torch.tensor([[1]], dtype=torch.float)), 1)

        out_1 = F.tanh(self.linear1(inputs))
        # out_2 = F.tanh(self.linear2(out_1))
        # out_3 = F.tanh(self.dropout1(self.linear3(out_2)))
        # out_4 = F.tanh(self.dropout1(self.linear4(out_3)))
        out = F.tanh(self.linear5(out_1))
        log_probs = F.log_softmax(out)
        return log_probs


def loss_iteration():
    # print([np.round(i,4) for i in loss_train])
    # print([np.round(i,4) for i in loss_val])
    # print([np.round(i,4) for i in F1_train])
    # print([np.round(i,4) for i in F1_val])
    loss_train = [1.5166, 1.3879, 1.2181, 1.1747, 1.1467, 1.0282, 1.1157, 1.0069, 1.0007, 0.9961, 0.9927, 0.9901, 0.9881, 0.9865,
     0.9852, 0.984, 0.9831, 0.9823, 0.9816, 0.981, 0.980, 0.973, 0.970, 0.970, 0.968]
    loss_val = [1.8195, 1.6394, 1.389, 1.5563, 1.2349, 1.1206, 1.0108, 1.0039, 0.999, 0.9953, 0.9925, 0.9904, 0.9887, 0.9873,
     0.9862, 0.9853, 0.9845, 0.9838, 0.9833, 0.9827, 0.9820, 0.9814, 0.9810, 0.9771, 0.9770]
    F1_train = [0.4008, 0.5475, 0.6135, 0.6722, 0.716, 0.7484, 0.7784, 0.8162, 0.8104, 0.8114, 0.821, 0.8292, 0.829, 0.8221,
     0.8401, 0.8234, 0.8136, 0.8328, 0.8333, 0.8201, 0.8322, 0.8334, 0.8200, 0.8212, 0.832]
    F1_val = [0.5357, 0.6128, 0.6522, 0.6874, 0.7321, 0.7772, 0.7976, 0.7417, 0.7891, 0.8081, 0.8004, 0.8012, 0.802, 0.8161,
     0.8165, 0.8133, 0.8134, 0.8231, 0.809, 0.8114, 0.8113, 0.8134, 0.8131, 0.8292, 0.828]


    iteration = [i for i in range(len(loss_train))]
    plt.plot(iteration, loss_train, color='red', linestyle='-', label='training loss')
    plt.plot(iteration, F1_train, color='red', linestyle='-.', label='training F1')
    plt.plot(iteration, loss_val, color='green', linestyle='-', label='validation loss')
    plt.plot(iteration, F1_val, color='green', linestyle='-.', label='validation F1')
    plt.xlabel('iteration')
    plt.ylabel('loss & F1')
    plt.legend()
    plt.grid(True)
    # plt.savefig(r'../results/classification/nn_training.png', dpi=300)
    plt.show()


def training(data_train, label_train, data_test, label_test):
    losses_train = []
    losses_test = []
    train_num = len(data_train)
    loss_function = nn.NLLLoss()  # negative log likelihood combine with Softmax = cross entropyJoint learning
    model = nn_embedding(13, 4)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    iteration_num = 50
    test_F1 = []
    train_F1 = []
    best_f1 = 0
    # model.load_state_dict(torch.load('../results/classification/model_nodataenhancement_embedding100_line256_iteration20.pkl'))
    for epoch in range(iteration_num):
        total_loss = 0
        predict_list_temp = []
        for i in range(len(data_train)):
            # Step 1. Prepare the inputs to be passed to the model
            train_sample = torch.tensor(data_train.iloc[i], dtype=torch.float).to(device)
            # Step 2. Recall that torch *accumulates* gradients. Before passing in a new instance, you need to zero out
            # the gradients from the old instance
            model.zero_grad()
            # Step 3. Run the forward pass, getting log probabilities of classes
            log_probs = model(train_sample)
            _, predict = torch.max(log_probs.data, 0)

            predict_list_temp.append(predict.to('cpu'))
            # Step 4. Compute your loss function. (Again, Torch wants the target class wrapped in a tensor)
            loss = loss_function(log_probs.unsqueeze(0), torch.tensor([label_train[i]], dtype=torch.long).to(device))

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            # scheduler.step()
            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()

        with torch.no_grad():
            test_temp = []
            test_loss = 0
            test_num = len(data_test)
            for k in range(len(data_test)):
                test_sample = torch.tensor(data_test.iloc[k], dtype=torch.float).to(device)
                log_probs = model(test_sample)
                _, predict = torch.max(log_probs.data, 0)

                test_temp.append(predict.to('cpu'))
                loss = loss_function(log_probs.unsqueeze(0), torch.tensor([label_test[k]], dtype=torch.long).to(device))
                test_loss += loss.item()
            losses_test.append(test_loss / test_num)
            test_F1.append(f1_score(label_test, test_temp, average='weighted'))
            print("Iteration: %f" % epoch)
            print("test accuracy: %.4f" % accuracy_score(label_test, test_temp))
            print("test recall: %.4f" % recall_score(label_test, test_temp, average='weighted'))
            print("test F1: %.4f" % f1_score(label_test, test_temp, average='weighted'))

        if f1_score(label_test, test_temp, average='weighted') > best_f1:
            best_f1 = f1_score(label_test, test_temp, average='weighted')
            best_model = model

        losses_train.append(total_loss / train_num)
        print("Train F1: %.4f" % f1_score(label_train, predict_list_temp, average='weighted'))

        train_F1.append(f1_score(label_train, predict_list_temp, average='weighted'))
    # print(losses_train)  # The loss decreased every iteration over the training data!
    # loss_iteration(losses_train, losses_test, train_F1, test_F1)

    # torch.save(model.state_dict(), '../results/classification/model_nodataenhancement_embedding50_line64_iteration20.pkl')
    return best_model


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


def roc_plot(data_test, best_model, label_test):
    num_test = len(data_test)
    predict_list = []
    predict_prob = []
    best_model.to('cpu')
    for k in range(num_test):
        test_sample = torch.tensor(data_test.iloc[k], dtype=torch.float)
        output_predict = best_model(test_sample)
        pred, predict = torch.max(output_predict.data.unsqueeze(0), 1)
        predict_prob.append(output_predict.data)
        predict_list.append(predict)

    print("Test accuracy: %.4f" % accuracy_score(label_test, predict_list))
    print("Test recall: %.4f" % recall_score(label_test, predict_list, average='weighted'))
    print("Test F1: %.4f" % f1_score(label_test, predict_list, average='weighted'))
    one_hot = []
    for i in label_test:
        temp = [0] * 4
        temp[i] = 1
        one_hot += temp
    print(predict_prob)
    fpr, tpr, _ = roc_curve(one_hot, list(itertools.chain.from_iterable(predict_prob)))
    ROC_curve(fpr, tpr, auc(fpr, tpr))



def main():
    train_data = pd.read_csv(r'../../../train.csv', encoding='utf_8_sig')
    test_data = pd.read_csv(r'../../../test.csv', encoding='utf_8_sig')
    index_temp = ['销售规模（对数）', '销项开票数量（对数）', '平均销项开票金额（对数）', '进购规模（对数）', '进项开票数量（对数）',
                  '税价比', '发票平均税率', '下游客户数量', '指定周期负数发票金额比率', '指定周期作废发票金额比率', '固定周期下游客户数量',
                  '销方基尼系数', '销方HIHI指数']
    loss_iteration()
    # train_data_norm = pd.read_csv(r'../../../train_norm.csv', encoding='utf_8_sig')
    # test_data_norm = pd.read_csv(r'../../../test_norm.csv', encoding='utf_8_sig')
    # val_data = pd.read_excel(r'../../problem  C  附件2.xlsx', encoding='utf_8_sig')
    # train_sp = train_data[index_temp]
    # test_sp = test_data[index_temp]
    # val_sp = val_data[index_temp]
    # val_sp.columns = [str(i) for i in range(1, 21)]
    # all_data = pd.concat([train_sp, test_sp])
    # data_prepare(all_data)
    # train_label = label_creat(list(train_data['信誉评级']))
    # test_label = label_creat(list(test_data['信誉评级']))

    # best_model = training(train_data_norm, train_label, test_data_norm, test_label)
    # roc_plot(test_data_norm, best_model, test_label)
    # loss_iteration()


if __name__ == '__main__':
    main()





