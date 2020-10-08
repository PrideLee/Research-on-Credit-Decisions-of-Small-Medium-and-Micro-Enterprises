import pandas as pd
import matplotlib.pyplot as plt
import datetime
import random
import collections
import imageio
import matplotlib as mpl
import seaborn as sns
import numpy as np
from collections import Counter
import random
import math
from pandas.tools.plotting import parallel_coordinates

mpl.rcParams['font.sans-serif'] = ['SimHei']

import plotly.graph_objects as go


def pie_chart(path_raw):
    raw_data_comp = pd.read_excel(path_raw, encoding='utf_8_sig', sheet_name='企业信息')
    temp_dict = dict(Counter(raw_data_comp['是否违约']))
    labels = list(temp_dict.keys())
    values = list(temp_dict.values())
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                                 insidetextorientation='radial')])
    fig.show()


def hot_map(cosin_epoch):
    y_grid = ['是', '否']
    x_grid = ['A', 'B', 'C', 'D']
    # 定义画布为1*1个划分，并在第1个位置上进行作图
    # array_save = np.array(cosin_epoch).transpose()
    sns.heatmap(cosin_epoch, annot=cosin_epoch, fmt=".2g", cmap="RdBu_r", xticklabels=x_grid, yticklabels=y_grid)
    plt.xlabel("信誉评级", fontsize=12.5)
    plt.ylabel("是否违约", fontsize=12.5)
    plt.title("信誉评级与是否违约的联合概率", fontsize=12.5)
    plt.show()


def correlation(path_raw):
    raw_data_comp = pd.read_excel(path_raw, encoding='utf_8_sig', sheet_name='企业信息')
    a_1 = 0
    b_1 = 0
    c_1 = 0
    d_1 = 0
    a_0 = 0
    b_0 = 0
    c_0 = 0
    d_0 = 0
    for i in range(len(raw_data_comp)):
        if raw_data_comp.iloc[i]['是否违约'] == '是' and raw_data_comp.iloc[i]['信誉评级'] == 'A':
            a_1 += 1
        elif raw_data_comp.iloc[i]['是否违约'] == '是' and raw_data_comp.iloc[i]['信誉评级'] == 'B':
            b_1 += 1
        elif raw_data_comp.iloc[i]['是否违约'] == '是' and raw_data_comp.iloc[i]['信誉评级'] == 'C':
            c_1 += 1
        elif raw_data_comp.iloc[i]['是否违约'] == '是' and raw_data_comp.iloc[i]['信誉评级'] == 'D':
            d_1 += 1
        elif raw_data_comp.iloc[i]['是否违约'] == '否' and raw_data_comp.iloc[i]['信誉评级'] == 'A':
            a_0 += 1
        elif raw_data_comp.iloc[i]['是否违约'] == '否' and raw_data_comp.iloc[i]['信誉评级'] == 'B':
            b_0 += 1
        elif raw_data_comp.iloc[i]['是否违约'] == '否' and raw_data_comp.iloc[i]['信誉评级'] == 'C':
            c_0 += 1
        elif raw_data_comp.iloc[i]['是否违约'] == '否' and raw_data_comp.iloc[i]['信誉评级'] == 'D':
            d_0 += 1
    break_dict = dict(Counter(raw_data_comp['是否违约']))
    repu_dict = dict(Counter(raw_data_comp['信誉评级']))
    a_1_p = round(a_1 / (break_dict['是'] + repu_dict['A'] - a_1), 4)
    b_1_p = round(b_1 / (break_dict['是'] + repu_dict['B'] - b_1), 4)
    c_1_p = round(c_1 / (break_dict['是'] + repu_dict['C'] - c_1), 4)
    d_1_p = round(d_1 / (break_dict['是'] + repu_dict['D'] - d_1), 4)
    a_0_p = round(a_0 / (break_dict['否'] + repu_dict['A'] - a_0), 4)
    b_0_p = round(b_0 / (break_dict['否'] + repu_dict['B'] - b_0), 4)
    c_0_p = round(c_0 / (break_dict['否'] + repu_dict['C'] - c_0), 4)
    d_0_p = round(d_0 / (break_dict['否'] + repu_dict['D'] - d_0), 4)
    hot_map(np.array([[a_1_p, b_1_p, c_1_p, d_1_p], [a_0_p, b_0_p, c_0_p, d_0_p]]))


def bbox(list_value, list_name):
    y = np.transpose(np.array(list_value))
    labels = list_name
    plt.boxplot(y, labels=labels, sym='o')
    plt.xlabel('指标', fontsize=12.5)
    plt.ylabel('指标值', fontsize=12.5)
    plt.title("企业风险评估指标箱型图", fontsize=12.5)
    plt.grid(True)
    plt.show()


def gini(money_list):
    money_list_sort = sorted(money_list, reverse=True)[1:]
    all_money = sum(money_list_sort)
    return round(1-(1/(len(money_list_sort)+1))*(2*sum([i/all_money for i in money_list_sort])+1), 5)


def hini(money_list):
    money_list_sort = sorted(money_list, reverse=True)
    total_all = sum(money_list_sort)
    hini_index = 0
    for i in money_list_sort[:min(50, len(money_list_sort))]:
        hini_index += (i/total_all)**2
    return round(hini_index, 5)


def sell_all(sell_money_dict, sell_tax_dict, sell_all_dict, sell_comp_dict, sell_status_dict):
    sell_scale = {}
    sell_num = {}
    sell_average = {}
    sell_tax_all_ratio = {}
    sell_tax_money_ratio = {}
    num_cust = {}
    sell_negative_dict = {}
    wrong_dict = {}
    gini_dict = {}
    hini_dict = {}
    for comp_temp in sell_money_dict:
        sell_scale[comp_temp] = round(np.log(sum(sell_money_dict[comp_temp])), 5)
        sell_num[comp_temp] = round(np.log(len(sell_money_dict[comp_temp])), 5)
        sell_average[comp_temp] = round(np.log(sum(sell_money_dict[comp_temp]) / len(sell_money_dict[comp_temp])), 5)
        sell_tax_all_ratio[comp_temp] = round(sum(sell_tax_dict[comp_temp]) / sum(sell_all_dict[comp_temp]), 5)
        sell_tax_money_ratio[comp_temp] = round(sum(sell_tax_dict[comp_temp]) / sum(sell_money_dict[comp_temp]), 5)
        num_cust[comp_temp] = len(list(set(sell_comp_dict[comp_temp])))
        sell_negative_dict[comp_temp] = round(
            sum([abs(i) for i in sell_all_dict[comp_temp] if i < 0]) / sum(sell_all_dict[comp_temp]), 5)
        wrong_dict[comp_temp] = round(sum(
            [abs(sell_all_dict[comp_temp][index_temp]) for index_temp, status_temp in enumerate(sell_status_dict) if
             status_temp == '作废发票']) / sum(sell_all_dict[comp_temp]), 5)
        gini_dict[comp_temp] = gini(sell_all_dict[comp_temp])
        hini_dict[comp_temp] = hini(sell_all_dict[comp_temp])
    return sell_scale, sell_num, sell_average, sell_tax_all_ratio, sell_tax_money_ratio, num_cust, sell_negative_dict, wrong_dict, gini_dict, hini_dict


def buy_all(sell_money_dict):
    sell_scale = {}
    sell_num = {}
    for comp_temp in sell_money_dict:
        sell_scale[comp_temp] = round(np.log(sum(sell_money_dict[comp_temp])), 5)
        sell_num[comp_temp] = round(np.log(len(sell_money_dict[comp_temp])), 5)
    return sell_scale, sell_num


def index_system_in(path_raw, path_save):
    raw_data_sell = pd.read_csv(path_raw, encoding='utf_8_sig', engine='python')
    sell_comp_dict = {}
    sell_money_dict = {}
    sell_tax_dict = {}
    sell_all_dict = {}
    sell_status_dict = {}
    for i in range(len(raw_data_sell)):
        if raw_data_sell.iloc[i]['企业代号'] in sell_comp_dict:
            sell_money_dict[raw_data_sell.iloc[i]['企业代号']].append(raw_data_sell.iloc[i]['金额'])
            # sell_comp_dict[raw_data_sell.iloc[i]['企业代号']].append(raw_data_sell.iloc[i]['销方单位代号'])
            # sell_tax_dict[raw_data_sell.iloc[i]['企业代号']].append(raw_data_sell.iloc[i]['税额'])
            # sell_all_dict[raw_data_sell.iloc[i]['企业代号']].append(raw_data_sell.iloc[i]['价税合计'])
            # sell_status_dict[raw_data_sell.iloc[i]['企业代号']].append(raw_data_sell.iloc[i]['发票状态'])
        else:
            sell_money_dict[raw_data_sell.iloc[i]['企业代号']] = [raw_data_sell.iloc[i]['金额']]
            # sell_comp_dict[raw_data_sell.iloc[i]['企业代号']] = [raw_data_sell.iloc[i]['销方单位代号']]
            # sell_tax_dict[raw_data_sell.iloc[i]['企业代号']] = [raw_data_sell.iloc[i]['税额']]
            # sell_all_dict[raw_data_sell.iloc[i]['企业代号']] = [raw_data_sell.iloc[i]['价税合计']]
            # sell_status_dict[raw_data_sell.iloc[i]['企业代号']] = [raw_data_sell.iloc[i]['发票状态']]
    buy_scale, buy_num = buy_all(sell_money_dict)
    pd.DataFrame({'进购规模（对数）': buy_scale, '进项开票数量（对数）': buy_num}).to_csv(path_save, encoding='utf_8_sig', index=False)


def index_system_out(path_raw, path_save):
    # pd.read_excel(path_raw, encoding='utf_8_sig', sheet_name='进项发票信息').to_csv(path_save, encoding='utf_8_sig', index=False)
    raw_data_sell = pd.read_csv(path_raw, encoding='utf_8_sig', engine='python')
    # raw_data_pru = pd.read_excel(path_raw, encoding='utf_8_sig', sheet_name='进项发票信息')
    sell_comp_dict = {}
    sell_money_dict = {}
    sell_tax_dict = {}
    sell_all_dict = {}
    sell_status_dict = {}
    for i in range(len(raw_data_sell)):
        if raw_data_sell.iloc[i]['企业代号'] in sell_comp_dict:
            sell_comp_dict[raw_data_sell.iloc[i]['企业代号']].append(raw_data_sell.iloc[i]['购方单位代号'])
            sell_money_dict[raw_data_sell.iloc[i]['企业代号']].append(raw_data_sell.iloc[i]['金额'])
            sell_tax_dict[raw_data_sell.iloc[i]['企业代号']].append(raw_data_sell.iloc[i]['税额'])
            sell_all_dict[raw_data_sell.iloc[i]['企业代号']].append(raw_data_sell.iloc[i]['价税合计'])
            sell_status_dict[raw_data_sell.iloc[i]['企业代号']].append(raw_data_sell.iloc[i]['发票状态'])
        else:
            sell_comp_dict[raw_data_sell.iloc[i]['企业代号']] = [raw_data_sell.iloc[i]['购方单位代号']]
            sell_money_dict[raw_data_sell.iloc[i]['企业代号']] = [raw_data_sell.iloc[i]['金额']]
            sell_tax_dict[raw_data_sell.iloc[i]['企业代号']] = [raw_data_sell.iloc[i]['税额']]
            sell_all_dict[raw_data_sell.iloc[i]['企业代号']] = [raw_data_sell.iloc[i]['价税合计']]
            sell_status_dict[raw_data_sell.iloc[i]['企业代号']] = [raw_data_sell.iloc[i]['发票状态']]
    sell_scale, sell_num, sell_average, sell_tax_all_ratio, sell_tax_money_ratio, num_cust, sell_negative_dict, \
    wrong_dict, gini_dict, hini_dict = sell_all(sell_money_dict, sell_tax_dict, sell_all_dict, sell_comp_dict, sell_status_dict)
    pd.DataFrame({'销售规模（对数）':sell_scale, '销项开票数量（对数）':sell_num, '平均销项开票金额（对数）':sell_average, '税价比':sell_tax_all_ratio,
                  '发票平均税率':sell_tax_money_ratio, '下游客户数量':num_cust, '指定周期负数发票金额比率':sell_negative_dict,
                  '指定周期作废发票金额比率':wrong_dict, '固定周期下游客户数量':num_cust, '销方基尼系数':gini_dict, '销方HIHI指数':hini_dict}).to_csv(path_save, encoding='utf_8_sig', index=False)




    # bbox([list(sell_scale.values()), list(sell_num.values()), list(sell_average.values())],
    #      ['销售规模（对数）', '销售开票数量（对数）', '平均销售开票金额（对数）'])


def score_comp(path_index, path_save):
    raw_data = pd.read_csv(path_index, encoding='utf_8_sig', engine='python')
    positive_index = ['销售规模（对数）', '销项开票数量（对数）', '平均销项开票金额（对数）', '进购规模（对数）',
                      '进项开票数量（对数）', '税价比', '发票平均税率', '下游客户数量', '固定周期下游客户数量']
    negative_index = ['指定周期负数发票金额比率', '指定周期作废发票金额比率', '销方基尼系数', '销方HIHI指数']
    class_index = ['信誉评级']
    bi_index = ['是否违约']
    dict_score = {}
    dict_score['企业代号'] = list(raw_data['企业代号'])
    for positive_temp in positive_index:
        dict_score[positive_temp] = []
        list_temp = list(raw_data[positive_temp])
        min_temp = min(list_temp)
        max_temp = max(list_temp)
        temp_20 = np.percentile(list_temp, 20)
        temp_40 = np.percentile(list_temp, 40)
        temp_60 = np.percentile(list_temp, 60)
        temp_80 = np.percentile(list_temp, 80)
        for val_temp in list_temp:
            if min_temp <= val_temp < temp_20:
                dict_score[positive_temp].append(1)
            elif temp_20 <= val_temp < temp_40:
                dict_score[positive_temp].append(2)
            elif temp_40 <= val_temp < temp_60:
                dict_score[positive_temp].append(3)
            elif temp_60 <= val_temp < temp_80:
                dict_score[positive_temp].append(4)
            elif temp_80 <= val_temp <= max_temp:
                dict_score[positive_temp].append(5)

    for negative_temp in negative_index:
        dict_score[negative_temp] = []
        list_temp = list(raw_data[negative_temp])
        min_temp = min(list_temp)
        max_temp = max(list_temp)
        temp_20 = np.percentile(list_temp, 20)
        temp_40 = np.percentile(list_temp, 40)
        temp_60 = np.percentile(list_temp, 60)
        temp_80 = np.percentile(list_temp, 80)
        for val_temp in list_temp:
            if min_temp <= val_temp < temp_20:
                dict_score[negative_temp].append(5)
            elif temp_20 <= val_temp < temp_40:
                dict_score[negative_temp].append(4)
            elif temp_40 <= val_temp < temp_60:
                dict_score[negative_temp].append(3)
            elif temp_60 <= val_temp < temp_80:
                dict_score[negative_temp].append(2)
            elif temp_80 <= val_temp <= max_temp:
                dict_score[negative_temp].append(1)
    dict_score['信誉评级'] = []
    for level_temp in list(raw_data['信誉评级']):
        if level_temp == 'A':
            dict_score['信誉评级'].append(5)
        elif level_temp == 'B':
            dict_score['信誉评级'].append(4)
        elif level_temp == 'C':
            dict_score['信誉评级'].append(2)
        elif level_temp == 'D':
            dict_score['信誉评级'].append(1)
    dict_score['是否违约'] = []
    for level_temp in list(raw_data['是否违约']):
        if level_temp == '否':
            dict_score['是否违约'].append(5)
        elif level_temp == '是':
            dict_score['是否违约'].append(1)

    pd.DataFrame(dict_score).to_csv(path_save, encoding='utf_8_sig', index=False)


def comprehensive_score(path_score, path_save):
    weight= [0.078,0.041,0.065,0.048,0.017,0.041,0.051,0.045,0.037,0.021,0.05,0.045,0.025,0.095,0.335]
    raw_data = pd.read_csv(path_score, encoding='utf_8_sig')
    score_all = []
    for i in range(len(raw_data)):
        score_temp = 0
        for j, weight_temp in zip(list(raw_data.iloc[i])[1:], weight):
            score_temp += float(j) * weight_temp
        score_all.append(round(score_temp, 5))
    pd.DataFrame({'综合得分':score_all}).to_csv(path_save, encoding='utf_8_sig', index=False)


def bar_plt(dict_main, list_main):
    all_colors = list(plt.cm.colors.cnames.keys())
    c = random.choices(all_colors, k=len(dict_main.values()))
    # Plot Bars
    fig = plt.figure(figsize=(16, 10), dpi=80)
    ax1 = fig.add_subplot(111)
    list_count = dict_main.values()
    num = [str(i) for i in list(dict_main.keys())]
    plt.hist(list_main, bins=50, normed=0, facecolor="yellow", edgecolor="black", alpha=0.7)
    # plt.bar(dict_main.keys(), list_count, color=c)
    # for i, val in enumerate(list_count):
    #     plt.text(i, val, float(val), fontdict={'size': 12.5})
    # Decoration
    plt.title("信誉为D企业得分统计柱状图", fontsize=12.5)
    ax1.set_ylabel('得分数目', fontsize=12.5)
    plt.show()


def plot_score(path_raw):
    raw_data = pd.read_csv(path_raw, encoding='utf_8_sig')
    score = list(raw_data['综合得分'])
    level = list(raw_data['信誉评级'])
    score_round = []
    for score_temp in score:
        score_round.append(round(score_temp, 1))
    score_a = []
    score_b = []
    score_c = []
    score_d = []
    for level_temp, score_temp in zip(level, score_round):
        if level_temp == 5:
            score_a.append(score_temp)
        elif level_temp == 4:
            score_b.append(score_temp)
        elif level_temp == 2:
            score_c.append(score_temp)
        elif level_temp == 1:
            score_d.append(score_temp)
    bar_plt(dict(Counter(score_d)), score_d)


def loss_fig(path_loss):
    raw_data_comp = pd.read_excel(path_loss, encoding='utf_8_sig', sheet_name='Sheet1')

    x_point_a = list(raw_data_comp['贷款年利率'])[1:]
    y_point_a = list(raw_data_comp['客户流失率'])[1:]
    y_point_b = list(raw_data_comp['Unnamed: 2'])[1:]
    y_point_c = list(raw_data_comp['Unnamed: 3'])[1:]
    x = np.arange(0.03, 0.15, 0.001)
    y_a = -0.17 * x ** (-0.68) + 1.548
    y_b = -0.297 * x ** (-0.546) + 1.721
    y_c = -0.51 * x ** (-0.43) + 2.05
    plt.plot(x, y_a, label="A: y=-0.17*x**(-0.68)+1.548", color="red", linewidth=2, linestyle='-')
    plt.plot(x, y_b, label="B: y=-0.297*x**(-0.546)+1.721", color="blue", linewidth=2, linestyle=':')
    plt.plot(x, y_c, label="C: y=-0.51*x**(-0.43)+2.05", color="black", linewidth=2, linestyle='-.')
    plt.plot(x_point_a, y_point_a, 'ro', alpha=0.6, linewidth=0.1)
    plt.plot(x_point_a, y_point_b, 'bo', alpha=0.6, linewidth=0.1)
    plt.plot(x_point_a, y_point_c, 'ko', alpha=0.6, linewidth=0.1)
    plt.grid(True)
    plt.xlim(0.03, 0.15)
    plt.ylim(0, 1.1)
    plt.xlabel('贷款年利率', fontsize=12.5)
    plt.ylabel('客户流失率', fontsize=12.5)
    plt.legend()
    plt.show()


def strategy_money(path_raw, path_save):
    raw_data = pd.read_csv(path_raw, encoding='utf_8_sig', engine='python')
    comp_index = list(raw_data['企业代号'])
    level = list(raw_data['信誉评级'])
    strategy_list = []
    strategy_rate_list = []
    for level_temp in level:
        if level_temp == 'D':
            strategy_list.append(0)
            strategy_rate_list.append(0)
        elif level_temp == 'A':
            strategy_list.append(70+random.randint(-10, 10))
            strategy_rate_list.append(12+random.randint(-2, 2))
        elif level_temp == 'B':
            strategy_list.append(50+random.randint(-10, 10))
            strategy_rate_list.append(8+random.randint(-2, 2))
        if level_temp == 'C':
            strategy_list.append(30+random.randint(-10, 10))
            strategy_rate_list.append(6+random.randint(-2, 2))
    if sum(strategy_list) - 10000>0:
        x0 = np.random.rand(len(raw_data))
        ratio = (sum(strategy_list) - 10000) / sum(x0)
        x1 = x0 * ratio
        for i in range(len(x1)):
            strategy_list[i] = round(strategy_list[i]-x1[i], 2)
    else:
        x0 = np.random.rand(len(raw_data))
        ratio = abs(sum(strategy_list) - 10000) / sum(x0)
        x1 = x0 * ratio
        for i in range(len(x1)):
            strategy_list[i] = round(strategy_list[i] + x1[i], 2)
    pd.DataFrame({'企业代号':comp_index, '信誉评级':level, '借贷金额（万元）':strategy_list, '借贷年利率':strategy_rate_list}).to_csv(path_save, encoding='utf_8_sig', index=False)


def ra_break():
    x = np.arange(0,1, 0.01)
    y = exp(-3*x)-0.2
    plt.plot(x,y, label='y=exp(-3x)-0.2')
    plt.legend()
    plt.grid(True)
    plt.xlabel('企业“可靠性”', fontsize=12.5)
    plt.ylabel('企业违约率', fontsize=12.5)
    plt.show()




def degree_eff():
    x = np.arange(-3, 3, 0.1)
    y_1 = -1/(1+exp(-2*x))+1
    y_2 = -1/(1+exp(-x))+1
    y_3 = -1/(1+exp(-0.5*x))+1
    plt.plot(x, y_1, label='y=-1/(1+exp(-2*x))+1', color="red")
    plt.plot(x, y_2, label='y=-1/(1+exp(-x))+1', color="blue")
    plt.plot(x, y_3, label='y=-1/(1+exp(-0.5*x))+1', color="green")
    plt.legend()
    plt.grid(True)
    plt.xlabel('企业规模', fontsize=12.5)
    plt.ylabel('企业受影响程度', fontsize=12.5)
    plt.show()


def change(money, total_money, raw_data, num):
    reed = [random.random() for i in range(num+1)]

    total = sum(reed)
    count_add = 0
    count_de = 0
    flag_add = 0
    flag_de = 0
    for i in range(len(raw_data)):
        if flag_add == 0:
            if raw_data.iloc[i]['信誉评级'] == 'A' or raw_data.iloc[i]['信誉评级'] == 'B':
                money[i] = min(money[i] + reed[count_add] / total * total_money, 100)
                count_add += 1
                if count_add == num:
                    flag_add = 1
        if flag_de == 0:
            if raw_data.iloc[i]['信誉评级'] == 'C':
                count_de += 1
                money[i] = max(money[i] - reed[count_de] / total * total_money, 10)
                if count_de == num:
                    flag_de = 1
        if flag_add == flag_de == 1:
            break
    return money


def strategy_adj(path_raw, path_save):
    raw_data = pd.read_csv(path_raw, encoding='utf_8_sig', engine='python')
    rate = list(raw_data['借贷年利率'])
    for i in range(len(rate)):
        if rate[i]>0:
            rate[i] = max(rate[i]-random.random()*3, 4)
    money = list(raw_data['借贷金额（万元）'])
    money_1 = change(money, 4000, raw_data, num=80)
    money_2 = []
    money_3 = []
    for i in money_1:
        if i!= 0:
            money_2.append(round(max(min(-5+10*random.random()+i,100),10),2))
        else:
            money_2.append(0)
    for i in money_1:
        if i!= 0:
            money_3.append(round(max(min(-10+20*random.random()+i,100),10),2))
        else:
            money_3.append(0)

    # money_2 = change(money, 5000, raw_data, num=50)
    # money_3 = change(money, 8000, raw_data, num=80)
    print(len(money_3))
    pd.DataFrame({'企业代号': list(raw_data['企业代号']), '信誉评级': list(raw_data['信誉评级']),
                  '借贷金额（万元, 0.5）': money_1, '借贷金额（万元, 1）': money_2, '借贷金额（万元, 2）': money_3,
                  '借贷年利率（%）':rate}).to_csv(path_save, encoding='utf_8_sig', index=False)








def main():

    path = r'../../../附件2：302家无信贷记录企业的相关数据.xlsx'
    path_out = r'../../../附件二销项发票.csv'
    path_in = r'../../../附件二进项发票.csv'
    path_index = r'../../B/result/index_2.csv'
    path_score = r'../../B/result/score.csv'
    path_pre = r'../../B/result/附件二企业信誉评级识别结果.csv'
    # strategy_money(path_pre, r'../../B/result/问题二借贷方案.csv')
    # ra_break()
    # degree_eff()
    # pre_analysis(path_pre)

    strategy_adj(r'../../B/result/问题二借贷方案.csv', r'../../B/result/借贷方案调整策略.csv')
    # loss_fig(r'../../../附件3：银行贷款年利率与客户流失率关系的统计数据.xlsx')
    # plot_score(path_score)
    # comprehensive_score(path_score, path_save=r'../../B/result/score_all.csv')
    # score_comp(path_index, path_save=r'../../B/result/score_1.csv')
    # correlation(path)


if __name__ == "__main__":
    main()
