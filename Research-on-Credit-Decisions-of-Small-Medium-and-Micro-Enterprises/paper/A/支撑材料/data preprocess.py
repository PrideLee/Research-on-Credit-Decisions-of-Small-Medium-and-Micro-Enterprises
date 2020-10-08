import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
import wordcloud
import datetime
import random
import collections
import jieba
import imageio
import matplotlib as mpl
import seaborn as sns
import numpy as np
from collections import Counter
import random
import math
from pandas.tools.plotting import parallel_coordinates

mpl.rcParams['font.sans-serif'] = ['SimHei']


def industry_plt(dict_main):
    all_colors = list(plt.cm.colors.cnames.keys())
    c = random.choices(all_colors, k=len(dict_main.values()))
    # Plot Bars
    fig = plt.figure(figsize=(16, 10), dpi=80)
    ax1 = fig.add_subplot(111)
    list_all = list(dict_main.values())
    list_count = []
    list_prec = []
    list_prec_sum = [0]

    for i in list_all:
        list_count.append(int(i))
        list_prec.append(float(i / sum(list_all)))
        list_prec_sum.append(min(round(list_prec_sum[-1] + list_prec[-1], 2), 1.0))
    print(list_prec_sum)
    list_prec_sum.pop(0)
    plt.bar(dict_main.keys(), list_count, color=c, width=.5)
    for i, val in enumerate(list_count):
        plt.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom',
                 fontdict={'fontweight': 500, 'size': 12.5})
    # Decoration
    plt.gca().set_xticklabels(dict_main.keys(), rotation=20, horizontalalignment='right', fontsize=12.5)
    plt.title("各企业所属行业类别分布", fontsize=12.5)
    ax1.set_ylabel('各企业所属行业类别数目', fontsize=12.5)
    ax1.set_yticks(np.arange(0, math.ceil(list_count[0] / 100)), 100)

    ax2 = ax1.twinx()
    ax2.plot(list_prec_sum)
    ax2.scatter([i for i in range(len(list_prec_sum))], list_prec_sum, s=20, marker='x', color='red')
    # ax2.plot()
    ax2.grid(True)
    # 设置数字标签
    for a, b in enumerate(list_prec_sum):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=12.5)
    ax2.set_yticks(np.arange(0, 1.1, 0.1))  # 设置右边纵坐标刻度
    ax2.set_ylabel('各企业所属行业类别数目占比', fontsize=12.5)
    plt.show()


def cloud_img(comps, img):
    cut_list = []
    for comp_temp in comps:
        cut_text = jieba.cut(comp_temp, cut_all=False)
        cut_list += cut_text
    cut_name = dict(collections.Counter(cut_list))
    cut_name_dict = {}
    for cut_name_temp in cut_name:
        if cut_name[cut_name_temp] > 1:
            cut_name_dict[cut_name_temp] = cut_name[cut_name_temp]
    print(cut_name_dict)

    # wordcloud_2 = wordcloud.WordCloud(font_path='C:\Windows\Fonts\STZHONGS.TTF', mask=img,
    #                                   background_color='white',
    #                                   prefer_horizontal=0.75, random_state=50).fit_words(cut_name_dict)
    # plt.figure("company name cloud")
    # plt.imshow(wordcloud_2)
    # plt.axis('off')
    # plt.show()


def comp_any(raw_data_comp):
    name_comp = raw_data_comp['企业名称']
    name_comp_clean = []
    for name_temp in name_comp:
        name_temp = name_temp.replace('*', '')
        if '有限责任公司' in name_temp:
            name_temp = name_temp.replace('有限责任公司', '')
        if '有限公司' in name_temp:
            name_temp = name_temp.replace('有限公司', '')
        if '分公司' in name_temp:
            name_temp = name_temp.replace('分公司', '')
        if '公司' in name_temp:
            name_temp = name_temp.replace('公司', '')
        name_comp_clean.append(name_temp)
    image = imageio.imread(r'money.jpg')
    cloud_img(name_comp_clean, image)


def plot_pie(dicts, name_title):
    labels = list(dicts.keys())
    X = list(dicts.values())
    fig = plt.figure(figsize=(24, 16), dpi=80)
    explode = tuple([0.01] * len(X))
    plt.pie(X, explode=explode, labels=labels, labeldistance=1.2, autopct='%1.2f%%', shadow=False, startangle=90,
            pctdistance=0.6)
    plt.legend(loc='lower center', ncol=5, fancybox=True, shadow=True)
    plt.title(name_title, fontsize=12.5)
    plt.show()


def read_data(path_raw):
    raw_data_comp = pd.read_excel(path_raw, encoding='utf_8_sig', sheet_name='企业信息')
    # comp_any(raw_data_comp)
    # plot_pie(dict(Counter(raw_data_comp['信誉评级'])), '信誉评级')
    plot_pie(dict(Counter(raw_data_comp['是否违约'])), '是否违约')


def func_deg():
    x = np.arange(0, 5.01, 0.01)
    y_1 = -x / 5 + 1
    y_2 = cos(np.pi / 10 * x)
    y_3 = 1 / (x + 1)
    y_4 = 1 / (1 + exp(x - 2.5))
    plt.plot(x, y_1, label="y=-x/5+1", color="red", linewidth=2, linestyle='-')
    plt.plot(x, y_2, label="y=cos(pi*x/10)", color="blue", linewidth=2, linestyle=':')
    plt.plot(x, y_3, label="y=1/(x+1)", color="black", linewidth=2, linestyle='-.')
    plt.plot(x, y_4, label="y=1/(1+exp(x-2.5))", color="green", linewidth=2, linestyle='--')
    plt.grid(True)
    plt.xlim(0, 5)
    plt.ylim(0, 1)
    plt.xlabel('风险打分', fontsize=12.5)
    plt.ylabel('违约率', fontsize=12.5)
    plt.legend()
    plt.show()


def loss_fig(path_loss):
    raw_data_comp = pd.read_excel(path_loss, encoding='utf_8_sig', sheet_name='Sheet1')
    print(raw_data_comp)
    x_point = list(raw_data_comp['贷款年利率'])[1:]
    y_point = list(raw_data_comp['客户流失率'])[1:]
    x = np.arange(0.03, 0.15, 0.001)
    y_1 = 0.7701 * exp(1.453 * x) - 2.306 * exp(-26.48 * x)
    y_2 = 640.9 * x ** 3 - 258.6 * x ** 2 + 37.97 * x - 1.12
    y_3 = -0.17 * x ** (-0.68) + 1.548
    y_4 = 2.405 * sin(23.62 * x - 1.446) + 1.559 * sin(28.6 * x + 1.141)
    plt.plot(x, y_1, label="y=0.77*exp(1.45*x)-2.31*exp(-26.48*x)", color="red", linewidth=2, linestyle='-')
    plt.plot(x, y_2, label="640.9*x**3-258.6*x**2+37.97*x-1.12", color="blue", linewidth=2, linestyle=':')
    plt.plot(x, y_3, label="-0.17*x**(-0.68) + 1.548", color="black", linewidth=2, linestyle='-.')
    plt.plot(x, y_4, label="2.41*sin(23.62*x-1.146)+1.56*sin(28.6*x+1.14)", color="green", linewidth=2, linestyle='--')
    plt.plot(x_point, y_point, 'yo', alpha=0.6, linewidth=0.1)
    plt.grid(True)
    plt.xlim(0.03, 0.15)
    plt.ylim(0, 1.1)
    plt.xlabel('贷款年利率', fontsize=12.5)
    plt.ylabel('客户流失率', fontsize=12.5)
    plt.legend()
    plt.show()


def bar_any(dict_main):
    all_colors = list(plt.cm.colors.cnames.keys())
    c = random.choices(all_colors, k=len(dict_main.values()))
    # Plot Bars
    fig = plt.figure(figsize=(16, 10), dpi=80)
    ax1 = fig.add_subplot(111)

    list_count = dict_main.values()

    num = [str(i) for i in list(dict_main.keys())]

    plt.bar(dict_main.keys(), list_count, color=c)
    for i, val in enumerate(list_count):
        plt.text(i, val, float(val), fontdict={'fontweight': 12.5, 'size': 12.5})
    # Decoration
    plt.title("企业不同信誉等级数目柱状图", fontsize=12.5)
    ax1.set_ylabel('各类信誉等级企业数目', fontsize=12.5)

    plt.show()



def pre_analysis(path_raw):
    raw_data = pd.read_csv(path_raw, encoding='utf_8_sig', engine='python')

    # plot_pie(dict(Counter(list(raw_data['信誉评级']))), '附件二企业信誉评级')
    bar_any(dict(Counter(list(raw_data['信誉评级']))))


def main():

    path = r'../../../附件2：302家无信贷记录企业的相关数据.xlsx'
    path_out = r'../../../附件二销项发票.csv'
    path_in = r'../../../附件二进项发票.csv'
    path_index = r'../../B/result/index_2.csv'
    path_score = r'../../B/result/score.csv'


    loss_fig(r'../../../附件3：银行贷款年利率与客户流失率关系的统计数据.xlsx')
    # plot_score(path_score)
    # comprehensive_score(path_score, path_save=r'../../B/result/score_all.csv')

    # score_comp(path_index, path_save=r'../../B/result/score_1.csv')
    # correlation(path)
    # index_system_out(path_out, r'../../../index_2.csv')

    # index_system_out(path, path_in)
    # index_system(path_in)
    # index_system_in(path_in, r'../../../index_add_2.csv')
    # pie_chart(path)

    # read_data(path)
    # path_loss = path = r'../../../附件3：银行贷款年利率与客户流失率关系的统计数据.xlsx'
    # loss_fig(path_loss)
    # func_deg()
    # comp_dict = {'公共管理': 10, '建筑业': 20, '商务服务业': 11, '其它': 9, '科学研究和技术服务业': 15, '批发和零售业': 15, '交通运输、仓储和邮政业': 4,
    #  '农、林、牧、渔业': 8, '住宿和餐饮业': 5, '制造业': 14, '文化、体育和娱乐业': 6, '教育': 6}
    # comp_dict_sort = {}
    # for i,j in sorted(comp_dict.items(), key=lambda item:item[1], reverse=True):
    #     comp_dict_sort[i] = j
    # industry_plt(comp_dict_sort)


if __name__ == "__main__":
    main()
