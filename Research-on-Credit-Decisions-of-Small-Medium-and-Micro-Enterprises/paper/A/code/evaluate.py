import pandas as pd
from collections import Counter
import numpy as np


dict_index = {1: 0.12, 17: 0.17, 2: 0.2, 4: 0.2, 3: 0.15, 10: 0.15, 5: 0.12, 11: 0.12, 12: 0.12, 13: 0.12, 6: 0.08, 18: 0.08, 20: 0.08, 7: 0.1, 14: 0.1, 8: 0.1, 9: 0.1, 15: 0.13, 16: 0.13, 19: 0.13}
weight_all = sum(dict_index.values())
negative = [9, 17]


def evaluate_index(raw_data, dic_temp):
    score_veh = []
    for i in range(len(raw_data)):
        temp = raw_data.iloc[i]
        score_temp = []
        for temp_index, temp_data in enumerate(temp):
            if temp_index+1 in dic_temp:
                if temp_index+1 in negative:
                    if temp[temp_index] > max(dic_temp[temp_index+1]):
                        score = 1
                    else:
                        for index_section, j in enumerate(dic_temp[temp_index+1]):

                            if temp[temp_index] <= j:
                                score = 5 - (index_section+1)
                                break

                else:
                    if temp[temp_index] > max(dic_temp[temp_index+1]):
                        score = 5
                    else:
                        for index_section, j in enumerate(dic_temp[temp_index+1]):
                            if temp[temp_index] <= j:
                                score = index_section+1
                                break
            else:
                if temp[temp_index] == 0:
                    score = 1
                else:
                    score = 5
            score_temp.append(score)
        score_sum = 0
        for score_index, score_i in enumerate(score_temp):
            score_sum += (score_i * (dict_index[score_index+1]/weight_all))
        score_veh.append(np.round(score_sum, 5))
    return score_veh


def evaluation(raw_data):
    dict_pre = {}
    for i in range(1, 21):
        list_temp = []
        temp = list(raw_data[str(i)])
        if len(Counter(temp)) > 2:
            list_temp.append(np.percentile(temp, 20))
            list_temp.append(np.percentile(temp, 40))
            list_temp.append(np.percentile(temp, 60))
            list_temp.append(np.percentile(temp, 80))
            dict_pre[i] = list_temp
    scores = evaluate_index(raw_data, dict_pre)
    return scores









def main():
    path = r'../result/附件二识别结果.csv'
    raw_data = pd.read_csv(path, encoding='utf_8_sig', engine='python')
    scores = evaluation(raw_data[[str(i) for i in range(1, 21)]])
    raw_data['性能得分'] = scores
    dict_code = {0:[], 1:[], 2:[], 3:[]}
    for index_temp, i in enumerate(list(raw_data['编队'])):
        dict_code[i].append(scores[index_temp])
    for i in dict_code:
        print(np.mean(dict_code[i]))
    # raw_data.to_csv(r'../result/附件二识别结果及打分.csv', encoding='utf_8_sig', index=False)


if __name__ == '__main__':
    main()



