#coding=gbk
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

excel_path = r'D:\Pycharm\Project\data_analyze\Data_processing\data_2\data.xlsx'

df = pd.read_excel(excel_path)


right_tags_list = []
predict_tags_list = []

for tag in df['right_tags']:
    if tag is not np.nan:
        right_tags_list.extend(list(tag))

for tag in df['pkuseg_tourism_cut_tags']:
    if tag is not np.nan:
        predict_tags_list.extend(list(tag))

# print(metrics.precision_score(right_tags_list, predict_tags_list, average='micro'))
# print(metrics.recall_score(right_tags_list, predict_tags_list, average='micro'))
# print(metrics.f1_score(right_tags_list, predict_tags_list, average='micro'))
#
print(metrics.precision_score(right_tags_list, predict_tags_list, average='macro'))
print(metrics.recall_score(right_tags_list, predict_tags_list, average='macro'))
print(metrics.f1_score(right_tags_list, predict_tags_list, average='macro'))

# target_names = ['B', 'M', 'E', 'S']
# print(classification_report(right_tags_list, predict_tags_list, target_names=target_names))

# print(cohen_kappa_score(right_tags_list, predict_tags_list))

# sum = 0
# for tag_true, tag_pred in zip(right_tags_list, predict_tags_list):
# print(accuracy_score(right_tags_list, predict_tags_list))
    # sum += accuracy_score(tag_true, tag_pred)
# print(sum/len(predict_tags_list))