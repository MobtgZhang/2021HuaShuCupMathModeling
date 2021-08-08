import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
def kde_method(filename, logger, pic_path):
    all_data = pd.read_excel(filename)
    a_list = ['a%d' % k for k in range(1, 9)]
    selected_items = ['目标客户编号', '品牌类型'] + a_list
    data = all_data.loc[:, selected_items]
    data = data.copy(deep=True)
    type_ids = ['1', '2', '3']
    for tp_id in type_ids:
        type_id_data = data[data['品牌类型'] == int(tp_id)]
        out_preprocess = type_id_data.loc[:, a_list]
        # 这里抽取80%作为trian_data,20%作为test_data
        length = len(out_preprocess)
        train_len = int(length * 0.8)
        data = shuffle(data)
        train_data = data.iloc[:train_len].loc[:, a_list]
        test_data = data.iloc[train_len:].loc[:, a_list]
        # KDE 分布图
        dist_cols = 4
        dist_rows = int(len(test_data.columns)/dist_cols)
        plt.figure(figsize=(6 * dist_cols, 6 * dist_rows))
        i = 1
        for col in test_data.columns:
            ax = plt.subplot(dist_rows, dist_cols, i)
            ax = sns.kdeplot(train_data[col], color='Red', shade=True)
            ax = sns.kdeplot(test_data[col], color='Blue', shade=True)
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax = ax.legend(['train', 'test'])
            i += 1
        kde_path = os.path.join(pic_path, 'kde')
        if not os.path.exists(kde_path):
            os.makedirs(kde_path)
        pic_file = os.path.join(kde_path, "kde_type_%s" % (tp_id))
        plt.savefig(pic_file)
        plt.close()
