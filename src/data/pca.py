import os
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
def plot_pca_scatter(x_pca,y_train,pics_path,type_id,value_type):
    colors = ['red', 'blue']
    plt.figure(figsize=(6,4))
    for i in range(len(colors)):
        px = x_pca[:, 0][y_train == i]
        py = x_pca[:, 1][y_train == i]
        plt.scatter(px, py, c=colors[i],s=5,alpha=0.9)
    plt.legend(("不购买","购买"))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    pic_path = os.path.join(pics_path,"pca")
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)
    pic_file = os.path.join(pic_path,"pca_type_%s_value_%s.png"%(value_type,type_id))
    plt.savefig(pic_file)
    plt.close()
def pca_method(filename,pics_path,processed_path):
    all_data = pd.read_excel(filename)
    # a标签参数的设置
    a_list = ['a%d' % k for k in range(1, 9)]
    # 标签b连续型数据集准备
    b_z_score = ['B2', 'B4', 'B8', 'B10', 'B13', 'B14', 'B15']
    b_max_min = ['B5', 'B7']
    b_other = ['B16', 'B17']
    b_c_list = ['B2', 'B4', 'B5', 'B7', 'B8', 'B10', 'B13', 'B14', 'B15', 'B16', 'B17']
    # 标签b离散型数据集准备
    b_d_list = ['B1', 'B3', 'B6', 'B9', 'B11', 'B12']
    type_ids = ['1', '2', '3']
    pca_path = os.path.join(processed_path, "pca")
    if not os.path.exists(pca_path):
        os.makedirs(pca_path)
    for tp_id in type_ids:
        type_id_data = all_data[all_data['品牌类型'] == int(tp_id)]
        a_data = type_id_data.loc[:, a_list]
        out_preprocess_y = type_id_data.loc[:, '购买意愿']
        a_data = a_data.values/100.0
        y_train = out_preprocess_y.values
        #用PCA对8维数据进行权重值拆解
        estimator_a = PCA(n_components=2)   # 使用PCA将原8维度图像压缩为2个维度
        pca_a_train = estimator_a.fit_transform(a_data)   # 利用训练特征决定20个正交维度的方向，并转化原训练特征

        weights_file = os.path.join(pca_path,"weights_a_value.xlsx")
        pd.DataFrame(estimator_a.components_).to_excel(weights_file,index=False,header=a_list)
        plot_pca_scatter(pca_a_train,y_train,pics_path,tp_id,"a")
        # b标签参数的设置
        # 连续型数据
        b_c_data = type_id_data.loc[:, b_c_list]
        b_c_data.loc[:, b_max_min] = (b_c_data.loc[:, b_max_min] - b_c_data.loc[:, b_max_min].min()) \
                                     / (b_c_data.loc[:, b_max_min].max() - b_c_data.loc[:, b_max_min].min())

        b_c_data.loc[:, b_z_score] = (b_c_data.loc[:, b_z_score] - b_c_data.loc[:, b_z_score].mean()) \
                                     / b_c_data.loc[:, b_z_score].std()
        b_c_data.loc[:, b_other] /= 100.0

        estimator_b_c = PCA(n_components=2)  # 使用PCA将原8维度图像压缩为2个维度
        pca_b_c_train = estimator_b_c.fit_transform(b_c_data)  # 利用训练特征决定20个正交维度的方向，并转化原训练特征

        weights_file = os.path.join(pca_path, "weights_b_continual_value.xlsx")
        pd.DataFrame(estimator_b_c.components_).to_excel(weights_file, index=False, header=b_c_list)
        plot_pca_scatter(pca_b_c_train, y_train, pics_path, tp_id, "bc")
