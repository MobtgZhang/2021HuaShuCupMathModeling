# 通过岭回归模型找出异常值，并绘制其分布
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def find_outliers(model, X, y,logger,pic_file, sigma=3,):
    # predict y values using model
    try:
        y_pred = pd.Series(model.predict(X), index=y.index)
    # if predicting fails, try fitting the model first
    except:
        model.fit(X, y)
        y_pred = pd.Series(model.predict(X), index=y.index)

    # calculate residuals between the model prediction and true y values
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()

    # calculate z statistic, define outliers to be where |z|>sigma
    z = (resid - mean_resid) / std_resid
    outliers = z[abs(z) > sigma].index

    # print and plot the results
    mse_score = mean_squared_error(y, y_pred)
    R2_score = model.score(X, y)
    logger.info('R2=%s'%(str(R2_score)))
    logger.info('Mse=%s'%(str(mse_score)))
    logger.info('-------------------------------------------------------')


    logger.info('%d outliers; ALL data shape:%s'%(len(outliers),str(X.shape)))

    plt.figure(figsize=(15, 5))
    ax_131 = plt.subplot(1, 3, 1)
    plt.plot(y, y_pred, '.')
    plt.plot(y.loc[outliers], y_pred.loc[outliers], 'ro')
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('y')
    plt.ylabel('y_pred')

    ax_132 = plt.subplot(1, 3, 2)
    plt.plot(y, y - y_pred, '.')
    plt.plot(y.loc[outliers], y.loc[outliers] - y_pred.loc[outliers], 'ro')
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('y')
    plt.ylabel('y - y_pred')

    ax_133 = plt.subplot(1, 3, 3)
    z.plot.hist(bins=50, ax=ax_133)
    z.loc[outliers].plot.hist(color='r', bins=50, ax=ax_133)
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('z')

    plt.savefig(pic_file)
    plt.close()
    return outliers,R2_score,mse_score
def ridge_method(filename,logger,pic_path,processed_path,sect_type,alpha):
    sect_type = sect_type.lower()
    assert sect_type.lower() in ['normal','mean']
    all_data = pd.read_excel(filename)
    a_list = ['a%d' % k for k in range(1, 9)]
    selected_items = ['目标客户编号', '品牌类型'] + a_list
    data = all_data.loc[:, selected_items]
    type_ids = ['1', '2', '3']
    ridge_item = []
    tp_data = all_data.loc[:,a_list]
    std_list = tp_data.std().values
    mean_list = tp_data.mean().values
    new_data = all_data.copy(deep=True)

    all_ridge_data = []
    for tp_id in type_ids:
        type_id_data = data[data['品牌类型'] == int(tp_id)]
        out_preprocess = type_id_data.loc[:, ['目标客户编号']+a_list]

        for k_id in range(len(a_list)):
            item = a_list[k_id]
            tmp_list = a_list.copy()
            index = tmp_list.index(item)
            del tmp_list[index]
            X_train = out_preprocess.loc[:,tmp_list]
            y_train = out_preprocess.loc[:,item]
            index_ids = out_preprocess.loc[:,'目标客户编号']
            ridge_path= os.path.join(pic_path,'ridge')
            if not os.path.exists(ridge_path):
                os.makedirs(ridge_path)
            pic_file = os.path.join(ridge_path,"ridge_%s_type_%s"%(item,tp_id))
            logger.info("******************process type:%s,value:%s***********************"%(tp_id,item))
            outliers,R2_score,mse_score = find_outliers(Ridge(alpha=alpha),X_train,y_train,logger,pic_file)
            tmp = [tp_id,item,R2_score,mse_score]
            all_ridge_data.append(tmp)
            for ids_item in index_ids[outliers].values:
                out_item = all_data[all_data['目标客户编号']==ids_item]
                index = new_data[new_data['目标客户编号']==ids_item].index
                ridge_item.append(out_item)
                if sect_type=='normal':
                    new_data.loc[index,item] = np.random.normal(mean_list[k_id],std_list[k_id])
                elif sect_type=='mean':
                    new_data.loc[index,item] = mean_list[k_id]
    ridge_file = os.path.join(processed_path,'ridge','ridge_find.xlsx')
    filter_file = os.path.join(processed_path, 'filter', 'filter_data2.xlsx')
    ridge_score_file = os.path.join(processed_path, 'ridge', 'ridge_score.xlsx')
    pd.concat(ridge_item).to_excel(ridge_file,index=False)
    new_data.to_excel(filter_file, index=False)
    pd.DataFrame(all_ridge_data).to_excel(ridge_score_file,index=False,header=['目标客户编号','评价指标','R2系数','MSE总差'])
