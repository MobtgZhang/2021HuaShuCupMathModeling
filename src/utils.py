import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def process_a_value(out_preprocess,pic_path,type_id):
    a_list = ['a1','a2','a3','a4','a5','a6','a7','a8']
    type_list = ['电池技术性能满意度得分','舒适性整体表现满意度得分','经济性整体满意度得分',
                 '安全性表现整体满意度得分','动力性表现整体满意度得分','驾驶操控性表现整体满意度得分',
                '外观内饰整体表现满意度得分','配置与质量品质整体满意度得分']
    type_dict = {
        '1':'合资品牌',
        '2':'自主品牌',
        '3':'新势力品牌'
    }
    for alpha,beta in zip(a_list,type_list):
        out = out_preprocess[alpha].values
        num1 = len(np.where((out > 0) & (out < 60))[0])
        num2 = len(np.where((out > 60) & (out < 80))[0])
        num3 = len(np.where((out > 80) & (out < 100))[0])
        values = [num1,num2,num3]
        labels = ['不满意','基本满意','比较满意']
        # 饼状图
        explode = [0]*len(values)
        index = values.index(max(values))
        explode[index] = 0.05
        plt.figure(figsize=(8,6))
        plt.pie(values, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=150)
        plt.title("%s中%s 占比图"%(type_dict[type_id],beta))
        fig_file = os.path.join(pic_path,"%s_pie_typeid_%s.png"%(alpha,type_id))
        plt.savefig(fig_file)
        plt.close()
        # 条形图
        plt.figure(figsize=(12, 6))
        plt.bar(labels, values, align='center', alpha=0.8, width=0.8)
        plt.xlabel('分数值区间')
        plt.ylabel('数量')
        plt.title('%s的%s统计图' %(type_dict[type_id],beta))
        save_fig = os.path.join(pic_path, '%s_bar_typeid_%s.png'%(alpha,type_id))
        plt.savefig(save_fig)
        plt.close()
def statical_data_a_value(filename,pic_path):
    all_data = pd.read_excel(filename)
    data = all_data.loc[:,['目标客户编号','品牌类型','a1','a2','a3','a4','a5','a6','a7','a8',]]
    a_list = ['a1','a2','a3','a4','a5','a6','a7','a8']
    # 对于品牌一进行筛选
    type_id1 = data[data['品牌类型'] == 1]
    out_preprocess = type_id1.loc[:,a_list]
    process_a_value(out_preprocess,pic_path,'1')
    # 对于品牌二进行筛选
    type_id2 = data[data['品牌类型'] == 2]
    out_preprocess = type_id2.loc[:,a_list]
    process_a_value(out_preprocess, pic_path, '2')
    # 对于品牌三进行筛选
    type_id3 = data[data['品牌类型'] == 3]
    out_preprocess = type_id3.loc[:,a_list]
    process_a_value(out_preprocess, pic_path, '3')
def single_draw_bar(labels, values, xlabel,title,save_fig,figsize=(18,12)):
    plt.figure(figsize=figsize)
    plt.bar(labels, values, align='center', alpha=0.8, width=0.8)
    plt.xlabel(xlabel)
    plt.ylabel('数量')
    plt.title(title)
    plt.savefig(save_fig)
    plt.close()
def single_draw_pie(labels, values,title,save_fig):
    # 饼状图
    explode = [0] * len(values)
    index = values.index(max(values))
    explode[index] = 0.1
    plt.figure(figsize=(10,6))
    plt.pie(values, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=150)
    plt.title(title)
    plt.savefig(save_fig)
    plt.close()
def process_b_value(out_preprocess,pic_path,type_id):
    type_dict = {
        '1': '合资品牌',
        '2': '自主品牌',
        '3': '新势力品牌'
    }
    # 户口情况统计图
    # 1 表示户口在老家；2表示户口在本城市，3表示其他
    house_regist_a = len(out_preprocess[out_preprocess['B1'] == 1])
    house_regist_b = len(out_preprocess[out_preprocess['B1'] == 2])
    house_regist_c = len(out_preprocess[out_preprocess['B1'] == 3])
    labels = ['户口在老家','户口在本城市','其他']
    values = [house_regist_a,house_regist_b,house_regist_c]
    xlabel = '户口类型'
    title_bar = '%s中户口情况统计图'%type_dict[type_id]
    title_pie = '%s中户口情况占比图' % type_dict[type_id]
    draw_dispersed_data(values,labels,'B1',type_id,pic_path,xlabel,title_bar,title_pie)
    # 居住时长统计图
    xlabel = '居住时长'
    title_bar = '%s中居住时长统计图' % type_dict[type_id]
    title_pie = '%s中居住时长占比图' % type_dict[type_id]
    draw_continual_data(out_preprocess,'B2',type_id,pic_path,xlabel,title_bar,title_pie,0,70,10)

    # 居住区域统计图
    # 1 表示市中心，2表示非市中心的城区，3表示城乡结合部，4表示县城，5表示乡镇中心地带，6表示农村
    value1 = len(out_preprocess[out_preprocess['B3'] == 1])
    value2 = len(out_preprocess[out_preprocess['B3'] == 2])
    value3 = len(out_preprocess[out_preprocess['B3'] == 3])
    value4 = len(out_preprocess[out_preprocess['B3'] == 4])
    value5 = len(out_preprocess[out_preprocess['B3'] == 5])
    value6 = len(out_preprocess[out_preprocess['B3'] == 6])
    labels = ['市中心','非市中心的城区','城乡结合部','县城','乡镇中心地带','农村']
    values = [value1,value2,value3,value4,value5,value6]
    xlabel = '居住区域类型'
    title_bar = '%s中居住区域统计图' % type_dict[type_id]
    title_pie = '%s中居住区域占比图' % type_dict[type_id]
    draw_dispersed_data(values, labels, 'B3', type_id, pic_path, xlabel, title_bar, title_pie)
    # 驾驶年龄
    xlabel = '驾龄'
    title_bar = '%s中驾龄统计图' % type_dict[type_id]
    title_pie = '%s中驾龄占比图' % type_dict[type_id]
    draw_continual_data(out_preprocess, 'B4', type_id, pic_path, xlabel, title_bar, title_pie, 0,40,10)
    # 家庭人口数量
    value1 = len(out_preprocess[out_preprocess['B5'] == 1])
    value2 = len(out_preprocess[out_preprocess['B5'] == 2])
    value3 = len(out_preprocess[out_preprocess['B5'] == 3])
    value4 = len(out_preprocess[out_preprocess['B5'] == 4])
    value5 = len(out_preprocess[out_preprocess['B5'] == 5])
    value6 = len(out_preprocess[out_preprocess['B5'] == 6])
    labels = ['%d人'%k for k in range(1,7)]
    values = [value1, value2, value3, value4, value5, value6]
    xlabel = '家庭人口数量'
    title_bar = '%s中家庭人口数量统计图' % type_dict[type_id]
    title_pie = '%s中家庭人口数量占比图' % type_dict[type_id]
    draw_dispersed_data(values, labels, 'B5', type_id, pic_path, xlabel, title_bar, title_pie)
    # 婚姻家庭情况
    # 1表示“未婚，单独居住”，2表示“未婚，与父母同住”，
    # 3表示“已婚/同居无子女（两人世界）”，4表示“已婚/同居无子女（与父母同住）”，
    # 5表示“已婚，有小孩，不与父母同住”，6表示“已婚，有小孩，与父母同住”，7表示“离异/丧偶”，8表示“其他”。
    value1 = len(out_preprocess[out_preprocess['B6'] == 1])
    value2 = len(out_preprocess[out_preprocess['B6'] == 2])
    value3 = len(out_preprocess[out_preprocess['B6'] == 3])
    value4 = len(out_preprocess[out_preprocess['B6'] == 4])
    value5 = len(out_preprocess[out_preprocess['B6'] == 5])
    value6 = len(out_preprocess[out_preprocess['B6'] == 6])
    value7 = len(out_preprocess[out_preprocess['B6'] == 7])
    value8 = len(out_preprocess[out_preprocess['B6'] == 8])
    labels = ['未婚，单独居住', '未婚，与父母同住', '已婚/同居无子女', '已婚/同居无子女', '已婚，有小孩，不与父母同住',
              '已婚，有小孩，与父母同住','离异/丧偶','其他']
    values = [value1, value2, value3, value4, value5, value6,value7,value8]
    xlabel = '婚姻家庭情况'
    title_bar = '%s中婚姻家庭情况统计图' % type_dict[type_id]
    title_pie = '%s中婚姻家庭情况占比图' % type_dict[type_id]
    draw_dispersed_data(values, labels, 'B6', type_id, pic_path, xlabel, title_bar, title_pie)
    # 家庭中孩子数量
    out_data = out_preprocess['B7'].values
    num1 = len(np.where(out_data == 1)[0])
    num2 = len(np.where(out_data == 2)[0])
    num3 = len(np.where(out_data == 0)[0])
    labels = ['1个孩子','2个孩子','NULL']
    values = [num1,num2,num3]
    xlabel = '家庭孩子数量'
    title_bar = '%s中家庭孩子数量统计图' % type_dict[type_id]
    title_pie = '%s中家庭孩子数量占比图' % type_dict[type_id]
    draw_dispersed_data(values, labels, 'B7', type_id, pic_path, xlabel, title_bar, title_pie)
    # 出生日期
    values = []
    labels = []
    out_data = out_preprocess['B8'].values
    for num in range(1900,2020,10):
        idex = len(np.where((out_data > num) & (out_data < num + 10))[0])
        if idex != 0:
            tp_num = num%1900 if num<2000 else num%2000
            labels.append('%d后' % (tp_num))
            values.append(idex)
    xlabel = '出生日期'
    title_bar = '%s中出生日期统计图' % type_dict[type_id]
    title_pie = '%s中出生日期占比图' % type_dict[type_id]
    draw_dispersed_data(values, labels, 'B8', type_id, pic_path, xlabel, title_bar, title_pie)
    # 最高学历
    # 1表示“未受过正式教育”，2表示“小学”，2表示“初中”，4表示“高中/中专/技校”，5表示“大专”，6表示“本科”，8表示“双学位/研究生及以上”。
    value1 = len(out_preprocess[out_preprocess['B9'] == 1])
    value2 = len(out_preprocess[out_preprocess['B9'] == 2])
    value3 = len(out_preprocess[out_preprocess['B9'] == 3])
    value4 = len(out_preprocess[out_preprocess['B9'] == 4])
    value5 = len(out_preprocess[out_preprocess['B9'] == 5])
    value6 = len(out_preprocess[out_preprocess['B9'] == 6])
    value7 = len(out_preprocess[out_preprocess['B9'] == 7])
    value8 = len(out_preprocess[out_preprocess['B9'] == 8])
    labels = ['未受过正式教育','小学','初中','高中/中专/技校','大专','本科','双学位/研究生及以上']
    values = [value1, value2, value3, value4, value5, value6, value7, value8]
    xlabel = '最高学历'
    title_bar = '%s中最高学历情况统计图' % type_dict[type_id]
    title_pie = '%s中最高学历情况占比图' % type_dict[type_id]
    draw_dispersed_data(values, labels, 'B9', type_id, pic_path, xlabel, title_bar, title_pie)
    # 工作年限
    xlabel = '工作年限'
    title_bar = '%s中工作年限统计图' % type_dict[type_id]
    title_pie = '%s中工作年限占比图' % type_dict[type_id]
    draw_continual_data(out_preprocess, 'B10', type_id, pic_path, xlabel, title_bar, title_pie, 0, 40, 10)
    # 所在单位性质
    # 1="机关单位/政府部门/基层组织"，2="科研/教育/文化/卫生/医疗等事业单位"，3="国有企业",4="私营/民营企业（雇员人数在8人以上）",
    # 5="外资企业",6="合资企业",7="个体户/小型公司（雇员人数在8人以下）",8="自由职业者",9="不工作"
    value1 = len(out_preprocess[out_preprocess['B11'] == 1])
    value2 = len(out_preprocess[out_preprocess['B11'] == 2])
    value3 = len(out_preprocess[out_preprocess['B11'] == 3])
    value4 = len(out_preprocess[out_preprocess['B11'] == 4])
    value5 = len(out_preprocess[out_preprocess['B11'] == 5])
    value6 = len(out_preprocess[out_preprocess['B11'] == 6])
    value7 = len(out_preprocess[out_preprocess['B11'] == 7])
    value8 = len(out_preprocess[out_preprocess['B11'] == 8])
    value9 = len(out_preprocess[out_preprocess['B11'] == 9])
    labels = ['机关单位/政府部门/基层组织', '科研/教育/文化/卫生/医疗等事业单位', '国有企业', '私营/民营企业', '外资企业', '合资企业', '个体户/小型公司','自由职业者','不工作']
    values = [value1, value2, value3, value4, value5, value6, value7, value8,value9]
    xlabel = '所在单位性质'
    title_bar = '%s中所在单位性质情况统计图' % type_dict[type_id]
    title_pie = '%s中所在单位性质情况占比图' % type_dict[type_id]
    draw_dispersed_data(values, labels, 'B11', type_id, pic_path, xlabel, title_bar, title_pie)
    # 职位
    # 1=高层管理者/企业主/老板，2=中层管理者，3=资深技术人员/高级技术人员，4=中级技术人员，5=初级技术人员
    # 6=资深职员/办事员，7=中级职员/办事员，8=初级职员/办事员，9=个体户/小型公司业主，10=自由职业者，11=其他
    value1 = len(out_preprocess[out_preprocess['B12'] == 1])
    value2 = len(out_preprocess[out_preprocess['B12'] == 2])
    value3 = len(out_preprocess[out_preprocess['B12'] == 3])
    value4 = len(out_preprocess[out_preprocess['B12'] == 4])
    value5 = len(out_preprocess[out_preprocess['B12'] == 5])
    value6 = len(out_preprocess[out_preprocess['B12'] == 6])
    value7 = len(out_preprocess[out_preprocess['B12'] == 7])
    value8 = len(out_preprocess[out_preprocess['B12'] == 8])
    value9 = len(out_preprocess[out_preprocess['B12'] == 9])
    value10 = len(out_preprocess[out_preprocess['B12'] == 10])
    value11 = len(out_preprocess[out_preprocess['B12'] == 11])
    labels = ['高层管理者/企业主/老板','中层管理者','资深技术人员/高级技术人员','中级技术人员','初级技术人员','资深职员/办事员',
              '中级职员/办事员','初级职员/办事员','个体户/小型公司业主','自由职业者','其他']
    values = [value1, value2, value3, value4, value5, value6, value7, value8, value9,value10,value11]

    xlabel = '职位'
    title_bar = '%s中职位情况统计图' % type_dict[type_id]
    title_pie = '%s中职位情况占比图' % type_dict[type_id]
    draw_dispersed_data(values, labels, 'B12', type_id, pic_path, xlabel, title_bar, title_pie)
    # 家庭年收入
    B_num_list = ['B13','B14','B15']
    B_name_num_list = ['家庭年收入', '个人年收入', '家庭可支配年收入']
    xlabel = "收入情况/万元"
    for num,name in zip(B_num_list,B_name_num_list):
        title_bar = '%s中%s统计图' % (type_dict[type_id],name)
        title_pie = '%s中%s占比图' % (type_dict[type_id],name)
        draw_continual_data(out_preprocess,num, type_id, pic_path, xlabel, title_bar, title_pie, 0, 100, 20)
    # 全年房贷的支出占家庭年总收入的比例
    # 全年车贷的支出占家庭年总收入的比例
    B_num_list = ['B16', 'B17']
    B_name_num_list = ['全年房贷的支出占家庭年总收入的比例', '全年车贷的支出占家庭年总收入的比例']
    xlabel = "百分比值"
    for num, name in zip(B_num_list, B_name_num_list):
        title_bar = '%s中%s统计图' % (type_dict[type_id], name)
        title_pie = '%s中%s占比图' % (type_dict[type_id], name)
        draw_continual_data(out_preprocess, num, type_id, pic_path, xlabel, title_bar, title_pie, 0, 100,10)
def draw_continual_data(out_preprocess,B_num_type,type_id,pic_path,
                        xlabel,title_bar,title_pie,start=0,end=100,step=20):
    values = []
    labels = []
    out_data = out_preprocess[B_num_type].values
    for num in range(start,end,step):
        idex = len(np.where((out_data > num) & (out_data < num + step))[0])
        if idex != 0:
            labels.append('%d~%d' % (num, num+step))
            values.append(idex)
    save_fig = os.path.join(pic_path, '%s_bar_type_%s.png' % (B_num_type,type_id))
    single_draw_bar(labels, values, xlabel, title_bar, save_fig)
    save_fig = os.path.join(pic_path, '%s_pie_type_%s.png' % (B_num_type,type_id))
    single_draw_pie(labels, values, title_pie, save_fig)
def draw_dispersed_data(values,labels,B_num_type,type_id,pic_path,
                        xlabel,title_bar,title_pie):
    out_data = dict(zip(labels, values))
    keys_list = []
    for key in out_data:
        if out_data[key] == 0:
            keys_list.append(key)
    for key in keys_list:
        del out_data[key]
    labels = list(out_data.keys())
    values = list(out_data.values())
    save_fig = os.path.join(pic_path, '%s_bar_type_%s.png' % (B_num_type,type_id))
    single_draw_bar(labels, values, xlabel, title_bar, save_fig)
    save_fig = os.path.join(pic_path, '%s_pie_type_%s.png' % (B_num_type,type_id))
    single_draw_pie(labels, values, title_pie, save_fig)
def statical_data_b_value(filename,pic_path):
    all_data = pd.read_excel(filename)
    b_list = ['B%d' % k for k in range(1,18)]
    selected_items = ['目标客户编号', '品牌类型'] + b_list
    data = all_data.loc[:, selected_items]
    type_ids = ['1','2','3']
    for tp_id in type_ids:
        type_id1 = data[data['品牌类型'] == int(tp_id)]
        out_preprocess = type_id1.loc[:, b_list]
        process_b_value(out_preprocess, pic_path, tp_id)