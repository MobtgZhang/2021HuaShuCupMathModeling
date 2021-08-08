import os
from src.utils import statical_data_a_value,statical_data_b_value
from src.data import filter_split_file,filter_file,create_dataset,create_all_dataset,create_test_dataset
from src.data import ridge_method,kde_method,pca_method

from src.model import trainer
from src.log import Logger
from config import save_args,parse_args,check_path
def first_problem(args,logger):
    # 异常B7数据修正以及a数据修正
    filter_path = os.path.join(args.processed_path, 'filter')
    if not os.path.exists(filter_path):
        os.makedirs(filter_path)
    filter_filename1 = os.path.join(filter_path,"filter_data1.xlsx")
    if not os.path.exists(filter_filename1):
        filter_file(args.raw_file, filter_path)
        logger.info("Saved file:%s" % filter_filename1)
    # ridge 数据修正
    # 岭回归和KDE检测
    ridge_pics_path = os.path.join(args.pics_path, 'ridge')
    ridge_processed_path = os.path.join(args.processed_path, 'ridge')
    kde_path = os.path.join(args.pics_path, 'kde')
    if not os.path.exists(ridge_pics_path):
        os.makedirs(ridge_pics_path)
        if not os.path.exists(ridge_processed_path):
            os.makedirs(ridge_processed_path)
        ridge_method(filter_filename1,logger,args.pics_path,args.processed_path,args.sect_type,args.alpha)
        logger.info("Saved path:%s" % ridge_pics_path)
        logger.info("Saved path:%s" % ridge_processed_path)
    filter_filename2 = os.path.join(filter_path,"filter_data2.xlsx")
    if not os.path.exists(kde_path):
        os.makedirs(kde_path)
        kde_method(filter_filename2, logger, args.pics_path)
        logger.info("Saved path:%s" % kde_path)
    # 将数据分为三个车型数据集
    filter_type_path = os.path.join(args.processed_path, 'filter_type')
    if not os.path.exists(filter_type_path):
        os.makedirs(filter_type_path)
        filter_split_file(filter_filename2,filter_type_path)
        logger.info("Saved path:%s" % filter_type_path)
    # 统计数据信息
    if not os.path.exists(args.pics_path):
        os.mkdir(args.pics_path)
        logger.info("Created the path:%s" % args.pics_path)
    a_pic_path = os.path.join(args.pics_path, "a_value")
    b_pic_path = os.path.join(args.pics_path, "b_value")
    if not os.path.exists(a_pic_path):
        os.makedirs(a_pic_path)
        logger.info("Created paths:%s" % a_pic_path)
        statical_data_a_value(filter_filename2,a_pic_path)
        logger.info("Saved pics:%s" % a_pic_path)
    if not os.path.exists(b_pic_path):
        logger.info("Created paths:%s" % b_pic_path)
        os.makedirs(b_pic_path)
        statical_data_b_value(filter_filename2,b_pic_path)
        logger.info("Saved pics:%s" % b_pic_path)
def second_problem(args,logger):
    filter_path = os.path.join(args.processed_path, 'filter')
    filter_filename2 = os.path.join(filter_path, "filter_data2.xlsx")
    pca_path = os.path.join(args.pics_path, "pca")
    if not os.path.exists(pca_path):
        pca_method(filter_filename2, args.pics_path, args.processed_path)
        logger.info("Saved pics:%s" % pca_path)
def third_problem(args,logger):
    filter_path = os.path.join(args.processed_path, 'filter')
    filter_filename2 = os.path.join(filter_path, "filter_data2.xlsx")
    # 创建数据集
    split_path = os.path.join(args.processed_path, 'train')
    if not os.path.exists(split_path):
        os.makedirs(split_path)
        create_dataset(filter_filename2,args.processed_path,args.percentage)
        logger.info("Saved path:%s" % split_path)
    dataset_file = os.path.join(split_path,'dataset.npz')
    if not os.path.exists(dataset_file):
        create_all_dataset(filter_filename2, args.processed_path, args.percentage)
        logger.info("Saved path:%s" % dataset_file)
    test_dataset_file = os.path.join(split_path,'test_dataset.npz')
    if not os.path.exists(test_dataset_file):
        create_test_dataset(args.test_file, args.processed_path)
        logger.info("Saved path:%s" % test_dataset_file)
    # 创建pytorch trainer 进行数据集训练
    results_path = os.path.join(args.processed_path, "results")
    model_file = os.path.join(results_path,'model.ckpt')
    if not os.path.exists(model_file):
        estimator = trainer(args)
        estimator.run()
        # 输出对应的数值信息
        estimator.predict()
        logger.info("Saved path:%s" % results_path)

def main():
    args = parse_args()
    logger = Logger(args)
    check_path(args)
    if args.do_first:
        first_problem(args,logger)
    if args.do_second:
        second_problem(args, logger)
    if args.do_third:
        third_problem(args, logger)
    save_args(args, logger)
if __name__ == '__main__':
    main()
