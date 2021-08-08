import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file",type=str,default="附录1 目标客户体验数据.xlsx",help="raw data file.")
    parser.add_argument("--test-file", type=str, default="附录3 待判定的数据.xlsx", help="raw data file.")
    parser.add_argument("--processed-path", type=str, default="./processed", help="normal data file.")
    parser.add_argument("--pics-path", type=str, default="./pictures", help="picture path.")
    parser.add_argument("--log-path", type=str, default="./log", help="logging path.")
    parser.add_argument("--sect-type", type=str, default="normal", help="type of filling the abnormal data.")
    parser.add_argument("--vec-type", type=str, default="one-hot",
                        help="The method of vectorizing the discrete data,including one-hot,dummy,effect and embedding.")
    parser.add_argument("--alpha", type=float, default=1.0, help="ridge data of alpha.")
    parser.add_argument("--train-batch-size", type=int, default=400, help="batch size of training data.")
    parser.add_argument("--test-batch-size", type=int, default=400, help="batch size of test data.")
    parser.add_argument("--train-times", type=int, default=3000, help="training times of data.")
    parser.add_argument("--percentage", type=float, default=0.80, help="ridge data of alpha.")
    parser.add_argument("--do-first", action='store_true',help="The first problem solving.")
    parser.add_argument("--do-second", action='store_true',help="The second problem solving.")
    parser.add_argument("--do-third", action='store_true',help="The third problem solving.")
    parser.add_argument("--cuda", action='store_false', help="Training model by cuda.")
    args = parser.parse_args()
    return args
def save_args(args,logger):
    save_args_file = os.path.join(args.log_path, "args.txt")
    line = str(args)
    with open(save_args_file, mode="w", encoding="utf-8") as wfp:
        wfp.write(line + "\n")
    logger.info("Args saved in file%s" % save_args_file)
def check_path(args):
    assert os.path.exists(args.raw_file)
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    if not os.path.exists(args.processed_path):
        os.mkdir(args.processed_path)
