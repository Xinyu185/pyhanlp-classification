"""
 * description:
 * date: 2022/05/27/09:16:00
 * author: xinyu
 * version: 1.0
"""

from pyhanlp import *

ClusterAnalyzer = JClass('com.hankcs.hanlp.mining.cluster.ClusterAnalyzer')
import pandas as pd


#得到kemeans算法对象
def train_or_load_classifier():
    return ClusterAnalyzer()


# 搜狗测试
def predict(classifier,class_size,path):
    dir = os.listdir(path)
    print(dir)
    file_name = ''
    for i in dir:
        paths = path + "\\" + i
        dirs = os.listdir(paths)[0:5]
        for j in dirs:
            file_path = paths + "\\" + j
            file_name = i + "-" + j
            # print(file_path)
            """
            在这加一步转码操作......
            """

            try:
                file = open(file_path, "r",encoding="utf-8")
                data = file.read()
                # datas = pd.read_csv(file_path,error_bad_lines=False,sep="\t",header=None,encoding = 'gb2312',warn_bad_lines=True)
                # print(data)
                classifier.addDocument(file_name, data)
            except:
                try:
                    file = open(file_path, "r",encoding="gbk")
                    data = file.read()
                    # datas = pd.read_csv(file_path,error_bad_lines=False,sep="\t",header=None,encoding = 'gb2312',warn_bad_lines=True)
                    # print(data)
                    classifier.addDocument(file_name, data)
                except:
                    print("...")
    # 指定分类个数
    # print(analyzer.kmeans(5))
    # print( len(analyzer.kmeans(5)))
    # 自动分类个数
    # print(classifier.repeatedBisection(1.0))
    # print(len(classifier.repeatedBisection(1.0)))
    return classifier.kmeans(class_size)

