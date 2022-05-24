"""
 * description:
 * date: 2022/05/24/20:47:00
 * author: xinyu
 * version: 1.0
"""
import pandas as pd
import os
from utils.classification.naivebayes_classification import train_or_load_classifier,predict #贝叶斯分类
# from svm_classification import train_or_load_classifier,predict #svm分类
#导入数据
def import_data(path):
    datas = pd.read_csv(path,encoding = 'utf-8',header=None,sep='\t',error_bad_lines=False)
    #,error_bad_lines=False必须加上此参数，否则会报如下错误
    #pandas.errors.ParserError: Error tokenizing data. C error: Expected 1 fields in line 14, saw 3
    return datas[0].values,datas[1].values
#创建文件夹
def new_folder():
    project_path = os.path.dirname(os.path.realpath(__file__))  # 获取项目路径
    news_path = os.path.join(project_path, 'data')  # 新闻数据存放目录路径
    if not os.path.exists(news_path):  # 创建data文件夹
        os.mkdir(news_path)

#创建文件
# def new_file(new_file_name,new_file_content,count):
    # if count<0:
    #     return "完成"
    # else:
    #     print(new_file_name[count])
    #     print(new_file_content[count])
    #     print(count)
    #     count=count-1
    #     new_file(new_file_name,new_file_content,count)
    #RecursionError: maximum recursion depth exceeded while calling a Python object 递归次数不能超过1000次
def new_file(new_file_name,new_file_content,classifier):
    for i in range(len(new_file_name)):
        classes = predict(classifier, new_file_content[i])
# 判断是否存在该文件夹，不存在则创建
        if not os.path.exists("data\\{}\\".format(classes)):  # 创建文件夹
            os.mkdir("data\\{}\\".format(classes))
        # w的意思:打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件
        try:
            with open(r"data\{}\{}.txt".format(classes,i), "w") as op:
                op.write(new_file_content[i])
                op.close()
        except:
            #编码错误的 跳过 并打印出来
            print(new_file_content[i])


if __name__ == '__main__':
    new_folder()
    files_name,files_content = import_data("data.txt")
    classifier =  train_or_load_classifier()
    # count = len(files_name)-1
    new_file(files_name,files_content,classifier)

