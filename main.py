"""
 * description:
 * date: 2022/05/24/20:47:00
 * author: xinyu
 * version: 1.0
"""
import pandas as pd
import os
# from utils.classification.naivebayes_classification import train_or_load_classifier,predict#贝叶斯分类
# from utils.classification.svm_classification import train_or_load_classifier,predict #svm分类
from utils.classification.kmeans_classification import train_or_load_classifier,predict
from utils.utils.doload import ensure_data

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
if __name__ == '__main__':
    sogou_corpus_path = ensure_data('搜狗文本分类语料库迷你版','http://file.hankcs.com/corpus/sogou-text-classification-corpus-mini.zip')
    new_folder()
    files_name,files_content = import_data("./test_data/allinone/data.txt")
    classifier =  train_or_load_classifier()
    class_array = predict(classifier,10,"test_data/文本分类语料库")
    print(class_array)

