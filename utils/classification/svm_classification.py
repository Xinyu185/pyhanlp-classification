"""
 * description:SVM线性分类器
 * date: 2022/05/24/20:47:00
 * author: xinyu
 * version: 1.0
"""
from pyhanlp.static import STATIC_ROOT, download

#导入搜狗中文库资料后面作为训练集
from utils.classification.naivebayes_classification import sogou_corpus_path
import os

def install_jar(name, url):
    dst = os.path.join(STATIC_ROOT, name)
    if os.path.isfile(dst):
        return dst
    download(url, dst)
    return dst


install_jar('text-classification-svm-1.0.2.jar', 'http://file.hankcs.com/bin/text-classification-svm-1.0.2.jar')
install_jar('liblinear-1.95.jar', 'http://file.hankcs.com/bin/liblinear-1.95.jar')
from pyhanlp import *

LinearSVMClassifier = SafeJClass('com.hankcs.hanlp.classification.classifiers.LinearSVMClassifier')
IOUtil = SafeJClass('com.hankcs.hanlp.corpus.io.IOUtil')


def train_or_load_classifier():
    model_path = sogou_corpus_path + '.svm.ser'
    if os.path.isfile(model_path):
        return LinearSVMClassifier(IOUtil.readObjectFrom(model_path))
    classifier = LinearSVMClassifier()
    classifier.train(sogou_corpus_path)
    model = classifier.getModel()
    IOUtil.saveObjectTo(model, model_path)
    return LinearSVMClassifier(model)


def predict(new_file_name,new_file_content,classifier):
    # print("《%16s》\t属于分类\t【%s】" % (text, classifier.classify(text)))
    # print(classifier.classify(text))
    for i in range(len(new_file_name)):
        classes = classifier.classify(new_file_content[i])
        # 判断是否存在该文件夹，不存在则创建
        if not os.path.exists("data\\{}\\".format(classes)):  # 创建文件夹
            os.mkdir("data\\{}\\".format(classes))
        # w的意思:打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件
        try:
            with open(r"data\{}\{}.txt".format(classes, i), "w") as op:
                op.write(new_file_content[i])
                op.close()
        except:
            # 编码错误的 跳过 并打印出来
            print(new_file_content[i])
    # 如需获取离散型随机变量的分布，请使用predict接口
    # print("《%16s》\t属于分类\t【%s】" % (text, classifier.predict(text)))

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