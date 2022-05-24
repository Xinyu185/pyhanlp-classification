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


def predict(classifier, text):
    # print("《%16s》\t属于分类\t【%s】" % (text, classifier.classify(text)))
    # print(classifier.classify(text))
    return classifier.classify(text)
    # 如需获取离散型随机变量的分布，请使用predict接口
    # print("《%16s》\t属于分类\t【%s】" % (text, classifier.predict(text)))

