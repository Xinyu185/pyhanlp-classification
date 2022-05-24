"""
 * description:贝叶斯分类器
 * date: 2022/05/24/20:47:00
 * author: xinyu
 * version: 1.0
"""
import os
from pyhanlp import SafeJClass
from utils.utils.utility import ensure_data
#导入搜狗中文库资料后面作为训练集
sogou_corpus_path = ensure_data('搜狗文本分类语料库迷你版','http://file.hankcs.com/corpus/sogou-text-classification-corpus-mini.zip')

NaiveBayesClassifier = SafeJClass('com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier')
IOUtil = SafeJClass('com.hankcs.hanlp.corpus.io.IOUtil')



def train_or_load_classifier():
    model_path = sogou_corpus_path + '.ser'
    if os.path.isfile(model_path):
        return NaiveBayesClassifier(IOUtil.readObjectFrom(model_path))
    classifier = NaiveBayesClassifier()
    classifier.train(sogou_corpus_path)
    model = classifier.getModel()
    IOUtil.saveObjectTo(model, model_path)
    return NaiveBayesClassifier(model)


def predict(classifier, text):
    # print("《%16s》\t属于分类\t【%s】" % (text, classifier.classify(text)))
    return classifier.classify(text)
    # 如需获取离散型随机变量的分布，请使用predict接口
    # print("《%16s》\t属于分类\t【%s】" % (text, classifier.predict(text)))


if __name__ == '__main__':
    classifier = train_or_load_classifier()

    # predict(classifier, "C罗获2018环球足球奖最佳球员 德尚荣膺最佳教练")
    # predict(classifier, "英国造航母耗时8年仍未服役 被中国速度远远甩在身后")
    # predict(classifier, "研究生考录模式亟待进一步专业化")
    # predict(classifier, "如果真想用食物解压,建议可以食用燕麦")
    # predict(classifier, "通用及其部分竞争对手目前正在考虑解决库存问题")
