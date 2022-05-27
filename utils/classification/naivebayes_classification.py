"""
 * description:贝叶斯分类器
 * date: 2022/05/24/20:47:00
 * author: xinyu
 * version: 1.0
"""
import os
from pyhanlp import SafeJClass
from utils.utils.doload import ensure_data
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


def predict(new_file_name,new_file_content,classifier):
    # print("《%16s》\t属于分类\t【%s】" % (text, classifier.classify(text)))
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
# def new_file():
    # if count<0:
    #     return "完成"
    # else:
    #     print(new_file_name[count])
    #     print(new_file_content[count])
    #     print(count)
    #     count=count-1
    #     new_file(new_file_name,new_file_content,count)
    #RecursionError: maximum recursion depth exceeded while calling a Python object 递归次数不能超过1000次


# if __name__ == '__main__':
#     classifier = train_or_load_classifier()
#
#     # predict(classifier, "C罗获2018环球足球奖最佳球员 德尚荣膺最佳教练")
#     # predict(classifier, "英国造航母耗时8年仍未服役 被中国速度远远甩在身后")
#     # predict(classifier, "研究生考录模式亟待进一步专业化")
#     # predict(classifier, "如果真想用食物解压,建议可以食用燕麦")
#     # predict(classifier, "通用及其部分竞争对手目前正在考虑解决库存问题")
