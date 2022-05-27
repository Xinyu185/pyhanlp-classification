"""
 * description:
 * date: 2022/05/27/09:16:00
 * author: xinyu
 * version: 1.0
"""

from pyhanlp import *

ClusterAnalyzer = JClass('com.hankcs.hanlp.mining.cluster.ClusterAnalyzer')

def kmeans_classification():
    return ClusterAnalyzer()


# if __name__ == '__main__':
#     analyzer = ClusterAnalyzer()
#     analyzer.addDocument("赵一", "流行, 流行, 流行, 流行, 流行, 流行, 流行, 流行, 流行, 流行, 蓝调, 蓝调, 蓝调, 蓝调, 蓝调, 蓝调, 摇滚, 摇滚, 摇滚, 摇滚")
#     print(analyzer.kmeans(3))
#     print(analyzer.repeatedBisection(3))
#     print(analyzer.repeatedBisection(1.0))  # 自动判断聚类数量k
