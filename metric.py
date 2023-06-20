"""
@Author: dejuzhang
@Software: PyCharm
@Time: 2022/10/16 15:17
"""

import numpy as np

def mapk(actual, predicted, k=10):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def precision_at_k(actual, predicted, k):
    """precision at k
    Args:
        actual - list of list
        predicted - list of list
    Return:
        precision@k
    """
    prec_at_k_list = [len(set(actual[i]) & set(predicted[i][:k])) / k
                      for i in range(len(actual))]
    result = sum(prec_at_k_list) / len(prec_at_k_list)
    return result


def recall_at_k(actual, predicted, k):
    """recall at k
    Args:
        actual - list of list
        predicted - list of list
    Return:
        recall@k
    """
    recall_at_k_list = [len(set(actual[i]) & set(predicted[i][:k])) / len(actual[i])
                        for i in range(len(actual))]
    return sum(recall_at_k_list) / len(recall_at_k_list)

def accuracy_at_k(actual, predicted, k):
    pass

def f1_at_k(actual, predicted, k):
    precision = precision_at_k(actual, predicted, k)
    recall = recall_at_k(actual, predicted, k)
    return 2 * (precision * recall) / (precision + recall)


