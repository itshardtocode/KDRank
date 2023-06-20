"""
@Author: dejuzhang
@Software: PyCharm
@Time: 2022/10/14 14:10
"""
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime
from metric import *

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def set_seed(seed):
    np.random.seed(seed)
    gpu = tf.config.list_physical_devices("GPU")
    if gpu:
        tf.random.set_seed(seed)

def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")

def custom_loss(args, model, x):
    user_emb, p_emb, n_emb = model(x, args.is_train)
    pos_size = args.batch_size
    neg_size = args.batch_size * args.neg_sample_ratio
    gt = tf.constant([1] * pos_size + [0] * neg_size,
                     dtype=tf.float32,
                     shape=[pos_size + neg_size])

    user_features = tf.tile(user_emb, multiples=[args.neg_sample_ratio + 1, 1])

    poi_features = tf.concat([p_emb, n_emb], axis=0)
    logits = tf.reduce_sum(tf.multiply(user_features, poi_features), axis=-1)

    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=gt, logits=logits, name="ce_loss"))

    losses = tf.reduce_sum(model.losses)

    reg_loss = tf.reduce_sum([loss + losses])

    return reg_loss


def evaluate(gt, pred_scores, k_list):
    eval_dict = dict()
    pred_scores[:, 0] = 1e9
    # sorting
    pred_ranking = np.flip(np.argsort(pred_scores, axis=1), axis=1).tolist()

    precision_dict = {}
    recall_dict = {}
    for k in k_list:
        precision_dict[k] = precision_at_k(gt, pred_ranking, k)
        recall_dict[k] = recall_at_k(gt, pred_ranking, k)
        eval_dict[k] = {
            "Precision": precision_dict[k],
            "Recall": recall_dict[k]
        }
    return eval_dict


def Test(args, model, data_loader, poi_geo_inf, user_poi_mat):
    test_user, target = data_loader.get_test_valid_dataset()
    scores = []
    bs = args.batch_size
    for i in tqdm(range(len(test_user) // bs + 1)):
        if i * bs >= len(test_user):
            break

        test_bu = test_user[i * bs: min((i + 1) * bs, len(test_user))]
        test_local, test_global = data_loader.get_user_graphs(test_bu)
        test_ukg = data_loader.get_user_kg_embed(test_bu)
        inputs = {
            "poi_geo_inf": poi_geo_inf,
            "user_poi_mat": user_poi_mat,
            "batch_local": test_local,
            "batch_global": test_global,
            "batch_user_kg": test_ukg,
            "batch_user": np.array(test_bu),
        }
        batch_scores = model(inputs, training=False)
        scores.append(batch_scores)

    scores = np.concatenate(scores, axis=0)
    assert scores.shape[0] == len(target), "[evaluate] sizes of scores and ground truth don't match"
    cprint("Evaluation...")

    eval_msg = evaluate(target, scores, args.candidate_k)

    return eval_msg
