"""
@Author: dejuzhang
@Software: PyCharm
@Time: 2022/10/14 14:41
"""

import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize
import pickle
from utils import load_pkl
import tensorflow as tf


class DataLoader:
    def __init__(self, args):
        self.args = args
        self.negative_ratio = args.neg_sample_ratio
        graph_dir = args.context_dir.format(args.dataset)
        train_test_dir = args.train_test_dir.format(args.dataset)
        self.local_feature = load_npz(graph_dir + "local_ppr.npz")
        self.global_feature = load_npz(graph_dir + "global_sim.npz")

        self.train_pos = pd.read_csv(train_test_dir + "train_pos.csv").values
        self.user_kg_embed = load_pkl(graph_dir + 'user_embed_dim_1000_k_2.pkl')

        with open(train_test_dir + "train_neg.pkl", 'rb') as f:
            self.train_neg = pickle.load(f)
            f.close()

        with open(train_test_dir + "test_instances.pkl", 'rb') as f:
            self.test_instances = pickle.load(f)
            f.close()

        self.set_to_dataset = {
            "train": self.train_pos,
            "test": self.test_instances,
        }

        
    def get_train_batch_iterator(self):
        neg_sample_func = lambda x: np.random.choice(self.train_neg[x], size=self.negative_ratio, replace=True)

        batch_size = self.args.batch_size
        total_batch = len(self.train_pos) // batch_size
        for i in range(total_batch):
            batch = self.train_pos[i * batch_size: (i + 1) * batch_size]
            batch_users = batch[:, 0]
            batch_items_pos = batch[:, 1]
            batch_items_neg = np.array([neg_sample_func(x) for x in batch_users]).flatten("F")

            yield (i, batch_users, batch_items_pos, batch_items_neg)

    def get_user_poi_adj_mat(self):
        """加载poi和user的邻接矩阵"""
        in_file = "./data/cities/{:}/city_user_poi_adj.npz".format(self.args.dataset)
        sp_mat = load_npz(in_file)  # coo
        sp_mat = normalize(sp_mat, norm="l1", axis=1)  # after normalize coo -> csr
        sp_mat = sp_mat.tocoo()  # tor(10790,10121)
        return sp_mat.data, (sp_mat.row, sp_mat.col), sp_mat.shape


    def get_user_graphs(self, user_array):

        lf_mat = self.local_feature[user_array]
        gf_mat = self.global_feature[user_array]
        return lf_mat.toarray(), gf_mat.toarray()


    def get_test_valid_dataset(self):
        user_id_list = list(self.test_instances.keys())
        gt_list = [self.test_instances[x].tolist() for x in user_id_list]

        return user_id_list, gt_list

    def get_poi_geo_inf(self):
        in_file = "./data/cities/{:}/business_influence_scores.csv".format(self.args.dataset)
        df = pd.read_csv(in_file, header=None)
        return tf.convert_to_tensor(df.values, dtype=tf.float32)

    def get_user_kg_embed(self, user_array):
        # user_array = user_array.cpu().numpy()
        return self.user_kg_embed[user_array]

    def get_user_poi_mat(self):
        up_data, (up_row, up_col), up_shape = self.get_user_poi_adj_mat()
        indices_list = list(zip(list(up_row), list(up_col)))
        sp_user_poi_mat = tf.SparseTensor(indices_list, list(up_data), up_shape)
        return tf.sparse.to_dense(sp_user_poi_mat, default_value=0)

