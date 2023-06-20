"""
@Author: dejuzhang
@Software: PyCharm
@Time: 2022/10/14 15:35
"""
import keras.layers
import tensorflow as tf


class KDRank(tf.keras.Model):
    def __init__(self, args):
        super(KDRank, self).__init__()
        self.args = args
        # self.dataloader = dataloader
        self.batch_size = args.batch_size
        self.nsr = args.neg_sample_ratio
        self.nsr = args.neg_sample_ratio
        self.reg_weight = args.reg_weight
        self.num_grids = args.num_gird * args.num_gird
        self.num_item = args.num_item
        self.num_user = args.num_user
        self.hidden_layers = args.hidden_layers
        self.hidden_dim = args.hidden_dim
        self.gat_heads = args.gat_heads
        self.training = args.is_train
        self.in_drop = args.gat_fn_dropout
        self.out_drop = args.gat_cm_dropout
        self.reg = tf.keras.regularizers.L2(l2=self.reg_weight)
        self.init = tf.keras.initializers.glorot_normal

        # self.poi_inf_mat = dataloader.load_poi_inf_mat()
        # self.user_poi_mat = dataloader.get_user_poi_mat()
        self.item_embedding = self.add_weight(name="item_embedding",
                                              shape=[self.num_item + 1, self.hidden_dim],
                                              dtype=tf.float32,
                                              initializer=self.init)

        self.local_context = LocalContext(self.hidden_layers, reg=self.reg, init=self.init)

        self.global_context = GlobalContext(self.hidden_dim, self.gat_heads, self.in_drop, self.out_drop, self.training)

        self.geo_embedding = self.add_weight(name="geo_embedding",
                                             shape=[self.num_user + 1, self.num_grids],
                                             dtype=tf.float32,
                                             initializer=self.init)

        self.knowledge = KnowledgeAgg(self.hidden_dim, self.init)

    def get_pos_neg_item_emb(self, data):
        """
        Args:
            data:

        Returns:

        """
        batch_pos = data["batch_pos"]
        batch_neg = data["batch_neg"]

        pos_item_embed = tf.nn.embedding_lookup(self.item_embedding, batch_pos)
        neg_item_embed = tf.nn.embedding_lookup(self.item_embedding, batch_neg)

        return pos_item_embed, neg_item_embed

    def get_scores(self, user_emb, poi_emb):
        """
        :param user_emb: user_embed
        :param poi_emb: self.item_embedding
        :return: x["poi_geo_inf"]
        """
        scores = tf.matmul(user_emb, poi_feat, transpose_b=True)

        return scores

    def call(self, x, training=True):
        upm, batch_global, batch_user, ukg = x["user_poi_mat"], x["batch_global"], x["batch_user"], x["batch_user_kg"]
        global_, _ = self.global_context(upm, batch_global, ukg, batch_user)
        local_ = self.local_context(x["batch_local"])
        user_geo_embed = tf.nn.embedding_lookup(self.geo_embedding, batch_user)
        user_embed = self.knowledge(local_, global_)
        user_embed = tf.concat([user_embed, user_geo_embed], axis=1, name="user_with_geo")
        if training:
            pos_item_embed, neg_item_embed = self.get_pos_neg_item_emb(x)

            return user_embed, pos_item_embed, neg_item_embed
        else:
            return self.get_scores(user_embed, self.item_embedding)


class LocalContext(tf.keras.Model):
    def __init__(self, units, reg=None, init=None):
        super(LocalContext, self).__init__()
        self.container = tf.keras.Sequential()
        for unit in units:
            self.container.add(tf.keras.layers.Dense(unit, activation='relu', use_bias=True,
                                                     kernel_regularizer=reg,
                                                     kernel_initializer=init))

    def call(self, inputs):
        return self.container(inputs)


class GlobalContext(tf.keras.Model):
    def __init__(self, unit, gat_heads, in_drop, out_drop, training):
        super(GlobalContext, self).__init__()
        self.unit = unit
        self.gat_heads = gat_heads
        self.in_drop = in_drop
        self.out_drop = out_drop
        self.training = training

        self.attn_ = AttenHead(unit=self.unit, in_drop=self.in_drop, out_drop=self.out_drop, training=self.training)

        self.out_layer = tf.keras.layers.Dense(self.unit, use_bias=False, activation='relu')

    def call(self, x, bias_mat, ukg, user_index):
        hidden_features = []
        attns = []
        bias_mat = -1e9 * (1 - bias_mat)
        for i in range(self.gat_heads):
            hid_feat, attn = self.attn_(x, bias_mat, ukg, user_index)
            hidden_features.append(hid_feat)
            attns.append(attn)

        h_1 = tf.concat(hidden_features, axis=-1)
        logits = self.out_layer(h_1)
        return logits, attns


class AttenHead(tf.keras.Model):
    def __init__(self, unit, in_drop, out_drop, residual=False, training=True):
        super(AttenHead, self).__init__()
        self.unit = unit
        self.in_drop = in_drop
        self.out_drop = out_drop
        self.residual = residual
        self.training = training
        self.in_dropout = tf.keras.layers.Dropout(self.in_drop)
        self.out_dropout = tf.keras.layers.Dropout(self.out_drop)
        self.b = self.add_weight('b', shape=[self.unit], initializer=tf.zeros_initializer())
        self.kg_fusion = KnowledgeFusion(self.unit)

        self.layer1 = tf.keras.layers.Dense(self.unit, use_bias=False)
        self.layer2 = tf.keras.layers.Dense(1)
        self.layer3 = tf.keras.layers.Dense(1)

    def call(self, x, bias_mat, ukg, user_index):
        user_feature = x
        if self.in_drop != 0.0:
            user_feature = self.in_dropout(x, training=self.training)
        """
        uhf: user hidden feature
        buhf: batch user hidden feature
        bukge: batch user knowledge graph embedding 
        """

        uhf = self.layer1(user_feature)
        buhf = tf.nn.embedding_lookup(uhf, user_index)
        bhkg = self.kg_fusion(ukg)
        buhf = bhkg + buhf

        f_1 = self.layer2(buhf)
        f_2 = self.layer3(uhf)

        logits = f_1 + tf.transpose(f_2)
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if self.out_drop != 0.0:
            coefs = self.out_dropout(coefs, training=self.training)
        if self.in_drop != 0.0:
            uhf = self.in_dropout(uhf, training=self.training)

        vals = tf.matmul(coefs, uhf)
        ret = tf.nn.relu(tf.nn.bias_add(vals, self.b))
        return ret, coefs


class KnowledgeFusion(tf.keras.layers.Layer):
    def __init__(self, init):
        super(KnowledgeFusion, self).__init__()
        self.init = init
        self.fusion = tf.keras.layers.Dense(self.init, activation='relu')

    def call(self, kg):
        return self.fusion(kg)


class KnowledgeAgg(tf.keras.Model):
    def __init__(self, unit, init):
        super(KnowledgeAgg, self).__init__()
        self.unit = unit
        self.init = init
        self.user_context = tf.keras.layers.Dense(self.unit, activation='relu')
        self.user_aware = tf.keras.layers.Dense(self.unit, activation='relu')
        self.agg_layer = tf.keras.layers.Dense(1,
                                               activation='relu',
                                               use_bias=False,
                                               kernel_initializer=self.init)

    def call(self, local_, global_):
        """
        :param local_: user ppr value
        :param global_: user global similarity
        :return:
        """
        user_context = self.user_context(local_)
        user_aware = self.user_aware(global_)

        knowledge_agg = [user_aware, user_context]

        user_emb = tf.stack(knowledge_agg, axis=1)

        user_hidden_emb = self.agg_layer(user_emb)

        user_hidden_emb = tf.nn.softmax(user_hidden_emb, axis=1)

        user_emb = tf.squeeze(tf.matmul(user_emb, user_hidden_emb, transpose_a=True))

        return user_emb
