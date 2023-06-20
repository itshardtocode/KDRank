import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

import tensorflow as tf
import argparse
import time

from dataloader import DataLoader
from model import KDRank
import utils
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=20, help="number of epoch")
parser.add_argument("--batch_size", type=int, default=256, help="how many data be used each epoch")
parser.add_argument("--dataset", type=str, default="phi", help="which dataset")
parser.add_argument("--save", type=bool, default=False, help="whether to save model")
parser.add_argument('--neg_sample_ratio', type=int, default=3, help="negative sample ratio")
parser.add_argument("--random_seed", type=int, default=723, help="set random seed")
parser.add_argument("--is_train", type=bool, default=True, help="train or test")
parser.add_argument("--print_iter", type=int, default=200, help="how many batches to print")

parser.add_argument("--train_test_dir", type=str, default="./data/train_test/{:}/", help="train/test data")
parser.add_argument("--context_dir", type=str, default="./data/{:}/", help="predefined knowledge graph and data")
parser.add_argument("--pre_dir", type=str, default="./data/cities/{:}/", help="preprocess data")

parser.add_argument("--hidden_layers", default=[64, 48])
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--reg_weight", type=float, default=0.001)
parser.add_argument("--embedding_dim", type=int, default=32)
parser.add_argument("--hidden_dim", type=int, default=32)
parser.add_argument("--num_item", type=int, default=10493)
parser.add_argument("--num_user", type=int, default=15913)
parser.add_argument("--gat_heads", type=int, default=2)
parser.add_argument("--gat_fn_dropout", type=float, default=0.3)
parser.add_argument("--gat_cm_dropout", type=float, default=0.3)
parser.add_argument("--num_gird", type=int, default=30)
parser.add_argument("--candidate_k", default=[i * 10 for i in range(1, 11)])

args = parser.parse_args()


def main(config, data_loader):
    user_poi_mat = data_loader.get_user_poi_mat()
    poi_geo_inf = data_loader.get_poi_geo_inf()

    config.is_train = True

    model = KDRank(config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    for epoch in range(1, config.epoch + 1):
        iteration = data_loader.get_train_batch_iterator()
        cprint("Training...")
        for index, batch_user, batch_pos, batch_neg in tqdm(iteration):
            batch_local, batch_global = data_loader.get_user_graphs(batch_user)
            batch_user_kg = data_loader.get_user_kg_embed(batch_user)
            inputs = {
                "poi_geo_inf": poi_geo_inf,
                "user_poi_mat": user_poi_mat,
                "batch_local": batch_local,
                "batch_global": batch_global,
                "batch_user_kg": batch_user_kg,
                "batch_user": batch_user,
                "batch_pos": batch_pos,
                "batch_neg": batch_neg,
            }

            with tf.GradientTape() as tape:
                loss_value = custom_loss(args, model, inputs)
                grads = tape.gradient(loss_value, model.variables)
                optimizer.apply_gradients(zip(grads, model.variables))

            if index and not (index % args.print_iter):
                print("epoch:{:}, loss:{:}".format(epoch, loss_value.numpy()))

        # Test
        eval_msg = Test(args, model, data_loader, poi_geo_inf, user_poi_mat)
        for key, value in eval_msg.items():
            print("Epoch:{:} K:{:} Precision:{:.6f} Recall:{:.6f}"
                  .format(epoch, key, value["Precision"], value["Recall"]))

if __name__ == '__main__':
    tf.config.set_soft_device_placement(True)
    # tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        gpu0 = gpus[5]
        tf.config.experimental.set_memory_growth(gpu0, True)
        tf.config.set_visible_devices([gpu0], "GPU")

    utils.set_seed(args.random_seed)
    cprint("tf version")
    print(tf.__version__)
    cprint("SEED:")
    print(args.random_seed)

    dataloader = DataLoader(args)

    main(args, dataloader)
