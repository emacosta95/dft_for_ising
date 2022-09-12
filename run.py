from src.gradient_descent import GradientDescent
import argparse
import torch
import numpy as np
from src.training.utils import from_txt_to_bool

parser = argparse.ArgumentParser()

parser.add_argument(
    "--n_instances", type=int, help="# of target samples (default=100)", default=100
)
parser.add_argument(
    "--loglr", type=int, help="value of the log(lr) (default=-1)", default=-1
)
parser.add_argument(
    "--cut", type=int, help="value of gradient cutoff (deprecated)", default=128
)
parser.add_argument(
    "--logdiffsoglia",
    type=int,
    help="value of the early stopping thrashold (default=-5)",
    default=-5,
)
parser.add_argument(
    "--n_ensambles",
    type=int,
    help="# of initial configuration (default=1)",
    default=1,
)
parser.add_argument(
    "--target_path",
    type=str,
    help="name of the target dataset (default='data/dataset/valid_sequential_64_l_0_h_15000_n.npz')",
    default="data/dataset/valid_sequential_64_l_0_h_15000_n.npz",
)
parser.add_argument(
    "--init_path",
    type=str,
    help="name of the init dataset (default='data/dataset/valid_sequential_64_l_0_h_150000_n.npz')",
    default="data/dataset/train_sequential_64_l_0_h_150000_n.npz",
)


parser.add_argument(
    "--model_name",
    type=str,
    help="name of model (default='ising_model_cnn_h_uniform_30_hc_3_ks_2_ps')",
    default="ising_model_cnn_30_hc_3_ks_2_ps",
)
parser.add_argument(
    "--run_name",
    type=str,
    help="name of the run (default='ising_model_cnn_h_uniform_30_hc_3_ks_2_ps')",
    default="h_3.0_ising_model_unet_gelu_3_layers_30_hc_ks_2_ps",
)
parser.add_argument(
    "--epochs", type=int, help="# of epochs (default=10001)", default=10001
)
parser.add_argument(
    "--variable_lr",
    type=str,
    help="if it is true implement a dynamic learning rate (default=True)",
    default="False",
)
parser.add_argument(
    "--early_stopping",
    type=str,
    help="if it is true implement the early stopping (default=False)",
    default="False",
)
parser.add_argument("--L", type=int, help="size of the system (default=14)", default=14)
parser.add_argument(
    "--resolution", type=int, help="resolution of the system (default=64)", default=64
)
parser.add_argument(
    "--final_lr",
    type=float,
    help="resolution of the system (default=10**-6)",
    default=10 ** -6,
)
parser.add_argument(
    "--num_threads",
    type=int,
    help="number of threads for the torch process (default=1)",
    default=1,
)
parser.add_argument(
    "--device",
    type=str,
    help="the threshold difference for the early stopping (default=device available)",
    default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
parser.add_argument(
    "--seed",
    type=int,
    help="seed for numpy and pytorch (default=42)",
    default=42,
)


args = parser.parse_args()

n_init=np.load(args.init_path)['density']

gd = GradientDescent(
    n_instances=args.n_instances,
    run_name=args.run_name,
    loglr=args.loglr,
    n_init=n_init,
    cut=args.cut,
    n_ensambles=args.n_ensambles,
    model_name=args.model_name,
    target_path=args.target_path,
    epochs=args.epochs,
    variable_lr=from_txt_to_bool(args.variable_lr),
    early_stopping=from_txt_to_bool(args.early_stopping),
    L=args.L,
    resolution=args.resolution,
    final_lr=args.final_lr,
    num_threads=args.num_threads,
    device=args.device,
    seed=args.seed,
    logdiffsoglia=args.logdiffsoglia,
)

gd.run()