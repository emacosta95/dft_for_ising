import argparse
import os
import random

import numpy as np
import torch as pt
import torch.nn as nn
from pyexpat import model
from torch.nn.modules import pooling
from tqdm import tqdm, trange

from src.training.models import (
    REDENT,
    Den2Cor,
    Den2CorCNN,
    Den2CorLSTM,
    Den2CorLSTM_beta,
    Den2CorLSTM_gamma,
    Den2CorRECURRENT,
    Den2CorRESNET,
)
from src.training.train_module import fit
from src.training.utils import (
    count_parameters,
    from_txt_to_bool,
    get_optimizer,
    make_data_loader_correlation_scale,
    make_data_loader_unet,
)

# %%

# parser arguments

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

parser.add_argument(
    "--load",
    type=bool,
    help="Loading or not the model",
    action=argparse.BooleanOptionalAction,
)

parser.add_argument(
    "--scalable_training",
    type=bool,
    help="Create a set of scalable datasets",
    action=argparse.BooleanOptionalAction,
)

parser.add_argument("--name", type=str, help="name of the model", default=None)

parser.add_argument(
    "--data_path",
    type=str,
    # nargs="+",
    help="list of data path (default=data/unet_dataset/train_unet_periodic_16_l_3.6_h_150000_n.npz)",
    default=[
        "data/correlation_1nn_rebuilt/train_1nn_correlation_map_h_1.8_150000_l_8_pbc_j_1.0.npz",
        "data/correlation_1nn_rebuilt/train_1nn_correlation_map_h_1.8_150000_l_12_pbc_j_1.0.npz",
        "data/correlation_1nn_rebuilt/train_1nn_correlation_map_h_1.8_150000_l_16_pbc_j_1.0.npz",
    ],
)

parser.add_argument(
    "--num_threads",
    type=int,
    help="the number of threads for pytorch (default=1)",
    default=1,
)

parser.add_argument(
    "--seed",
    type=int,
    help="seed for pytorch and numpy (default=42)",
    default=42,
)


parser.add_argument(
    "--device",
    type=str,
    help="the threshold difference for the early stopping (default=device available)",
    default=("cuda" if pt.cuda.is_available() else "cpu"),
)

parser.add_argument(
    "--patiance",
    type=int,
    help="num of epochs tollerance for the early stopping (default=5)",
    default=5,
)

parser.add_argument(
    "--early_stopping",
    type=float,
    help="the threshold difference for the early stopping (default=10**-4)",
    default=10 ** -4,
)

parser.add_argument(
    "--lr",
    type=float,
    help="learning rate (default=0.001)",
    default=0.001,
)

parser.add_argument(
    "--bs",
    type=int,
    help="batch size (default=100)",
    default=100,
)


parser.add_argument(
    "--epochs",
    type=int,
    help="training epochs (default=1200)",
    default=1200,
)


parser.add_argument(
    "--input_channels", type=int, help="# input channels (default=1)", default=1
)
parser.add_argument(
    "--input_size",
    type=int,
    help="number of features of the input (default=64)",
    default=64,
)


parser.add_argument(
    "--hidden_channels",
    type=int,
    nargs="+",
    help="list of hidden channels (default=120)",
    default=[30, 60, 120],
)

parser.add_argument(
    "--n_block_layers",
    type=int,
    help="number of additional convolution in a given block (default=0)",
    default=0,
)


parser.add_argument(
    "--pooling_size",
    type=int,
    help="pooling size in the Avg Pooling (default=2)",
    default=2,
)

parser.add_argument(
    "--padding",
    type=int,
    help="padding dimension (default=4)",
    default=4,
)


parser.add_argument(
    "--kernel_size",
    type=int,
    help="kernel size (default=9)",
    default=9,
)

parser.add_argument(
    "--padding_mode",
    type=str,
    help="the padding mode of the model (default='circular')",
    default="circular",
)


parser.add_argument(
    "--model_name",
    type=str,
    help="name of the model (default='unet_periodic_ising_cnn_64_size')",
    default="h_3.0_unet_periodic_ising_cnn_64_size",
)

parser.add_argument(
    "--model_type",
    type=str,
    help="could be either REDENT or Den2Cor",
    default="REDENT",
)


def main(args):

    # hyperparameters

    device = pt.device(args.device)
    input_channels = args.input_channels
    input_size = args.input_size

    # 256 for model test, 30 for the others
    hc = args.hidden_channels
    output_size = input_size
    pooling_size = args.pooling_size
    padding = args.padding  # 6
    padding_mode = args.padding_mode
    kernel_size = args.kernel_size  # 13
    n_conv_layers = len(args.hidden_channels)
    # Select the number of threads
    pt.set_num_threads(args.num_threads)
    # Initialize the seed
    pt.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    pt.cuda.manual_seed(args.seed)
    pt.backends.cudnn.deterministic = (
        True  # Note that this Deterministic mode can have a performance impact
    )
    pt.backends.cudnn.benchmark = False
    # Set hyperparameters
    epochs = args.epochs
    lr = args.lr
    bs = args.bs
    patiance = args.patiance
    early_stopping = args.early_stopping
    # Set the model name
    model_name = args.model_name
    name_hc = f"_{hc}_hc"
    name_ks = f"_{kernel_size}_ks"
    name_pooling_size = f"_{pooling_size}_ps"
    name_n_conv = f"_{len(hc)}_nconv"
    name_n_block = f"_{args.n_block_layers}_nblock"
    model_name = (
        model_name + name_hc + name_ks + name_pooling_size + name_n_conv + name_n_block
    )
    # Set the dataset path
    if args.load:
        print(f"loading the model {args.name}")
        if os.path.isfile(f"losses_dft_pytorch/{args.name}" + "_loss_valid_dft"):
            history_valid = pt.load(
                f"losses_dft_pytorch/{args.name}" + "_loss_valid_dft"
            )
            history_train = pt.load(
                f"losses_dft_pytorch/{args.name}" + "_loss_train_dft"
            )
        else:
            history_valid = []
            history_train = []
        print(len(history_train), len(history_valid))
        model = pt.load(f"model_rep/{args.name}", map_location=device)
        model.loss_dft = nn.MSELoss()
        model_name = args.name
    else:
        history_valid = []
        history_train = []

        if args.model_type == "REDENT":
            model = REDENT(
                Loss=nn.MSELoss(),
                in_channels=input_channels,
                Activation=nn.GELU(),
                hidden_channels=hc,
                ks=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
                # pooling_size=pooling_size,
                n_conv_layers=n_conv_layers,
                out_features=input_size,
                in_features=input_size,
                out_channels=input_channels,
                n_block_layers=args.n_block_layers,
            )
        elif args.model_type == "Den2Cor":
            model = Den2Cor(
                Loss=nn.MSELoss(),
                in_channels=input_channels,
                Activation=nn.GELU(),
                hidden_channels=hc,
                ks=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
                # pooling_size=pooling_size,
                n_conv_layers=n_conv_layers,
                out_features=input_size,
                in_features=input_size,
                out_channels=input_channels,
                n_block_layers=args.n_block_layers,
            )

        elif args.model_type == "Den2CorRESNET":
            model = Den2CorRESNET(
                Loss=nn.MSELoss(),
                in_channels=input_channels,
                Activation=nn.GELU(),
                hidden_channels=hc,
                ks=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
                pooling_size=pooling_size,
                n_conv_layers=n_conv_layers,
                out_features=input_size,
                in_features=input_size,
                out_channels=input_channels,
                n_block_layers=args.n_block_layers,
            )

        elif args.model_type == "Den2Magn":
            model = REDENT(
                Loss=nn.MSELoss(),
                in_channels=input_channels,
                Activation=nn.GELU(),
                hidden_channels=hc,
                ks=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
                # pooling_size=pooling_size,
                n_conv_layers=n_conv_layers,
                out_features=input_size,
                in_features=input_size,
                out_channels=input_channels,
                n_block_layers=args.n_block_layers,
            )

        elif args.model_type == "Den2CorLSTM":
            model = Den2CorLSTM_beta(
                Loss=nn.MSELoss(),
                in_channels=input_channels,
                Activation=nn.Softplus(),
                hidden_channels=hc,
                ks=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
                pooling_size=pooling_size,
                n_conv_layers=n_conv_layers,
                out_features=input_size,
                in_features=input_size,
                out_channels=input_channels,
                n_block_layers=args.n_block_layers,
            )

    model = model.to(pt.double)
    model = model.to(device=device)

    print(model)
    print(count_parameters(model))
    print(args)

    if args.scalable_training:

        train_dl, valid_dl = make_data_loader_correlation_scale(
            file_names=args.data_path, bs=bs, split=0.8
        )

    else:
        train_dl, valid_dl = make_data_loader_unet(
            file_name=args.data_path,
            bs=bs,
            split=0.8,
            pbc=False,
            model_type=args.model_type,
        )

    opt = get_optimizer(lr=lr, model=model)
    fit(
        supervised=True,
        model=model,
        train_dl=train_dl,
        opt=opt,
        epochs=epochs,
        valid_dl=valid_dl,
        checkpoint=True,
        name_checkpoint=model_name,
        history_train=history_train,
        history_valid=history_valid,
        loss_func=nn.MSELoss(),
        patiance=patiance,
        early_stopping=early_stopping,
        device=device,
    )

    print(model)


if __name__ == "__main__":

    args = parser.parse_args()

    main(args)
