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
    Den2CorLSTM_beta,
    Den2CorRESNET,
    REDENTnopooling,
)
from src.training.dft_models import REDENTScalableTraining, REDENTnobias, Den2Func
from src.training.train_module import fit
from src.training.utils import (
    count_parameters,
    from_txt_to_bool,
    get_optimizer,
    make_data_loader_unet_scale,
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


parser.add_argument("--name", type=str, help="name of the model", default=None)

parser.add_argument(
    "--data_path",
    type=str,
    nargs="+",
    help="list of data path (default=data/unet_dataset/train_unet_periodic_16_l_3.6_h_150000_n.npz)",
    default=[
        "data/dataset_2nn/train_unet_periodic_2nn_augmentation_16_l_5.44_h_30000_n.npz",
        "data/dataset_2nn/train_unet_periodic_2nn_augmentation_12_l_5.44_h_30000_n.npz",
        "data/dataset_2nn/train_unet_periodic_2nn_augmentation_8_l_5.44_h_30000_n.npz",
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
    default=3000,
)


parser.add_argument(
    "--input_channels", type=int, help="# input channels (default=1)", default=1
)
parser.add_argument(
    "--input_size",
    type=int,
    help="number of features of the input (default=16)",
    default=16,
)


parser.add_argument(
    "--hidden_channels",
    type=int,
    nargs="+",
    help="list of hidden channels (default=[40, 40, 40, 40])",
    default=[40 for i in range(4)],
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
    help="pooling size in the Avg Pooling (default=1)",
    default=1,
)

parser.add_argument(
    "--padding",
    type=int,
    help="padding dimension (default=2)",
    default=2,
)


parser.add_argument(
    "--kernel_size",
    type=int,
    help="kernel size (default=5)",
    default=5,
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
    default="Den2Func",
)

parser.add_argument(
    "--keys",
    type=str,
    nargs="+",
    help="the keys of the dataset to correlate",
    default=["density", "density_F"],
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
            history_best = pt.load(f"losses_dft_pytorch/{args.name}" + "_loss_best_dft")
        else:
            history_valid = []
            history_train = []
            history_best = []
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
        elif args.model_type == "REDENTnopooling":
            model = REDENTnopooling(
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

        elif args.model_type == "REDENTnobias":
            model = REDENTnobias(
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

        elif args.model_type == "REDENTScalableTraining":
            model = REDENTScalableTraining(
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

        elif args.model_type == "Den2Func":
            model = Den2Func(
                Loss=nn.MSELoss(),
                in_channels=input_channels,
                Activation=nn.GELU(),
                hidden_channels=hc,
                ks=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
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

    train_dls = []
    valid_dls = []

    for file_name in args.data_path:
        train_dl, valid_dl = make_data_loader_unet(
            file_name=file_name, bs=bs, split=0.8, keys=args.keys
        )
        train_dls.append(train_dl)
        valid_dls.append(valid_dl)

    opt = get_optimizer(lr=lr, model=model)
    fit(
        supervised=True,
        model=model,
        train_dls=train_dls,
        opt=opt,
        epochs=epochs,
        valid_dls=valid_dls,
        checkpoint=True,
        name_checkpoint=model_name,
        history_train=history_train,
        history_valid=history_valid,
        history_best=history_best,
        loss_func=nn.MSELoss(),
        patiance=patiance,
        early_stopping=early_stopping,
        device=device,
    )

    print(model)


if __name__ == "__main__":

    args = parser.parse_args()

    main(args)
