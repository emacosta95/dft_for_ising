from typing import List
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple
import os
from torchmetrics import R2Score
from tqdm import tqdm, trange
from torch.utils.data import Dataset, TensorDataset, DataLoader
import matplotlib.pyplot as plt



def decreasing(val_losses, best_loss, min_delta=0.001):
    """for early stopping"""
    try:
        is_decreasing = val_losses[-1] < best_loss - min_delta
    except:
        is_decreasing = True
    return is_decreasing


def fit(
    epochs: int,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    loss_func: nn.Module,
    supervised: bool,
    checkpoint: bool,
    name_checkpoint: str,
    history_train: List,
    history_valid: List,
    patiance: int,
    early_stopping: float,
    device:str,
) -> Tuple:
    """This function fits the model using the selected optimizer.
        It will return a list with the loss values and the accuracy as a tuple (loss,accuracy).

    Argument:

    epochs: number of epochs
    model: the selected model choosen for the train
    opt: the optimization class
    train_dl: the Dataloader for the training set
    valid_dl: the DataLoader for the validation set
    loss_func: The loss function used for the training
    checkpoint: if true a model is saved every 5 epochs
    name_checkpoint: if checkpoint is true, the name of the checkpoint model
    history_train: the record of train losses over the past epochs
    history_valid: the record of valid losses over the past epochs

    return: the evolution of the train and valid losses

    """

    loss_func = loss_func

    wait = 0
    if supervised:
        r_max = -100000
    best_loss = 10 ** 9

    for epoch in trange(epochs, desc="train epoch"):

        model.train()
        loss_ave_train = 0
        loss_ave_valid = 0
        kldiv_train = 0
        kldiv_valid = 0

        tqdm_iterator = tqdm(
            enumerate(train_dl),
            total=len(train_dl),
            desc=f"batch [loss_ave: None]",
            leave=False,
        )

        for batch_idx, batch in tqdm_iterator:

            batch = batch
            if not (supervised):
                loss, _ = model.train_generative_step(batch, device)
            else:
                loss = model.train_step(batch, device)
            loss.backward()

            opt.step()
            opt.zero_grad()

            tqdm_iterator.set_description(f"train batch [avg loss: {loss.item():.3f}]")
            tqdm_iterator.refresh()

        model.eval()
        if supervised:
            r2 = R2Score()

        for batch in valid_dl:
            if supervised:
                r2 = model.r2_computation(batch, device, r2)
            else:
                loss, kldiv = model.train_generative_step(batch, device)
                loss_ave_valid += loss.item()
                kldiv_valid += kldiv
        if supervised:
            r_ave_train = r2.compute()
            r2.reset()

        for batch in train_dl:
            if supervised:
                r2 = model.r2_computation(batch, device, r2)
            else:
                loss, kldiv = model.train_generative_step(batch, device)
                loss_ave_train += loss.item()
                kldiv_train += kldiv
        if supervised:
            r_ave_valid = r2.compute()
            r2.reset()

            history_train.append(r_ave_train)
            history_valid.append(r_ave_valid)
            print(r_ave_valid)

        else:
            kldiv_train = kldiv_train / len(train_dl)
            kldiv_valid = kldiv_valid / len(valid_dl)

            loss_ave_train = loss_ave_train / len(train_dl)
            history_train.append(loss_ave_train)
            loss_ave_valid = loss_ave_valid / len(valid_dl)
            history_valid.append(loss_ave_valid)

        wait = +1
        if supervised:
            metric = r_max
        else:
            metric = best_loss
        if decreasing(history_valid, metric, early_stopping):
            wait = 0
        if wait >= patiance:
            print(f"EARLY STOPPING AT {early_stopping}")

        if checkpoint:

            if supervised:
                if r_max <= r_ave_valid:
                    print("Decreasing!")

                    torch.save(
                        model,
                        f"model_rep/{name_checkpoint}",
                    )
                    r_max = r_ave_valid
            else:
                if best_loss >= loss_ave_valid:
                    print("Decreasing!")
                    torch.save(
                        model,
                        f"model_rep/{name_checkpoint}",
                    )
                    best_loss = loss_ave_valid

            if supervised:
                text = ""
            else:
                text = "_generative"

            torch.save(
                history_train,
                f"losses_dft_pytorch/{name_checkpoint}_loss_train" + text,
            )
            torch.save(
                history_valid,
                f"losses_dft_pytorch/{name_checkpoint}_loss_valid" + text,
            )

        if supervised:
            print(
                f"R_ave_overfitting={r_ave_train} \n"
                f"R_ave_valid={r_ave_valid} \n"
                f"R_max={r_max} \n"
                f"epochs={epoch}\n"
            )
        else:
            print(
                f"kldiv_valid={kldiv_valid} \n"
                f"kldiv_train={kldiv_train} \n"
                f"loss_ave_train={loss_ave_train} \n"
                f"loss_ave_valid={loss_ave_valid} \n"
                f"epochs={epoch}\n"
            )

    return history_train, history_valid
