#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2022-07-15 15:31
import torch
import torch.optim as optim
import config
import argparse
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm
from dataset import Tusimple
from dataset_culane import Culane
from scnn_vgg import SCNNVgg
from scnn_mobilenet import SCNNMobileNet
import wandb
from torchmetrics import F1Score, MeanMetric

device = torch.device("cuda:0")


def train():
    if args.dataset == "tusimple_culane":
	    train_dataset_tusimple = Tusimple("train")
        train_dataset_culane = Culane("train")
		train_dataset = ConcatDataset([train_dataset_tusimple, train_dataset_culane])

    if args.dataset == "tusimple":
	    train_dataset = Tusimple("train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = Tusimple("test")
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    net = None
    if args.model == "vgg":
        net = SCNNVgg(pretrained=True)
    if args.model == "mobilenet":
        net = SCNNMobileNet(pretrained=True)

    net = net.to(device)
    net.train()
    optimizer = optim.SGD(
        net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4
    )

    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.MAX_LR,
        total_steps=args.epoch * len(train_loader),
    )

    best_loss = 65535
    if args.resume:
        try:
            save_dict = torch.load(net.get_model_name())
            net.load_state_dict(save_dict["net"])
            optimizer.load_state_dict(save_dict["optimizer"])
            lr_scheduler.load_state_dict(save_dict["lr_scheduler"])
            best_loss.load_state_dict(save_dict["best_loss"])
            print("load pth done")
        except:
            pass

    for i in range(args.epoch):
        f1_score_metric = F1Score(
            num_classes=5, average="none", mdmc_reduce="global"
        ).to(device)
        val_loss_metric = MeanMetric().to(device)

        loss_metric = MeanMetric().to(device)

        progress = tqdm(range(len(train_loader) + len(test_loader)))
        for index, samples in enumerate(train_loader):
            for sample in samples:
                img = sample["img"].to(device)
                label = sample["label"].to(device)
                exist = sample["exist"].to(device)
                optimizer.zero_grad()
                _, _, loss = net(img, label, exist)
                loss.backward()
                optimizer.step()

                loss_metric.update(loss.item())

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(
                        {
                            "net": net.state_dict(),
                            "optimimzer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "best_loss": best_loss,
                        },
                        net.get_model_name(),
                    )
            lr_scheduler.step()

            if index == len(train_loader) - 1:
                progress.set_description(
                    f"#{i}, loss: {loss_metric.compute():.3f}, lr: {lr_scheduler.get_last_lr()[0]:.3f}",
                )
            progress.update(1)

        with torch.no_grad():
            for _, sample in enumerate(test_loader):
                img = sample["img"].to(device)
                label = sample["label"].to(device)
                exist = sample["exist"].to(device)
                seg_pred, _, loss = net(img, label, exist)
                f1_score_metric.update(seg_pred, label)
                val_loss_metric.update(loss.item())
                progress.update(1)

        val_score = torch.mean(f1_score_metric.compute()[1:]).item()
        val_loss = val_loss_metric.compute().item()
        loss = loss_metric.compute().item()

        wandb.log(
            {
                "val_score": val_score,
                "loss": loss,
            }
        )
        wandb.log(
            {
                "val_loss": val_loss,
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epoch",
        type=int,
        default=config.EPOCH,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.BATCH,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=config.LR,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument("--model", choices=["vgg", "mobilenet"], default="mobilenet")
    parser.add_argument("--dataset", choices=["tusimple_culane", "tusimple"], default="tusimple_culane")
    args = parser.parse_args()

    wandb.init(
        project="scnn",
        entity="sunway",
    )
    wandb.config.update(args)

    train()
