from __future__ import annotations

import logging
import os
import pprint

import torch
import yaml
from torch import nn
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from dataset.ssdg_dataset import SSDGDataset
from dataset.style_sampler import RandomStyleSampler
from evaluate import evaluate
from model.siab import SIAB
from model.ema import EMA
from model.unet import UNet
from utils import AverageMeter, DiceLoss, fix_seed, init_log, sigmoid_rampup
from utils.env import get_module_version
from utils.mask_convert import converter
from utils.parse_args import parse_args
from utils.sampler import MultiDomainSampler


def main():
    args = parse_args()

    fix_seed(args.seed)

    cfg: dict = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    cfg.update(yaml.load(open(args.shared_config, "r"), Loader=yaml.Loader))
    cfg.update(yaml.load(open(args.train_config, "r"), Loader=yaml.Loader))

    torch.set_num_threads(cfg["num_threads"])

    convert = converter[cfg["dataset"]]
    trainset = SSDGDataset(name=cfg["dataset"],
                           root=cfg["data_root"],
                           target_domain=args.domain,
                           mode="train",
                           n_domains=cfg["n_domains"],
                           image_size=cfg["image_size"])
    trainset_u, trainset_l, indices = trainset.split_ulb_lb(args.ratio)
    trainset.config_augmentation("strong+style",
                                 sampler=RandomStyleSampler(mode="hist"))

    valset = SSDGDataset(name=cfg["dataset"],
                         root=cfg["data_root"],
                         target_domain=args.domain,
                         mode="val",
                         n_domains=cfg["n_domains"],
                         image_size=cfg["image_size"])

    logger = init_log("global", logging.INFO)
    logger.propagate = 0  # type: ignore

    logger.info("labeled: \n{}".format(trainset_l))
    logger.info("unlabeled: \n{}".format(trainset_u))

    env = get_module_version([
        "numpy",
        "PIL",
        "scipy",
        "skimage",
        "torch",
        "torchvision",
    ])
    env = {"env": env}
    all_args = {**cfg, **vars(args), **env}
    logger.info("cfg: \n{}\n".format(pprint.pformat(all_args)))

    os.makedirs(args.save_path, exist_ok=True)

    with open(args.save_path + "/split", "w") as f:
        f.write(str(indices))

    model = UNet(in_chns=cfg["n_channels"],
                 class_num=cfg["n_classes"],
                 dropout=cfg.get("dropout", False))
    model = SIAB.convert_siab(model,
                              cfg["n_domains"],
                              num_global_in=cfg["num_global_in"])
    # init model global_in mixing coefficients
    if "init_bn_weight" in cfg:
        model.init_bn_weight(cfg["init_bn_weight"])

    model.cuda()
    model_ema = EMA(model, decay=args.decay)

    normal, alpha = model.separate_parameters()
    extra_lr = 10
    params = [
        {
            "params": normal,
            "lr": cfg["lr"],
        },
        {
            "params": alpha,
            "lr": cfg["lr"] * extra_lr,
            "weight_decay": 0.0,
        },
    ]
    if cfg["optimizer"] == "sgd":
        optimizer = SGD(params, cfg["lr"], momentum=0.9)
    elif cfg["optimizer"] == "adamw":
        optimizer = AdamW(params, cfg["lr"])
    elif cfg["optimizer"] == "adam":
        optimizer = Adam(params, cfg["lr"])
    else:
        raise NotImplementedError

    criterion_ce = nn.CrossEntropyLoss()
    criterion_ce_pixel = nn.CrossEntropyLoss(reduction="none")
    criterion_dice = DiceLoss(n_classes=cfg["n_classes"])
    dice_args = dict(softmax="softmax", onehot=True)

    def conf_thresh(x):
        return x > cfg["conf_thresh"]

    sampler_l = MultiDomainSampler(trainset_l.lengths,
                                   balanced=cfg["balanced"])
    trainloader_l = DataLoader(trainset_l,
                               batch_size=cfg["batch_size"],
                               pin_memory=True,
                               num_workers=cfg["num_workers"],
                               drop_last=True,
                               sampler=sampler_l)
    sampler_u = MultiDomainSampler(trainset.lengths, balanced=cfg["balanced"])
    trainloader_u = DataLoader(trainset,
                               batch_size=cfg["batch_size"],
                               pin_memory=True,
                               num_workers=cfg["num_workers"],
                               drop_last=True,
                               sampler=sampler_u)
    sampler_u_mix = MultiDomainSampler(trainset.lengths,
                                       balanced=cfg["balanced"])
    trainloader_u_mix = DataLoader(trainset,
                                   batch_size=cfg["batch_size"],
                                   pin_memory=True,
                                   num_workers=cfg["num_workers"],
                                   drop_last=True,
                                   sampler=sampler_u_mix)
    trainloader_l = iter(trainloader_l)
    trainloader_u = iter(trainloader_u)
    trainloader_u_mix = iter(trainloader_u_mix)

    valloader = DataLoader(valset,
                           batch_size=1,
                           pin_memory=True,
                           num_workers=1,
                           drop_last=False)

    n_iters = cfg["iters"]
    total_iters = n_iters * cfg["epochs"]
    previous_best = 0.0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, "latest.pth")):
        checkpoint = torch.load(os.path.join(args.save_path, "latest.pth"))
        model.load_state_dict(checkpoint["model"])
        model_ema.module.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]

        if epoch >= cfg["epochs"] - 1:
            logger.info("************ Skip trained checkpoint at epoch %i\n" %
                        epoch)
            exit()

        # reset learning rate
        current_iters = epoch * n_iters
        lr = cfg["lr"] * (1 - current_iters / total_iters)**0.9
        optimizer.param_groups[0]["lr"] = lr

        logger.info("************ Load from checkpoint at epoch %i\n" % epoch)

    writer = SummaryWriter(args.save_path)

    for epoch in range(epoch + 1, cfg["epochs"]):
        logger.info(
            "===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}".
            format(epoch, optimizer.param_groups[0]["lr"], previous_best))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s1 = AverageMeter()
        total_loss_s2 = AverageMeter()
        total_loss_b = AverageMeter()
        total_mask_ratio = AverageMeter()

        for i in range(n_iters):
            img_x, domain_x, mask_x = next(trainloader_l)
            img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2, domain_u, *_ = next(
                trainloader_u)
            img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _, domain_u_mix, *_ = next(
                trainloader_u_mix)

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_w_mix = img_u_w.cuda(), img_u_w_mix.cuda()
            img_u_s1, img_u_s1_mix = img_u_s1.cuda(), img_u_s1_mix.cuda()
            img_u_s2, img_u_s2_mix = img_u_s2.cuda(), img_u_s2_mix.cuda()
            cutmix_box1 = cutmix_box1.cuda()
            cutmix_box2 = cutmix_box2.cuda()
            domain_x = domain_x.tolist()
            domain_u = domain_u.tolist()
            domain_u_mix = domain_u_mix.tolist()

            model.train()
            model_ema.train()

            with torch.no_grad():
                batch_u = torch.cat(
                    [img_u_w, img_u_w_mix, img_u_w, img_u_w_mix])
                domain_id = (domain_u + domain_u_mix + [-1] *
                             (len(domain_u) + len(domain_u_mix)))

                pred_u_w, pred_u_w_mix, pred_u_wg, pred_u_wg_mix = model_ema(
                    batch_u, domain_id=domain_id).detach().chunk(4)
                # forward only for statistics stability
                model(batch_u, domain_id=domain_id)

                conf_u_w, mask_u_w = (pred_u_w.softmax(dim=1) +
                                      pred_u_wg.softmax(dim=1)).div(2).max(
                                          dim=1)
                conf_u_w_mix, mask_u_w_mix = (
                    pred_u_w_mix.softmax(dim=1) +
                    pred_u_wg_mix.softmax(dim=1)).div(2).max(dim=1)

            cutmix_box_in1 = cutmix_box1.unsqueeze(1).expand_as(img_u_s1)
            cutmix_box_out1 = cutmix_box1
            cutmix_box_in2 = cutmix_box2.unsqueeze(1).expand_as(img_u_s2)
            cutmix_box_out2 = cutmix_box2

            img_u_s1[cutmix_box_in1 == 1] = \
                img_u_s1_mix[cutmix_box_in1 == 1]
            img_u_s2[cutmix_box_in2 == 1] = \
                img_u_s2_mix[cutmix_box_in2 == 1]

            pred_x = model(torch.cat([img_x, img_x]),
                           domain_id=domain_x + [-1] * len(domain_x))
            pred_u_s1, pred_u_s2 = model(torch.cat([img_u_s1,
                                                    img_u_s2])).chunk(2)

            mask_u_w_cutmixed1 = mask_u_w.clone()
            conf_u_w_cutmixed1 = conf_u_w.clone()
            mask_u_w_cutmixed2 = mask_u_w.clone()
            conf_u_w_cutmixed2 = conf_u_w.clone()

            mask_u_w_cutmixed1[cutmix_box_out1 == 1] = \
                mask_u_w_mix[cutmix_box_out1 == 1]
            conf_u_w_cutmixed1[cutmix_box_out1 == 1] = \
                conf_u_w_mix[cutmix_box_out1 == 1]
            mask_u_w_cutmixed2[cutmix_box_out2 == 1] = \
                mask_u_w_mix[cutmix_box_out2 == 1]
            conf_u_w_cutmixed2[cutmix_box_out2 == 1] = \
                conf_u_w_mix[cutmix_box_out2 == 1]

            conf_mask_cutmixed1 = conf_thresh(conf_u_w_cutmixed1).float()
            conf_mask_cutmixed2 = conf_thresh(conf_u_w_cutmixed2).float()
            conf_mask = conf_thresh(conf_u_w).float()

            mask_x = convert(mask_x)
            mask_x = torch.cat([mask_x, mask_x])
            loss_x = (criterion_ce(pred_x, mask_x) +
                      criterion_dice(pred_x, mask_x, **dice_args)) / 2.0
            loss_u_s1 = (criterion_ce_pixel(pred_u_s1, mask_u_w_cutmixed1) *
                         conf_mask_cutmixed1).mean()
            loss_u_s2 = (criterion_ce_pixel(pred_u_s2, mask_u_w_cutmixed2) *
                         conf_mask_cutmixed2).mean()
            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0

            if cfg["num_random"] == "all" or cfg["num_random"] > 0:
                with SIAB.stop_grad(model):
                    pred_u_b = model(img_u_w,
                                     domain_id=domain_u,
                                     random_layer=cfg["num_random"],
                                     p=cfg["p"])
                loss_u_b = (criterion_ce_pixel(pred_u_b, mask_u_w) *
                            conf_mask).mean()
                loss_u = (loss_u_s +
                          loss_u_b * cfg["weight_b"]) / (1 + cfg["weight_b"])
            else:
                loss_u_b = torch.zeros_like(loss_u_s)
                loss_u = loss_u_s

            current_iters = epoch * n_iters + i
            loss = (loss_x + loss_u *
                    sigmoid_rampup(current_iters, cfg["rampup"])) / 2.0

            mask_ratio = conf_thresh(conf_u_w).float().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s1.update(loss_u_s1.item())
            total_loss_s2.update(loss_u_s2.item())
            total_loss_b.update(loss_u_b.item())
            total_mask_ratio.update(mask_ratio.item())

            lr = cfg["lr"] * (1 - current_iters / total_iters)**0.9
            optimizer.param_groups[0]["lr"] = lr
            model_ema.update(model, current_iters)

            if i % (n_iters // 8) == 0:
                logger.info("Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, "
                            "Loss s: {:.3f}/{:.3f}, Loss b: {:.4f}, "
                            "Mask ratio: {:.3f}".format(
                                i, total_loss.avg, total_loss_x.avg,
                                total_loss_s1.avg, total_loss_s2.avg,
                                total_loss_b.avg, total_mask_ratio.avg))
            writer.add_scalar("train/loss_all", total_loss.avg, current_iters)
            writer.add_scalar("train/loss_x", total_loss_x.avg, current_iters)
            writer.add_scalar("train/loss_s1", total_loss_s1.avg,
                              current_iters)
            writer.add_scalar("train/loss_s2", total_loss_s2.avg,
                              current_iters)
            writer.add_scalar("train/loss_b", total_loss_b.avg, current_iters)
            writer.add_scalar("train/mask_ratio", total_mask_ratio.avg,
                              current_iters)

        # evaluation
        mean_dice, _, dice_class_domain = evaluate(model,
                                                   valloader,
                                                   cfg,
                                                   is_target_domain=True)
        dice_class = dice_class_domain[0]

        for (cls_idx, dice) in enumerate(dice_class):
            logger.info("***** Evaluation ***** >>>> "
                        "Class [{:}] Dice: {:.2f}".format(cls_idx, dice))
        logger.info("***** Evaluation ***** >>>> "
                    "MeanDice: {:.2f}\n".format(mean_dice))

        writer.add_scalar("eval/MeanDice", mean_dice, epoch)
        for i, dice in enumerate(dice_class):
            writer.add_scalar("eval/Class_%s_dice" % i, dice, epoch)

        is_best = mean_dice > previous_best
        previous_best = max(mean_dice, previous_best)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "previous_best": previous_best,
            "ema": model_ema.module.state_dict(),
            "num_global_in": cfg["num_global_in"],
        }
        torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))
        if is_best:
            torch.save(checkpoint, os.path.join(args.save_path, "best.pth"))

    writer.close()


if __name__ == "__main__":
    main()
