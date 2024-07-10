import argparse

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset.ssdg_dataset import SSDGDataset
from evaluate_metrics import evaluate_volume_metrics
from model.siab import SIAB
from model.unet import UNet
from utils import fix_seed

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--domain', type=int, required=True)
parser.add_argument('--key', default="model")
parser.add_argument('--seed', type=int, default=1339)

def main():
    args = parser.parse_args()
    fix_seed(args.seed)

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    valset = SSDGDataset(name=cfg["dataset"],
                         root=cfg["data_root"],
                         target_domain=args.domain,
                         mode="val",
                         n_domains=cfg["n_domains"],
                         image_size=cfg["image_size"])

    valloader = DataLoader(valset,
                           batch_size=1,
                           shuffle=False,
                           num_workers=1,
                           pin_memory=True,
                           drop_last=False)

    ckpt = torch.load(args.path)
    model = UNet(class_num=cfg["n_classes"], in_chns=cfg["n_channels"])
    model = SIAB.convert_siab(model,
                              num_domains=cfg["n_domains"],
                              num_global_in=ckpt["num_global_in"])
    model.load_state_dict(ckpt[args.key])
    model.cuda()

    collectors = evaluate_volume_metrics(model,
                                         valloader,
                                         cfg,
                                         is_target_domain=True,
                                         verbose=True)
    print(f"{valset} {args.path}")
    for collector in collectors:
        metric = collector.metric
        mean, _, domain_classwise = collector.get()
        classwise = domain_classwise[0]
        instance_values = collector.averaged_instances[0]
        std = np.std(instance_values)
        print(f"[{metric}] Mean: {mean:.4f}±{std:.4f}, "
              f"Class: {','.join([f'{i:.4f}' for i in classwise])}")


if __name__ == '__main__':
    main()
