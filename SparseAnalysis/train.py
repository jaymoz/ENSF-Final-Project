import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from data import CIFAR10Data
from module import CIFAR10Module

from nni.compression.pruning import *

from nni.compression.speedup import ModelSpeedup

import wandb
import yaml
from lightning.pytorch.callbacks import ModelSummary

# from omegaconf import OmegaConf


def get_model_param(model):
    param_num = {}
    total_param = 0
    for name, param in model.named_parameters():
        n = name.split(".")[:2]
        n = ".".join(n)
        if n not in param_num:
            param_num[n] = param.numel()
        else:
            param_num[n] += param.numel()
        total_param += param.numel()
    return param_num, total_param


# def main(args):
def main():
    with open("./sweep.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # args = OmegaConf.load("./config.yaml")

    run = wandb.init(project="pruning_analysis", config=config)

    args = wandb.config
    if bool(args.download_weights):
        CIFAR10Data.download_weights()
    else:
        seed_everything(0)
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

        if args.logger == "wandb":
            logger = WandbLogger(name=args.classifier, project="cifar10")
        elif args.logger == "tensorboard":
            logger = TensorBoardLogger("cifar10", name=args.classifier)

        trainer = Trainer(
            fast_dev_run=bool(args.dev),
            logger=logger if not bool(args.dev + args.test_phase) else None,
            deterministic=True,
            log_every_n_steps=1,
            max_epochs=args.max_epochs,
            precision=args.precision,
        )

        model = CIFAR10Module(args)
        data = CIFAR10Data(args)
        print(model)
        param_num, total_param = get_model_param(model.model)
        print(param_num, total_param)

        if bool(args.pretrained):
            state_dict = os.path.join(
                "cifar10_models", "state_dicts", args.classifier + ".pt"
            )
            model.model.load_state_dict(torch.load(state_dict))
        columns = [
            "model",
            "pruner",
            "sparse ratio",
            "num_epochs",
            "pre pruning accuracy",
            "post pruning accuracy",
        ]
        my_data = [
            args.classifier,
            args.pruning_method,
            args.sparse_ratio,
            args.max_epochs,
        ]

        prepacc = trainer.test(model, data.test_dataloader())[0]["acc/test"]

        my_data.append(prepacc)
        model.prune()

        print("Pruned Model: \n")
        param_num, total_param = get_model_param(model.model)
        print(param_num, total_param)
        print(model.model)
        trainer.fit(model, data.train_dataloader())

        postpacc = trainer.test(model, data.test_dataloader())[0]["acc/test"]
        my_data.append(postpacc)
        print(my_data)
        table = wandb.Table(data=[my_data], columns=columns)
        run.log({"table": table})


if __name__ == "__main__":
    # parser = ArgumentParser()

    # # PROGRAM level args
    # parser.add_argument("--data_dir", type=str, default="./data/cifar10")
    # parser.add_argument("--download_weights", type=int, default=0, choices=[0, 1])
    # parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
    # parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    # parser.add_argument(
    #     "--logger", type=str, default="wandb", choices=["tensorboard", "wandb"]
    # )

    # # TRAINER args
    # parser.add_argument("--classifier", type=str, default="vgg11_bn")
    # parser.add_argument("--pretrained", type=int, default=1, choices=[0, 1])

    # parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    # parser.add_argument("--batch_size", type=int, default=128)
    # parser.add_argument("--max_epochs", type=int, default=100)
    # parser.add_argument("--num_workers", type=int, default=8)
    # parser.add_argument("--gpu_id", type=str, default="0")

    # parser.add_argument("--learning_rate", type=float, default=1e-2)
    # parser.add_argument("--weight_decay", type=float, default=1e-2)
    # parser.add_argument(
    #     "--pruning_method",
    #     type=str,
    #     default="l2norm",
    #     choices=[
    #         "level",
    #         "l1norm",
    #         "l2norm",
    #         "fpgm",
    #         "slim",
    #         "taylor",
    #         "linear",
    #         "agp",
    #         "movement",
    #     ],
    # )
    # parser.add_argument("--sparse_ratio", type=float, default=0.5)

    # args = parser.parse_args()

    # main(args)
    main()
