"""Training entry point for Eco-Mamba on Animal Kingdom."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchmetrics.classification import MultilabelAveragePrecision

from eco_mamba.data.data_module import AnimalKingdomDataModule
from eco_mamba.models.eco_mamba import EcoMambaTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Eco-Mamba on Animal Kingdom.")
    parser.add_argument("--dataset", default="animalkingdom", choices=["animalkingdom"])
    parser.add_argument("--data-root", default="Animal_Kingdom", help="Animal Kingdom dataset directory.")
    parser.add_argument("--clip-model-path", default="openai/clip-vit-base-patch16")
    parser.add_argument("--pretrained-root", default="pretrained")
    parser.add_argument("--videomamba-version", default="m", choices=["t", "s", "m"])
    parser.add_argument("--num-frames", default=12, type=int, help="EES frame budget K.")
    parser.add_argument("--sampling-strategy", default="ees", choices=["uniform", "random", "motion", "relation", "ees"])
    parser.add_argument("--decoder-layers", default=16, type=int)
    parser.add_argument("--freeze-backbone-epochs", default=100, type=int)
    parser.add_argument("--finetune-backbone-epochs", default=150, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--test-every", default=5, type=int)
    parser.add_argument("--group-loss-weight", default=1.0, type=float)
    parser.add_argument("--hier-loss-weight", default=0.2, type=float)
    parser.add_argument("--checkpoint", default="", help="Checkpoint to resume.")
    parser.add_argument("--save-path", default="checkpoints/eco_mamba.pth")
    parser.add_argument("--gpu", default="0", help="CUDA index, or 'cpu'.")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--no-context-branch", action="store_false", dest="use_context_branch")
    parser.add_argument("--no-hierarchical-decoder", action="store_false", dest="use_hierarchical_decoder")
    parser.add_argument("--no-videomamba-pretrained", action="store_false", dest="use_videomamba_pretrained")
    parser.set_defaults(use_context_branch=True, use_hierarchical_decoder=True, use_videomamba_pretrained=True)
    return parser.parse_args()


def main(args):
    device = f"cuda:{args.gpu}" if args.gpu != "cpu" and torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device != "cpu":
        torch.cuda.manual_seed_all(args.seed)

    data = AnimalKingdomDataModule(args, args.data_root)
    train_loader, validation_loader = data.train_loader(), data.validation_loader()
    class_list = list(data.action_to_id().keys())
    metric = MultilabelAveragePrecision(num_labels=len(class_list), average="micro")
    trainer = EcoMambaTrainer(train_loader, validation_loader, nn.BCEWithLogitsLoss(), metric,
                              class_list, args.test_every, False, device, args)
    start_epoch = trainer.load_checkpoint(args.checkpoint, map_location=device) if args.checkpoint else 0
    trainer.train(start_epoch)
    trainer.save_model(args.save_path)
    print(f"[INFO] Saved final Eco-Mamba model to: {Path(args.save_path)}")


if __name__ == "__main__":
    main(parse_args())
