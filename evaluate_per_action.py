"""Export per-action AP and ethogram-group AP for an Eco-Mamba checkpoint."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchmetrics.classification import MultilabelAveragePrecision

from eco_mamba.data.data_module import AnimalKingdomDataModule
from eco_mamba.data.ethogram import AK_ACTION_TO_GROUP_ID, AK_GROUP_NAMES
from eco_mamba.metrics import collect_per_class_auc, compute_group_map
from eco_mamba.models.eco_mamba import EcoMambaTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an Eco-Mamba checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="Animal_Kingdom")
    parser.add_argument("--clip-model-path", default="openai/clip-vit-base-patch16")
    parser.add_argument("--pretrained-root", default="pretrained")
    parser.add_argument("--videomamba-version", default="m", choices=["t", "s", "m"])
    parser.add_argument("--num-frames", default=12, type=int)
    parser.add_argument("--sampling-strategy", default="ees", choices=["uniform", "random", "motion", "relation", "ees"])
    parser.add_argument("--decoder-layers", default=16, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", default="per_action_results")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--dataset", default="animalkingdom", choices=["animalkingdom"])
    parser.add_argument("--test-every", default=5, type=int)
    parser.add_argument("--freeze-backbone-epochs", default=100, type=int)
    parser.add_argument("--finetune-backbone-epochs", default=150, type=int)
    parser.add_argument("--group-loss-weight", default=1.0, type=float)
    parser.add_argument("--hier-loss-weight", default=.2, type=float)
    parser.set_defaults(use_context_branch=True, use_hierarchical_decoder=True, use_videomamba_pretrained=False)
    return parser.parse_args()


def main(args):
    device = f"cuda:{args.gpu}" if args.gpu != "cpu" and torch.cuda.is_available() else "cpu"
    data = AnimalKingdomDataModule(args, args.data_root)
    validation_loader = data.validation_loader()
    class_list = list(data.action_to_id().keys())
    metric = MultilabelAveragePrecision(num_labels=len(class_list), average="micro")
    trainer = EcoMambaTrainer(validation_loader, validation_loader, nn.BCEWithLogitsLoss(), metric,
                             class_list, args.test_every, False, device, args)
    trainer.load_checkpoint(args.checkpoint, map_location=device)
    trainer.model.eval()

    logits, targets = [], []
    with torch.no_grad():
        for clip, label in validation_loader:
            output = trainer.model(clip.to(device, non_blocking=True))
            logits.append(output["action_logits"].cpu().numpy())
            targets.append(label.numpy())
    logits, targets = np.concatenate(logits), np.concatenate(targets)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    action_ap = collect_per_class_auc(logits, targets, class_list)
    pd.DataFrame([{"action_id": index, "action": name, "AP": action_ap[name]}
                  for index, name in enumerate(class_list)]).sort_values("AP", ascending=False).to_csv(
                      output_dir / "per_action_ap.csv", index=False)
    group_map, group_ap = compute_group_map(logits, targets, AK_ACTION_TO_GROUP_ID, AK_GROUP_NAMES)
    pd.DataFrame([{"group": name, "AP": group_ap[name]} for name in AK_GROUP_NAMES]).sort_values(
        "AP", ascending=False).to_csv(output_dir / "per_group_ap.csv", index=False)
    print(f"[INFO] Group mAP: {group_map:.4f}")
    print(f"[INFO] Results written to: {output_dir}")


if __name__ == "__main__":
    main(parse_args())
