"""Eco-Mamba: ecology-aware adaptive Mamba for animal action recognition."""

import time
import math
import torch
from torch import nn, Tensor
from torch.optim import AdamW
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel, logging
import numpy as np
import os
from typing import Dict

from eco_mamba.utils.training import AverageMeter
from eco_mamba.models.video_mamba_backbone import (
    configure_pretrained_weights, videomamba_middle, videomamba_small, videomamba_tiny,
)
from eco_mamba.models.hierarchical_ethogram_loss import HierarchicalEthogramLoss
from eco_mamba.data.ethogram import AK_ACTION_TO_GROUP_ID, AK_GROUP_NAMES, NUM_GROUPS
from eco_mamba.metrics import (
    build_group_targets,
    compute_group_map,
    hierarchical_consistency_rate
)
from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm
    layer_norm_fn = None
    rms_norm_fn = None
except (ImportError, Exception):
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class DecoderMambaBlock(nn.Module):
    """Pre-normalized residual Mamba block for the Eco-Mamba decoder."""

    def __init__(self, d_model, mixer_cls, norm_cls, residual_in_fp32=False, **_):
        super().__init__()
        self.mixer = mixer_cls(d_model)
        self.norm = norm_cls(d_model)
        self.residual_in_fp32 = residual_in_fp32

    def forward(self, hidden_states: Tensor, residual: Tensor | None = None):
        residual = hidden_states if residual is None else residual + hidden_states
        normalized = self.norm(residual.to(dtype=self.norm.weight.dtype))
        hidden_states = self.mixer(normalized)
        return hidden_states, residual.float() if self.residual_in_fp32 else residual


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    """Create a Mamba block"""
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = DecoderMambaBlock(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ActionQueryClassifier(nn.Module):
    """Class-specific linear classifier for decoded action queries."""
    
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias
        
        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)
    
    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class EcoMamba(nn.Module):
    """
    EES-selected frames are encoded by a VideoMamba event stream and a frozen CLIP
    scene stream. RFG fuses scene/relation conditions into event tokens, then HED
    jointly predicts ethogram groups and fine-grained actions.
    """
    
    def __init__(
        self,
        class_embed: Tensor,
        num_frames: int,
        version: str = 'm',
        event_frame_budget: int = 12,
        use_videomamba_pretrained: bool = True,
        use_context_branch: bool = True,
        context_resolution: int = 112,
        use_hierarchical_decoder: bool = True,
        decoder_layers: int = 16,
        clip_model_path: str = "openai/clip-vit-base-patch16",
        pretrained_root: str = "pretrained",
    ):
        super().__init__()
        
        # Config
        self.num_classes, self.embed_dim = class_embed.shape
        self.num_frames = num_frames
        self.event_frame_budget = event_frame_budget
        self.use_context_branch = use_context_branch
        self.use_hierarchical_decoder = use_hierarchical_decoder
        self.context_resolution = context_resolution
        
        assert version in ['m', 's', 't'], 'version must be m or s or t'
        
        # ===== VideoMamba Backbone =====
        configure_pretrained_weights(pretrained_root)
        if version == 'm':
            self.backbone = videomamba_middle(num_frames=num_frames, pretrained=use_videomamba_pretrained)
            backbone_dim = 576
        elif version == 's':
            self.backbone = videomamba_small(num_frames=num_frames, pretrained=use_videomamba_pretrained)
            backbone_dim = 384
        elif version == 't':
            self.backbone = videomamba_tiny(num_frames=num_frames, pretrained=use_videomamba_pretrained)
            backbone_dim = 192
        
        # ===== Event Stream (High-Resolution) =====
        self.event_projection = nn.Linear(in_features=backbone_dim, out_features=self.embed_dim, bias=False)
        self.event_position = PositionalEncoding(d_model=self.embed_dim)
        
        # ===== Context Stream (Low-Resolution) =====
        if use_context_branch:
            self.image_model = CLIPVisionModel.from_pretrained(clip_model_path)
            self.context_proj = nn.Linear(in_features=768, out_features=self.embed_dim, bias=False)
            
            # Relation feature projection
            self.relation_proj = nn.Linear(in_features=6, out_features=self.embed_dim, bias=False)
        else:
            self.image_model = None
            self.context_proj = None
            self.relation_proj = None
        
        # ===== Fusion Gates =====
        self.relational_fusion_gate = nn.Sequential(
            nn.Linear(self.embed_dim * 2 + 6, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
            nn.Sigmoid()
        )
        
        # ===== Query Decoder =====
        self.query_embed = nn.Parameter(class_embed)
        if use_context_branch:
            self.query_context_projection = nn.Linear(
                in_features=768 + self.embed_dim,
                out_features=self.embed_dim,
                bias=False
            )
        else:
            self.query_context_projection = nn.Linear(
                in_features=self.embed_dim,
                out_features=self.embed_dim,
                bias=False
            )
        
        # ===== Decoder Mamba Layers =====
        self.decoder_blocks = nn.ModuleList([
            create_block(
                self.embed_dim,
                layer_idx=i,
            )
            for i in range(decoder_layers)
        ])
        
        # ===== Output Layers =====
        # Action head: [B, K+C, D] -> [B, C, K]
        self.decoder_token_count = event_frame_budget + self.num_classes
        self.query_token_projection = nn.Linear(in_features=self.decoder_token_count, out_features=self.num_classes, bias=False)
        self.action_classifier = ActionQueryClassifier(self.num_classes, self.embed_dim, bias=True)
        
        # ===== Hierarchical Decoder =====
        if use_hierarchical_decoder:
            # Number of behavior groups
            self.num_groups = NUM_GROUPS
            # Group-level head
            self.group_head = nn.Linear(self.embed_dim, self.num_groups, bias=True)
            # Condition group predictions on actions
            self.group_condition_proj = nn.Linear(self.num_groups, self.embed_dim, bias=False)
        else:
            self.num_groups = 0
            self.group_head = None
            self.group_condition_proj = None
    
    def _build_relation_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract relation features from images
        
        Args:
            images: [B, T, C, H, W]
        
        Returns:
            [B, 6] relation features
        """
        b, t = images.shape[:2]
        
        # Convert to grayscale for analysis
        gray = torch.mean(images, dim=2)  # [B, T, H, W]
        
        # Compute features
        features = []
        
        # 1. Average brightness
        brightness = gray.mean(dim=(2, 3))  # [B, T]
        brightness_mean = brightness.mean(dim=1)
        brightness_var = brightness.var(dim=1)
        features.append(brightness_mean.unsqueeze(1))
        features.append(brightness_var.unsqueeze(1))
        
        # 2. Motion (frame difference)
        if t > 1:
            motion = torch.abs(torch.diff(gray, dim=1))  # [B, T-1, H, W]
            motion_mean = motion.mean(dim=(2, 3)).mean(dim=1)  # [B]
            motion_var = motion.mean(dim=(2, 3)).var(dim=1)  # [B]
            features.append(motion_mean.unsqueeze(1))
            features.append(motion_var.unsqueeze(1))
        else:
            features.append(torch.zeros(b, 1, device=images.device))
            features.append(torch.zeros(b, 1, device=images.device))
        
        # 3. Edge energy (simple Sobel)
        edge_energy = torch.zeros(b, device=images.device)
        for bi in range(min(b, 2)):  # Only first 2 samples for efficiency
            for ti in range(min(t, 4)):
                img = gray[bi, ti]
                # Sobel operator
                gx = torch.abs(img[:-2, 1:-1] - img[2:, 1:-1])
                gy = torch.abs(img[1:-1, :-2] - img[1:-1, 2:])
                edge_energy[bi] += (gx + gy).mean()
        features.append(edge_energy.unsqueeze(1) / max(1, b))
        
        # 4. Contrast
        contrast = gray.std(dim=(2, 3)).mean(dim=1)
        features.append(contrast.unsqueeze(1))
        
        relation_feat = torch.cat(features, dim=1)  # [B, 6]
        return relation_feat
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            images: [B, T, C, H, W]
        
        Returns:
            dict with action_logits, group_logits (if hierarchical)
        """
        b, t, c, h, w = images.size()
        
        # ===== Event Stream (Backbone) =====
        x_event = self.backbone.forward_features(images.reshape(b, c, t, h, w))
        x_event = self.event_projection(F.adaptive_avg_pool1d(x_event.transpose(1, 2), self.event_frame_budget).transpose(1, 2))
        x_event = self.event_position(x_event)  # [B, K, D]
        
        # ===== Context Stream =====
        if self.use_context_branch and self.image_model is not None:
            # The frozen Scene Stream follows the paper's 112x112 CLIP input.
            scene_images = F.interpolate(
                images.reshape(b * t, c, h, w), size=(self.context_resolution, self.context_resolution),
                mode="bilinear", align_corners=False,
            )
            context_feat = self.image_model(scene_images)[1]  # [B*T, 768]
            context_feat = context_feat.reshape(b, t, -1).mean(dim=1)  # [B, 768]
            context_proj = self.context_proj(context_feat)  # [B, D]
            
            # Relation features
            relation_feat = self._build_relation_features(images)  # [B, 6]
            
            # Fusion gating
            context_repeat = context_proj.unsqueeze(1).repeat(1, self.event_frame_budget, 1)  # [B, K, D]
            relation_proj = self.relation_proj(relation_feat).unsqueeze(1)  # [B, 1, D]
            
            gate_input = torch.cat([
                x_event,
                context_repeat,
                relation_feat.unsqueeze(1).expand(b, self.event_frame_budget, 6)
            ], dim=2)  # [B, K, 2D+6]
            
            gate = self.relational_fusion_gate(gate_input)  # [B, K, 1]
            # RFG: adaptively inject scene and relation conditions into each event token.
            x_fused = x_event + gate * (context_repeat + relation_proj)
            
            # Query generation
            query_embed = self.query_context_projection(
                torch.concat((
                    self.query_embed.unsqueeze(0).repeat(b, 1, 1),
                    context_feat.unsqueeze(1).repeat(1, self.num_classes, 1)
                ), 2)
            )
        else:
            x_fused = x_event
            query_embed = self.query_context_projection(
                self.query_embed.unsqueeze(0).repeat(b, 1, 1)
            )
        
        # ===== Decoder Input =====
        x = torch.cat((x_fused, query_embed), dim=1)  # [B, K+C, D]
        
        # ===== Mamba Decoder =====
        residual = None
        for layer in self.decoder_blocks:
            x, residual = layer(x, residual)
        
        # ===== Action Head =====
        _, d1, d2 = x.size()
        x_out = self.query_token_projection(x.reshape(b, d2, d1)).reshape(b, self.num_classes, -1)
        action_logits = self.action_classifier(x_out)  # [B, C]
        
        output = {"action_logits": action_logits}
        
        # ===== Hierarchical Decoder =====
        if self.use_hierarchical_decoder and self.group_head is not None:
            # Video representation
            video_repr = x_fused.mean(dim=1)  # [B, D]
            group_logits = self.group_head(video_repr)  # [B, G]
            
            # Group conditioning
            group_cond = self.group_condition_proj(torch.sigmoid(group_logits)).unsqueeze(1)  # [B, 1, D]
            x_action_cond = x_out + group_cond.repeat(1, self.num_classes, 1)
            action_logits_cond = self.action_classifier(x_action_cond)
            
            output["action_logits"] = action_logits_cond
            output["group_logits"] = group_logits
        
        return output


class EcoMambaTrainer:
    """
    Eco-Mamba training and evaluation controller
    """
    
    def __init__(
        self,
        train_loader,
        test_loader,
        criterion,
        eval_metric,
        class_list,
        test_every,
        distributed,
        gpu_id,
        args
    ):
        super().__init__()
        
        self.device = torch.device(gpu_id)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion.to(self.device)
        self.eval_metric = eval_metric.to(self.device)
        self.class_list = class_list
        self.test_every = test_every
        self.distributed = distributed
        self.gpu_id = self.device
        self.args = args
        
        # Extract config from args
        self.freeze_backbone_epochs = getattr(args, 'freeze_backbone_epochs', 100)
        self.finetune_backbone_epochs = getattr(args, 'finetune_backbone_epochs', 150)
        use_hierarchical_decoder = getattr(args, 'use_hierarchical_decoder', True)
        self.use_hierarchical_decoder = use_hierarchical_decoder
        self.eval_group_metrics = getattr(args, 'eval_group_metrics', True)
        
        # Get number of frames
        num_frames = self.train_loader.dataset[0][0].shape[0]
        
        # Get text features from CLIP
        logging.set_verbosity_error()
        class_embed = self._get_text_features(class_list)
        
        # Create model
        model = EcoMamba(
            class_embed=class_embed,
            num_frames=num_frames,
            version=args.videomamba_version,
            event_frame_budget=args.num_frames,
            use_videomamba_pretrained=getattr(args, 'use_videomamba_pretrained', True),
            use_context_branch=getattr(args, 'use_context_branch', True),
            use_hierarchical_decoder=use_hierarchical_decoder,
            decoder_layers=args.decoder_layers,
            clip_model_path=args.clip_model_path,
            pretrained_root=args.pretrained_root,
        ).to(self.device)
        
        if distributed:
            self.model = DDP(model, device_ids=[self.device.index])
        else:
            self.model = model
        
        # Freeze context encoder
        if hasattr(self.model, 'module'):
            model_ref = self.model.module
        else:
            model_ref = self.model
        
        if hasattr(model_ref, 'image_model') and model_ref.image_model is not None:
            for p in model_ref.image_model.parameters():
                p.requires_grad = False
        
        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=0.0001,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10)
        
        # Loss function
        if use_hierarchical_decoder:
            self.eco_loss = HierarchicalEthogramLoss(
                action_to_group=AK_ACTION_TO_GROUP_ID,
                num_groups=NUM_GROUPS,
                action_weight=1.0,
                group_weight=getattr(args, 'group_loss_weight', 1.0),
                consistency_weight=getattr(args, 'hier_loss_weight', 0.2),
            ).to(self.device)
        else:
            self.eco_loss = None
        
        # Checkpoint config - use save_path directory if provided, otherwise use default
        if hasattr(args, 'save_path') and args.save_path:
            # Extract directory from save_path
            save_dir = os.path.dirname(args.save_path)
            if save_dir:
                self.checkpoint_dir = save_dir
            else:
                self.checkpoint_dir = os.path.join('.', 'checkpoints')
        else:
            self.checkpoint_dir = os.path.join('.', 'checkpoints')
        
        self.checkpoint_every = 10
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Logging
        self.log_file = os.path.join(self.checkpoint_dir, 'training.log')
    
    @staticmethod
    def _get_prompt(cl_names):
        return [f"an animal is {class_name}" for class_name in cl_names]
    
    def _get_text_features(self, cl_names):
        text_model = CLIPTextModel.from_pretrained(self.args.clip_model_path)
        tokenizer = CLIPTokenizer.from_pretrained(self.args.clip_model_path)
        
        prompts = self._get_prompt(cl_names)
        texts = tokenizer(prompts, padding=True, return_tensors="pt")
        text_class = text_model(**texts).pooler_output.detach()
        return text_class
    
    def _log(self, msg: str):
        """Write to log file and stdout"""
        print(msg, flush=True)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')
    
    def _train_batch(self, data, label):
        """Train single batch"""
        self.optimizer.zero_grad()
        
        output = self.model(data)
        
        if self.use_hierarchical_decoder and self.eco_loss is not None:
            # Build group targets
            group_targets = torch.from_numpy(
                build_group_targets(label.cpu().numpy(), AK_ACTION_TO_GROUP_ID, NUM_GROUPS)
            ).to(self.device).float()
            
            action_logits = output["action_logits"]
            group_logits = output["group_logits"]
            
            loss, loss_dict = self.eco_loss(
                action_logits=action_logits,
                action_targets=label,
                group_logits=group_logits,
                group_targets=group_targets,
            )
        else:
            action_logits = output["action_logits"]
            loss = self.criterion(action_logits, label)
            loss_dict = {}
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), loss_dict
    
    def _train_epoch(self, epoch: int):
        """Train single epoch"""
        self.model.train()
        loss_meter = AverageMeter()
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            if len(batch) == 3:
                data, label, meta = batch
            else:
                data, label = batch
            
            data = data.to(self.gpu_id, non_blocking=True)
            label = label.to(self.gpu_id, non_blocking=True)
            
            loss, loss_dict = self._train_batch(data, label)
            loss_meter.update(loss, data.shape[0])
        
        elapsed_time = time.time() - start_time
        self.scheduler.step()
        
        if self._is_main_process():
            msg = (f"Epoch [{epoch + 1}] [{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}] "
                  f"loss: {loss_meter.avg:.4f}")
            self._log(msg)
    
    def _is_main_process(self) -> bool:
        return not self.distributed or self.device.index == 0
    
    def save_checkpoint(self, ckpt_path: str, epoch: int):
        """Save checkpoint"""
        os.makedirs(os.path.dirname(ckpt_path) or '.', exist_ok=True)
        model_state = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
        ckpt = {
            'epoch': int(epoch),
            'model': model_state,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        torch.save(ckpt, ckpt_path)
    
    def load_checkpoint(self, ckpt_path: str, map_location=None) -> int:
        """Load checkpoint"""
        ckpt = torch.load(ckpt_path, map_location=map_location)
        if isinstance(ckpt, dict):
            model_state = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
        else:
            model_state = ckpt
        legacy_prefixes = {
            'linear1.': 'event_projection.', 'pos_encod.': 'event_position.',
            'context_gate.': 'relational_fusion_gate.', 'linear2.': 'query_context_projection.',
            'layers.': 'decoder_blocks.', 'linear3.': 'query_token_projection.',
            'group_linear.': 'action_classifier.',
        }
        model_state = {
            next((new + key.removeprefix('module.')[len(old):] for old, new in legacy_prefixes.items() if key.removeprefix('module.').startswith(old)), key.removeprefix('module.')): value
            for key, value in model_state.items()
        }
        
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(model_state, strict=True)
        else:
            self.model.load_state_dict(model_state, strict=True)
        
        if isinstance(ckpt, dict) and 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if isinstance(ckpt, dict) and 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        
        last_epoch = int(ckpt.get('epoch', -1)) if isinstance(ckpt, dict) else -1
        return last_epoch + 1
    
    def _maybe_save_checkpoint(self, epoch: int):
        """Maybe save checkpoint"""
        if not self._is_main_process():
            return
        if self.checkpoint_every <= 0:
            return
        if (epoch + 1) % self.checkpoint_every != 0:
            return
        
        ckpt_name = f'ckpt_epoch_{epoch + 1:04d}.pth'
        ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)
        self.save_checkpoint(ckpt_path, epoch)
        self._log(f"[INFO] Saved checkpoint to: {ckpt_path}")
    
    def train(self, start_epoch: int = 0):
        """Train with the paper's frozen-backbone and joint-fine-tuning stages."""
        # Get model reference
        model_ref = self.model.module if isinstance(self.model, DDP) else self.model
        
        # Stage 1: Frozen backbone
        self._log('[INFO] ===== STAGE 1: Frozen backbone =====')
        for param in model_ref.backbone.parameters():
            param.requires_grad = False
        
        first_stage_end = self.freeze_backbone_epochs
        for epoch in range(start_epoch, first_stage_end):
            self._train_epoch(epoch)
            if (epoch + 1) % self.test_every == 0:
                eval_results = self.test()
                if self._is_main_process():
                    if isinstance(eval_results, dict):
                        msg = "[INFO] Evaluation results: " + ", ".join(
                            f"{k}={v*100:.2f}" for k, v in eval_results.items()
                        )
                    else:
                        msg = f"[INFO] Evaluation Metric: {eval_results * 100:.2f}"
                    self._log(msg)
            
            self._maybe_save_checkpoint(epoch)
        
        # Stage 2: Unfrozen backbone
        self._log('[INFO] ===== STAGE 2: Unfrozen backbone =====')
        for param in model_ref.backbone.parameters():
            param.requires_grad = True
        
        for epoch in range(max(start_epoch, first_stage_end), first_stage_end + self.finetune_backbone_epochs):
            self._train_epoch(epoch)
            if (epoch + 1) % self.test_every == 0:
                eval_results = self.test()
                if self._is_main_process():
                    if isinstance(eval_results, dict):
                        msg = "[INFO] Evaluation results: " + ", ".join(
                            f"{k}={v*100:.2f}" for k, v in eval_results.items()
                        )
                    else:
                        msg = f"[INFO] Evaluation Metric: {eval_results * 100:.2f}"
                    self._log(msg)
            
            self._maybe_save_checkpoint(epoch)
    
    def test(self) -> Dict[str, float]:
        """Test and compute metrics"""
        self.model.eval()
        
        all_action_logits = []
        all_action_targets = []
        all_group_logits = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                if len(batch) == 3:
                    data, label, meta = batch
                else:
                    data, label = batch
                
                data = data.to(self.gpu_id, non_blocking=True)
                label = label.to(self.gpu_id, non_blocking=True)
                
                output = self.model(data)
                action_logits = output["action_logits"]
                
                all_action_logits.append(action_logits.cpu().numpy())
                all_action_targets.append(label.cpu().numpy())
                
                if self.use_hierarchical_decoder and "group_logits" in output:
                    all_group_logits.append(output["group_logits"].cpu().numpy())
        
        all_action_logits = np.concatenate(all_action_logits, axis=0)
        all_action_targets = np.concatenate(all_action_targets, axis=0)
        
        # Compute metrics
        results = {}
        
        # Action mAP - use numerically stable sigmoid
        action_probs = np.where(
            all_action_logits >= 0,
            1 / (1 + np.exp(-all_action_logits)),
            np.exp(all_action_logits) / (1 + np.exp(all_action_logits))
        )
        self.eval_metric.reset()
        # Ensure correct data types for torchmetrics
        # MultilabelAveragePrecision expects preds as float and target as int/long
        action_probs_tensor = torch.from_numpy(action_probs).float()
        action_targets_tensor = torch.from_numpy(all_action_targets).long()
        self.eval_metric.update(action_probs_tensor, action_targets_tensor)
        action_mAP = self.eval_metric.compute().item()
        results["action_mAP"] = action_mAP
        
        # Group metrics
        if self.use_hierarchical_decoder and self.eval_group_metrics and len(all_group_logits) > 0:
            all_group_logits = np.concatenate(all_group_logits, axis=0)
            
            # Group mAP
            group_mAP, _ = compute_group_map(
                all_action_logits,
                all_action_targets,
                AK_ACTION_TO_GROUP_ID,
                AK_GROUP_NAMES
            )
            results["group_mAP"] = group_mAP
            
            # HCR - use numerically stable sigmoid
            action_probs = np.where(
                all_action_logits >= 0,
                1 / (1 + np.exp(-all_action_logits)),
                np.exp(all_action_logits) / (1 + np.exp(all_action_logits))
            )
            group_probs = np.where(
                all_group_logits >= 0,
                1 / (1 + np.exp(-all_group_logits)),
                np.exp(all_group_logits) / (1 + np.exp(all_group_logits))
            )
            hcr = hierarchical_consistency_rate(
                action_probs, group_probs, AK_ACTION_TO_GROUP_ID
            )
            results["HCR"] = hcr
        
        return results
    
    def save_model(self, file_path: str):
        """Save the final model weights to the explicitly requested path."""
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        model = self.model.module if isinstance(self.model, DDP) else self.model
        torch.save(model.state_dict(), file_path)
