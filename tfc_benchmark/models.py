from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50


MODEL_NAMES = (
    "latefusion",
    "utde",
    "daft",
    "mmtm",
    "aug",
    "inforeg",
    "healnet",
    "flexmoe",
    "drfuse",
    "umse",
    "shaspec",
    "m3care",
    "medfuse",
    "smil",
)


@dataclass
class ModelOutput:
    logits: Tensor
    aux_loss: Tensor
    unimodal_logits: Optional[Tensor] = None
    extras: Optional[Dict[str, Tensor]] = None


def masked_bce(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


class TinyImageEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.network(image)


class ImageEncoder(nn.Module):
    def __init__(
        self, hidden_dim: int, pretrained: bool = True, backbone: str = "resnet50"
    ):
        super().__init__()
        if backbone == "tiny":
            self.network = TinyImageEncoder(hidden_dim)
        elif backbone == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            network = resnet50(weights=weights)
            network.fc = nn.Linear(network.fc.in_features, hidden_dim)
            self.network = network
        else:
            raise ValueError(f"unsupported image backbone: {backbone}")

    def forward(self, image: Tensor) -> Tensor:
        return self.network(image)


class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, clinical: Tensor) -> Tensor:
        return self.network(clinical)


class ThreeModalBackbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        clinical_dim: int = 10,
        dropout: float = 0.2,
        pretrained: bool = True,
        image_backbone: str = "resnet50",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Separate image weights are intentional: tongue and face have different domains.
        self.tongue_encoder = ImageEncoder(
            hidden_dim, pretrained=pretrained, backbone=image_backbone
        )
        self.face_encoder = ImageEncoder(
            hidden_dim, pretrained=pretrained, backbone=image_backbone
        )
        self.clinical_encoder = ClinicalEncoder(clinical_dim, hidden_dim, dropout)
        self.missing_embeddings = nn.Parameter(torch.zeros(3, hidden_dim))
        nn.init.normal_(self.missing_embeddings, std=0.02)

    def encode(self, batch: Mapping[str, Tensor], learned_missing: bool = False) -> Tensor:
        features = torch.stack(
            [
                self.tongue_encoder(batch["tongue_image"]),
                self.face_encoder(batch["face_image"]),
                self.clinical_encoder(batch["clinical"]),
            ],
            dim=1,
        )
        mask = batch["modality_mask"].to(features.dtype).unsqueeze(-1)
        if learned_missing:
            missing = self.missing_embeddings.unsqueeze(0).expand(features.size(0), -1, -1)
            return features * mask + missing * (1.0 - mask)
        return features * mask


class TFCModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_labels: int = 3,
        clinical_dim: int = 10,
        dropout: float = 0.2,
        pretrained: bool = True,
        image_backbone: str = "resnet50",
        **_: object,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.dropout = dropout
        self.backbone = ThreeModalBackbone(
            hidden_dim=hidden_dim,
            clinical_dim=clinical_dim,
            dropout=dropout,
            pretrained=pretrained,
            image_backbone=image_backbone,
        )
        self.unimodal_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, num_labels) for _ in range(3)]
        )

    def encode(self, batch: Mapping[str, Tensor], learned_missing: bool = False) -> Tensor:
        return self.backbone.encode(batch, learned_missing=learned_missing)

    def unimodal_predictions(self, features: Tensor) -> Tensor:
        return torch.stack(
            [head(features[:, index]) for index, head in enumerate(self.unimodal_heads)],
            dim=1,
        )

    def zero(self, features: Tensor) -> Tensor:
        return features.new_zeros(())


class LateFusion(TFCModel):
    def __init__(self, **kwargs: object):
        super().__init__(**kwargs)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_labels),
        )

    def forward(self, batch: Mapping[str, Tensor]) -> ModelOutput:
        features = self.encode(batch)
        logits = self.classifier(features.flatten(1))
        return ModelOutput(logits, self.zero(features), self.unimodal_predictions(features))


class UTDE(TFCModel):
    """Static three-modal adaptation: modality tokens with interleaved self-attention."""

    def __init__(self, num_heads: int = 4, num_layers: int = 2, **kwargs: object):
        super().__init__(**kwargs)
        self.modality_embedding = nn.Parameter(torch.randn(1, 3, self.hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(self.hidden_dim * 3, self.num_labels)

    def forward(self, batch: Mapping[str, Tensor]) -> ModelOutput:
        features = self.encode(batch, learned_missing=False) + self.modality_embedding
        key_padding = ~batch["modality_mask"].bool()
        encoded = self.transformer(features, src_key_padding_mask=key_padding)
        encoded = encoded * batch["modality_mask"].unsqueeze(-1)
        logits = self.classifier(encoded.flatten(1))
        return ModelOutput(logits, self.zero(features), self.unimodal_predictions(features))


class DAFT(TFCModel):
    def __init__(self, **kwargs: object):
        super().__init__(**kwargs)
        self.affine = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim * 2) for _ in range(2)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_labels),
        )

    def forward(self, batch: Mapping[str, Tensor]) -> ModelOutput:
        features = self.encode(batch)
        clinical = features[:, 2]
        modulated = []
        for index in range(2):
            gamma, beta = self.affine[index](clinical).chunk(2, dim=-1)
            modulated.append((1.0 + torch.tanh(gamma)) * features[:, index] + beta)
        fused = torch.stack([modulated[0], modulated[1], clinical], dim=1)
        fused = fused * batch["modality_mask"].unsqueeze(-1)
        return ModelOutput(
            self.classifier(fused.flatten(1)),
            self.zero(features),
            self.unimodal_predictions(features),
        )


class MMTM(TFCModel):
    def __init__(self, compression_ratio: int = 4, **kwargs: object):
        super().__init__(**kwargs)
        compressed = max(self.hidden_dim, self.hidden_dim * 3 // compression_ratio)
        self.squeeze = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, compressed),
            nn.ReLU(inplace=True),
        )
        self.excitation = nn.ModuleList(
            [nn.Linear(compressed, self.hidden_dim) for _ in range(3)]
        )
        self.classifier = nn.Linear(self.hidden_dim * 3, self.num_labels)

    def forward(self, batch: Mapping[str, Tensor]) -> ModelOutput:
        features = self.encode(batch)
        joint = self.squeeze(features.flatten(1))
        recalibrated = torch.stack(
            [
                features[:, index] * torch.sigmoid(head(joint))
                for index, head in enumerate(self.excitation)
            ],
            dim=1,
        )
        return ModelOutput(
            self.classifier(recalibrated.flatten(1)),
            self.zero(features),
            self.unimodal_predictions(features),
        )


class AUG(TFCModel):
    def __init__(self, merge_alpha: float = 0.65, **kwargs: object):
        super().__init__(**kwargs)
        self.merge_alpha = merge_alpha
        self.fusion_head = nn.Linear(self.hidden_dim * 3, self.num_labels)

    def forward(self, batch: Mapping[str, Tensor]) -> ModelOutput:
        features = self.encode(batch)
        unimodal = self.unimodal_predictions(features)
        mask = batch["modality_mask"].unsqueeze(-1)
        average = (unimodal * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        joint = self.fusion_head(features.flatten(1))
        logits = self.merge_alpha * joint + (1.0 - self.merge_alpha) * average
        aux = self.zero(features)
        if "targets" in batch:
            losses = torch.stack(
                [
                    masked_bce(unimodal[:, index], batch["targets"], batch["target_mask"])
                    for index in range(3)
                ]
            )
            # Boost weaker modalities: penalize spread while retaining their supervised loss.
            aux = losses.mean() + 0.25 * (losses.max() - losses.min())
        return ModelOutput(logits, aux, unimodal)


class InfoReg(TFCModel):
    def __init__(self, gamma: float = 0.25, **kwargs: object):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.classifier = nn.Linear(self.hidden_dim * 3, self.num_labels)

    def forward(self, batch: Mapping[str, Tensor]) -> ModelOutput:
        features = self.encode(batch)
        unimodal = self.unimodal_predictions(features)
        logits = self.classifier(features.flatten(1))
        aux = self.zero(features)
        if "targets" in batch:
            losses = torch.stack(
                [
                    masked_bce(unimodal[:, index], batch["targets"], batch["target_mask"])
                    for index in range(3)
                ]
            )
            # Equalize information acquisition across modalities without detaching branches.
            aux = losses.mean() + self.gamma * torch.var(losses, unbiased=False)
        return ModelOutput(logits, aux, unimodal)


class HEALNet(TFCModel):
    def __init__(
        self,
        num_heads: int = 4,
        latent_tokens: int = 16,
        depth: int = 2,
        **kwargs: object,
    ):
        super().__init__(**kwargs)
        self.latents = nn.Parameter(torch.randn(1, latent_tokens, self.hidden_dim) * 0.02)
        self.cross_attention = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.hidden_dim, num_heads, dropout=self.dropout, batch_first=True
                )
                for _ in range(depth)
            ]
        )
        self.feed_forward = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, self.hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim * 4, self.hidden_dim),
                )
                for _ in range(depth)
            ]
        )
        self.classifier = nn.Linear(self.hidden_dim, self.num_labels)

    def forward(self, batch: Mapping[str, Tensor]) -> ModelOutput:
        modalities = self.encode(batch, learned_missing=True)
        latents = self.latents.expand(modalities.size(0), -1, -1)
        key_padding = ~batch["modality_mask"].bool()
        for attention, feed_forward in zip(self.cross_attention, self.feed_forward):
            update, _ = attention(
                latents, modalities, modalities, key_padding_mask=key_padding
            )
            latents = latents + update
            latents = latents + feed_forward(latents)
        return ModelOutput(
            self.classifier(latents.mean(dim=1)),
            self.zero(modalities),
            self.unimodal_predictions(modalities),
        )


class Expert(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value)


class FlexMoE(TFCModel):
    def __init__(
        self, num_experts: int = 8, top_k: int = 2, **kwargs: object
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.experts = nn.ModuleList(
            [Expert(self.hidden_dim * 3, self.dropout) for _ in range(num_experts)]
        )
        self.router = nn.Sequential(
            nn.Linear(3, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, num_experts),
        )
        self.classifier = nn.Linear(self.hidden_dim * 3, self.num_labels)

    def forward(self, batch: Mapping[str, Tensor]) -> ModelOutput:
        features = self.encode(batch, learned_missing=True).flatten(1)
        router_logits = self.router(batch["modality_mask"])
        top_values, top_indices = torch.topk(router_logits, self.top_k, dim=-1)
        weights = torch.softmax(top_values, dim=-1)
        all_expert_outputs = torch.stack(
            [expert(features) for expert in self.experts], dim=1
        )
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, features.size(-1))
        selected = torch.gather(all_expert_outputs, 1, gather_index)
        fused = (selected * weights.unsqueeze(-1)).sum(dim=1)
        importance = torch.softmax(router_logits, dim=-1).mean(dim=0)
        balance = self.num_experts * torch.sum(importance**2) - 1.0
        return ModelOutput(
            self.classifier(fused),
            balance,
            self.unimodal_predictions(features.view(-1, 3, self.hidden_dim)),
            {"router_logits": router_logits},
        )


class DrFuse(TFCModel):
    def __init__(self, align_weight: float = 0.1, **kwargs: object):
        super().__init__(**kwargs)
        self.align_weight = align_weight
        self.shared = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(3)]
        )
        self.specific = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(3)]
        )
        self.attention = nn.Linear(self.hidden_dim, 1)
        self.classifier = nn.Linear(self.hidden_dim * 2, self.num_labels)

    def forward(self, batch: Mapping[str, Tensor]) -> ModelOutput:
        features = self.encode(batch, learned_missing=True)
        shared = torch.stack(
            [layer(features[:, i]) for i, layer in enumerate(self.shared)], dim=1
        )
        specific = torch.stack(
            [layer(features[:, i]) for i, layer in enumerate(self.specific)], dim=1
        )
        mask = batch["modality_mask"].unsqueeze(-1)
        centroid = (shared * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        attention_logits = self.attention(specific).squeeze(-1)
        attention_logits = attention_logits.masked_fill(
            ~batch["modality_mask"].bool(), -1e4
        )
        weights = torch.softmax(attention_logits, dim=1)
        distinct = (specific * weights.unsqueeze(-1)).sum(dim=1)
        alignment = (((shared - centroid.unsqueeze(1)) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)
        orthogonal = torch.abs(F.cosine_similarity(shared, specific, dim=-1))
        orthogonal = (orthogonal * batch["modality_mask"]).sum() / batch[
            "modality_mask"
        ].sum().clamp_min(1.0)
        aux = self.align_weight * (alignment + orthogonal)
        return ModelOutput(
            self.classifier(torch.cat([centroid, distinct], dim=-1)),
            aux,
            self.unimodal_predictions(features),
            {"attention": weights},
        )


class UMSE(TFCModel):
    def __init__(
        self, num_heads: int = 4, bottleneck_tokens: int = 2, **kwargs: object
    ):
        super().__init__(**kwargs)
        self.bottlenecks = nn.Parameter(
            torch.randn(1, bottleneck_tokens, self.hidden_dim) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            self.hidden_dim,
            num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Linear(self.hidden_dim, self.num_labels)

    def forward(self, batch: Mapping[str, Tensor]) -> ModelOutput:
        features = self.encode(batch, learned_missing=True)
        bottlenecks = self.bottlenecks.expand(features.size(0), -1, -1)
        tokens = torch.cat([features, bottlenecks], dim=1)
        encoded = self.transformer(tokens)
        fused = encoded[:, 3:].mean(dim=1)
        return ModelOutput(
            self.classifier(fused),
            self.zero(features),
            self.unimodal_predictions(features),
        )


class ShaSpec(TFCModel):
    def __init__(
        self, shared_weight: float = 0.05, domain_weight: float = 0.05, **kwargs: object
    ):
        super().__init__(**kwargs)
        self.shared_weight = shared_weight
        self.domain_weight = domain_weight
        self.shared = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(3)]
        )
        self.specific = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(3)]
        )
        self.domain_classifier = nn.Linear(self.hidden_dim, 3)
        self.classifier = nn.Linear(self.hidden_dim * 2, self.num_labels)

    def forward(self, batch: Mapping[str, Tensor]) -> ModelOutput:
        features = self.encode(batch, learned_missing=True)
        shared = torch.stack(
            [self.shared[i](features[:, i]) for i in range(3)], dim=1
        )
        specific = torch.stack(
            [self.specific[i](features[:, i]) for i in range(3)], dim=1
        )
        mask = batch["modality_mask"].unsqueeze(-1)
        shared_mean = (shared * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        specific_mean = (specific * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        consistency = (((shared - shared_mean.unsqueeze(1)) ** 2) * mask).sum()
        consistency = consistency / mask.sum().clamp_min(1.0)
        valid = batch["modality_mask"].bool().reshape(-1)
        domain_targets = (
            torch.arange(3, device=features.device)
            .view(1, 3)
            .expand(features.size(0), -1)
            .reshape(-1)
        )
        domain_logits = self.domain_classifier(specific.reshape(-1, self.hidden_dim))
        domain_loss = (
            F.cross_entropy(domain_logits[valid], domain_targets[valid])
            if valid.any()
            else self.zero(features)
        )
        aux = self.shared_weight * consistency + self.domain_weight * domain_loss
        return ModelOutput(
            self.classifier(torch.cat([shared_mean, specific_mean], dim=-1)),
            aux,
            self.unimodal_predictions(features),
        )


class M3Care(TFCModel):
    def __init__(self, stability_weight: float = 0.01, **kwargs: object):
        super().__init__(**kwargs)
        self.stability_weight = stability_weight
        self.gates = nn.ModuleList(
            [nn.Linear(self.hidden_dim * 2, self.hidden_dim) for _ in range(3)]
        )
        self.classifier = nn.Linear(self.hidden_dim * 3, self.num_labels)

    def forward(self, batch: Mapping[str, Tensor]) -> ModelOutput:
        features = self.encode(batch, learned_missing=True)
        mask = batch["modality_mask"]
        completed = features.clone()
        stability = self.zero(features)
        for index in range(3):
            observed = mask[:, index].bool()
            missing = ~observed
            if missing.any() and observed.any():
                reference = features[observed, index]
                context = features[:, [j for j in range(3) if j != index]].mean(dim=1)
                ref_context = context[observed]
                similarity = torch.softmax(
                    context[missing] @ ref_context.t() / math.sqrt(self.hidden_dim),
                    dim=-1,
                )
                imputed = similarity @ reference
                gate = torch.sigmoid(
                    self.gates[index](
                        torch.cat([context[missing], imputed], dim=-1)
                    )
                )
                completed[missing, index] = gate * imputed + (1.0 - gate) * context[missing]
                stability = stability + torch.mean((imputed - context[missing]) ** 2)
        return ModelOutput(
            self.classifier(completed.flatten(1)),
            self.stability_weight * stability,
            self.unimodal_predictions(features),
        )


class MedFuse(TFCModel):
    def __init__(self, **kwargs: object):
        super().__init__(**kwargs)
        self.lstm = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(self.hidden_dim * 2, self.num_labels)

    def forward(self, batch: Mapping[str, Tensor]) -> ModelOutput:
        features = self.encode(batch, learned_missing=True)
        # Keep sequence positions stable and use the final valid-state weighted pooling.
        sequence, _ = self.lstm(features)
        mask = batch["modality_mask"].unsqueeze(-1)
        pooled = (sequence * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return ModelOutput(
            self.classifier(pooled),
            self.zero(features),
            self.unimodal_predictions(features),
        )


class SMIL(TFCModel):
    def __init__(self, reconstruction_weight: float = 0.1, **kwargs: object):
        super().__init__(**kwargs)
        self.reconstruction_weight = reconstruction_weight
        self.reconstructors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for _ in range(3)
            ]
        )
        self.classifier = nn.Linear(self.hidden_dim * 3, self.num_labels)

    def forward(self, batch: Mapping[str, Tensor]) -> ModelOutput:
        raw = self.encode(batch, learned_missing=False)
        mask = batch["modality_mask"].unsqueeze(-1)
        observed_mean = (raw * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        completed = raw.clone()
        reconstruction = self.zero(raw)
        for index in range(3):
            estimate = self.reconstructors[index](observed_mean)
            missing = ~batch["modality_mask"][:, index].bool()
            completed[missing, index] = estimate[missing]
            observed = ~missing
            if observed.any():
                reconstruction = reconstruction + F.mse_loss(
                    estimate[observed], raw[observed, index]
                )
        return ModelOutput(
            self.classifier(completed.flatten(1)),
            self.reconstruction_weight * reconstruction,
            self.unimodal_predictions(raw),
        )


MODEL_REGISTRY = {
    "latefusion": LateFusion,
    "utde": UTDE,
    "daft": DAFT,
    "mmtm": MMTM,
    "aug": AUG,
    "inforeg": InfoReg,
    "healnet": HEALNet,
    "flexmoe": FlexMoE,
    "drfuse": DrFuse,
    "umse": UMSE,
    "shaspec": ShaSpec,
    "m3care": M3Care,
    "medfuse": MedFuse,
    "smil": SMIL,
}


def build_model(name: str, **kwargs: object) -> TFCModel:
    normalized = name.lower().replace("-", "")
    aliases = {"late_fusion": "latefusion", "flex_moe": "flexmoe"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model {name!r}; available models: {', '.join(MODEL_NAMES)}"
        )
    return MODEL_REGISTRY[normalized](**kwargs)

