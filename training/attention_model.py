from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


class ResNet18AttentionPooling(nn.Module):
    """
    Subject-level AD/CN classifier using:

    1. ResNet18 to extract one 512-dimensional feature vector per slice
    2. Learned attention weights across all slices
    3. A subject-level classifier
    """

    def __init__(
        self,
        checkpoint_path,
        attention_hidden_size=128,
        freeze_feature_extractor=True,
    ):
        super().__init__()

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Experiment C checkpoint not found: {checkpoint_path}"
            )

        # Build the same ResNet18 architecture used in Experiment C.
        resnet = models.resnet18(weights=None)
        resnet.fc = nn.Linear(resnet.fc.in_features, 2)

        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
        )
        resnet.load_state_dict(state_dict)

        # Remove the original two-class FC layer.
        # The remaining network returns a 512-dimensional feature vector.
        self.feature_extractor = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        if freeze_feature_extractor:
            for parameter in self.feature_extractor.parameters():
                parameter.requires_grad = False

        # Learns one importance score for every slice feature.
        self.attention = nn.Sequential(
            nn.Linear(512, attention_hidden_size),
            nn.Tanh(),
            nn.Linear(attention_hidden_size, 1),
        )

        # Uses the aggregated subject feature for AD/CN prediction.
        self.classifier = nn.Linear(512, 2)

        self.freeze_feature_extractor = freeze_feature_extractor

    def train(self, mode=True):
        """
        Keep the frozen ResNet feature extractor in evaluation mode,
        while allowing the attention and classifier modules to train.
        """
        super().train(mode)

        if self.freeze_feature_extractor:
            self.feature_extractor.eval()

        return self

    def forward(self, images):
        """
        Args:
            images:
                Tensor of shape:
                [batch_size, num_slices, 3, 224, 224]

        Returns:
            logits:
                [batch_size, 2]

            attention_weights:
                [batch_size, num_slices]
        """

        batch_size, num_slices, channels, height, width = images.shape

        # Treat every slice as an independent image for ResNet.
        images = images.reshape(
            batch_size * num_slices,
            channels,
            height,
            width,
        )

        # Since the feature extractor is frozen, gradients are unnecessary.
        if self.freeze_feature_extractor:
            with torch.no_grad():
                features = self.feature_extractor(images)
        else:
            features = self.feature_extractor(images)

        # [B × S, 512, 1, 1] -> [B, S, 512]
        features = features.flatten(start_dim=1)
        features = features.reshape(
            batch_size,
            num_slices,
            512,
        )

        # Produce one raw attention score per slice.
        attention_scores = self.attention(features).squeeze(-1)

        # Normalize across the 30 slices.
        attention_weights = torch.softmax(
            attention_scores,
            dim=1,
        )

        # Weighted subject-level feature vector.
        subject_features = torch.sum(
            features * attention_weights.unsqueeze(-1),
            dim=1,
        )

        logits = self.classifier(subject_features)

        return logits, attention_weights