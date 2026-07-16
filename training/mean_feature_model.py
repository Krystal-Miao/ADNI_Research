from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


class ResNet18MeanFeaturePooling(nn.Module):
    """
    Subject-level AD/CN classifier using mean feature aggregation.

    Pipeline:
        MRI slices
        -> frozen ResNet18 feature extractor
        -> one 512-dimensional feature per slice
        -> average features across slices
        -> subject-level classifier
    """

    def __init__(
        self,
        checkpoint_path,
        freeze_feature_extractor=True,
    ):
        super().__init__()

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Experiment C checkpoint not found: {checkpoint_path}"
            )

        # Rebuild the Experiment C ResNet18 architecture.
        resnet = models.resnet18(weights=None)
        resnet.fc = nn.Linear(resnet.fc.in_features, 2)

        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
        )
        resnet.load_state_dict(state_dict)

        # Remove the original slice-level classifier.
        # Output shape becomes [N, 512, 1, 1].
        self.feature_extractor = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        if freeze_feature_extractor:
            for parameter in self.feature_extractor.parameters():
                parameter.requires_grad = False

        # New subject-level classifier.
        self.classifier = nn.Linear(512, 2)

        self.freeze_feature_extractor = freeze_feature_extractor

    def train(self, mode=True):
        """
        Keep the frozen ResNet feature extractor in evaluation mode.
        """
        super().train(mode)

        if self.freeze_feature_extractor:
            self.feature_extractor.eval()

        return self

    def forward(self, images):
        """
        Args:
            images:
                [batch_size, num_slices, 3, 224, 224]

        Returns:
            logits:
                [batch_size, 2]

            subject_features:
                [batch_size, 512]
        """

        (
            batch_size,
            num_slices,
            channels,
            height,
            width,
        ) = images.shape

        # Combine subject and slice dimensions so ResNet can process
        # every slice as an independent image.
        images = images.reshape(
            batch_size * num_slices,
            channels,
            height,
            width,
        )

        if self.freeze_feature_extractor:
            with torch.no_grad():
                slice_features = self.feature_extractor(images)
        else:
            slice_features = self.feature_extractor(images)

        # [B*S, 512, 1, 1] -> [B*S, 512]
        slice_features = slice_features.flatten(start_dim=1)

        # [B*S, 512] -> [B, S, 512]
        slice_features = slice_features.reshape(
            batch_size,
            num_slices,
            512,
        )

        # Every slice feature contributes equally.
        subject_features = slice_features.mean(dim=1)

        logits = self.classifier(subject_features)

        return logits, subject_features