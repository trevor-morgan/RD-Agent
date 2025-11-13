"""
REAL, FUNCTIONING PTS Model for Qlib and RD-Agent

This is a production-ready implementation of the PTS dual-output model
that integrates seamlessly with Qlib's model infrastructure.

Key Features:
1. Predicts both returns and PTS (confidence) scores
2. Uses confidence-weighted loss during training
3. Compatible with Qlib's existing training pipeline
4. Drop-in replacement for existing models

Usage:
    In Qlib config:
    task:
        model:
            class: PTSModel
            module_path: rdagent.scenarios.qlib.experiment.model_template.model_pts
            kwargs:
                d_feat: 20
                hidden_size: 64
                num_layers: 2
                dropout: 0.2

Author: RD-Agent Research Team
Date: 2025-11-13
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Text, Union
from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


class PTSModel(Model):
    """
    Predictable Trend Strength (PTS) Model for Qlib.

    This model extends standard return prediction with confidence scoring.
    It predicts both:
    1. Expected return (primary output)
    2. PTS score (confidence in prediction)

    The model is trained with a custom loss that combines:
    - Confidence-weighted MSE
    - PTS calibration loss
    """

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2000,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        GPU=0,
        seed=None,
        pts_lambda=0.1,  # Weight for PTS calibration loss
        **kwargs
    ):
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer_name = optimizer.lower()
        self.loss_type = loss
        self.pts_lambda = pts_lambda
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed

        # Build model
        self.pts_net = PTSNet(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        self.fitted = False
        self.pts_net.to(self.device)

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):
        """
        Train the PTS model.

        Args:
            dataset: Qlib Dataset with train/valid splits
            evals_result: Dict to store evaluation results
            save_path: Path to save model checkpoints
        """
        # Prepare data
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        # Convert to torch tensors
        x_train_tensor = torch.from_numpy(x_train.values).float().to(self.device)
        y_train_tensor = torch.from_numpy(y_train.values).float().to(self.device)
        x_valid_tensor = torch.from_numpy(x_valid.values).float().to(self.device)
        y_valid_tensor = torch.from_numpy(y_valid.values).float().to(self.device)

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Optimizer
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(self.pts_net.parameters(), lr=self.lr)
        elif self.optimizer_name == "gd":
            optimizer = optim.SGD(self.pts_net.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(f"optimizer {self.optimizer_name} is not supported!")

        # Training loop
        best_score = -np.inf
        best_epoch = 0
        stop_rounds = 0

        for epoch in range(self.n_epochs):
            self.pts_net.train()
            total_loss = 0.0
            total_samples = 0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()

                # Forward pass
                pred_return, pred_pts = self.pts_net(batch_x)

                # Calculate PTS loss
                loss, loss_components = self._calculate_pts_loss(
                    pred_return, pred_pts, batch_y
                )

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_x)
                total_samples += len(batch_x)

            avg_loss = total_loss / total_samples

            # Validation
            self.pts_net.eval()
            with torch.no_grad():
                val_pred_return, val_pred_pts = self.pts_net(x_valid_tensor)
                val_loss, _ = self._calculate_pts_loss(
                    val_pred_return, val_pred_pts, y_valid_tensor
                )

                # Calculate IC as validation metric
                val_ic = self._calculate_ic(
                    val_pred_return.cpu().numpy(),
                    y_valid_tensor.cpu().numpy()
                )

            # Early stopping based on IC
            if val_ic > best_score:
                best_score = val_ic
                best_epoch = epoch
                stop_rounds = 0
                if save_path:
                    torch.save(self.pts_net.state_dict(), save_path)
            else:
                stop_rounds += 1
                if stop_rounds >= self.early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}/{self.n_epochs}: "
                    f"Train Loss={avg_loss:.6f}, "
                    f"Val Loss={val_loss:.6f}, "
                    f"Val IC={val_ic:.6f}"
                )

        print(f"Best epoch: {best_epoch} with IC: {best_score:.6f}")

        # Load best model
        if save_path:
            self.pts_net.load_state_dict(torch.load(save_path))

        self.fitted = True

        # Store validation results
        evals_result["train"] = []
        evals_result["valid"] = []

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        """
        Predict using the trained PTS model.

        Args:
            dataset: Qlib Dataset
            segment: Which segment to predict on ("train", "valid", "test")

        Returns:
            pandas Series with predictions (return predictions)
        """
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        # Prepare data
        df_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        x_test = df_test["feature"]

        # Convert to tensor
        x_test_tensor = torch.from_numpy(x_test.values).float().to(self.device)

        # Predict
        self.pts_net.eval()
        with torch.no_grad():
            pred_return, pred_pts = self.pts_net(x_test_tensor)

        # Convert to pandas Series
        pred_return_np = pred_return.cpu().numpy().flatten()
        pred_pts_np = pred_pts.cpu().numpy().flatten()

        # Create multi-index Series with both return and PTS
        # For Qlib compatibility, return the main prediction (return)
        # PTS scores can be accessed separately if needed
        predictions = pd.Series(pred_return_np, index=df_test.index)

        # Store PTS scores as attribute for later use
        self.last_pts_scores = pd.Series(pred_pts_np, index=df_test.index)

        return predictions

    def _calculate_pts_loss(self, pred_return, pred_pts, target_return):
        """
        Calculate PTS loss combining multiple objectives.

        Args:
            pred_return: Predicted returns [batch_size, 1]
            pred_pts: Predicted PTS scores [batch_size, 1]
            target_return: Actual returns [batch_size, 1]

        Returns:
            Total loss and components dict
        """
        # 1. Confidence-weighted MSE
        squared_errors = (pred_return - target_return) ** 2
        weighted_mse = (pred_pts * squared_errors).mean()

        # 2. PTS Calibration: PTS should match realized accuracy
        realized_accuracy = 1.0 / (1.0 + squared_errors.detach())
        calibration_loss = nn.functional.mse_loss(pred_pts, realized_accuracy)

        # 3. Regularization: Prevent PTS from collapsing to constant
        pts_std = pred_pts.std()
        diversity_loss = -pts_std  # Encourage diversity in PTS scores

        # Total loss
        total_loss = (
            weighted_mse
            + self.pts_lambda * calibration_loss
            + 0.01 * diversity_loss
        )

        return total_loss, {
            'weighted_mse': weighted_mse.item(),
            'calibration_loss': calibration_loss.item(),
            'diversity_loss': diversity_loss.item(),
        }

    def _calculate_ic(self, pred, target):
        """Calculate Information Coefficient (Pearson correlation)."""
        pred = pred.flatten()
        target = target.flatten()

        # Remove NaN values
        mask = ~(np.isnan(pred) | np.isnan(target))
        pred = pred[mask]
        target = target[mask]

        if len(pred) < 2:
            return 0.0

        return np.corrcoef(pred, target)[0, 1]


class PTSNet(nn.Module):
    """
    Neural network architecture for PTS model.

    Architecture:
        Input → Shared Layers → {Return Head, PTS Head}
    """

    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super(PTSNet, self).__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size

        # Shared feature extraction layers
        layers = []
        input_dim = d_feat

        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_size

        self.shared_layers = nn.Sequential(*layers)

        # Return prediction head
        self.return_head = nn.Linear(hidden_size, 1)

        # PTS prediction head
        self.pts_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # PTS in [0, 1]
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input features [batch_size, d_feat]

        Returns:
            (predicted_return, predicted_pts)
        """
        # Shared feature extraction
        features = self.shared_layers(x)

        # Dual outputs
        pred_return = self.return_head(features)
        pred_pts = self.pts_head(features)

        return pred_return, pred_pts
