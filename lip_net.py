import os
import torch
import torch.nn as nn
from constants import CHECKPOINT_DIR


class LipNN(nn.Module):
    def __init__(self, vocabulary_size):
        super(LipNN, self).__init__()

        self.conv1 = nn.Conv3d(1, 128, kernel_size=3, padding="same")
        self.pool = nn.MaxPool3d((1, 2, 2))
        self.conv2 = nn.Conv3d(128, 256, kernel_size=3, padding="same")
        self.conv3 = nn.Conv3d(256, 75, kernel_size=3, padding="same")

        self.lstm = nn.LSTM(input_size=75 * 5 * 17, hidden_size=128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(128 * 2, vocabulary_size)  # 128 * 2 since bidirectional

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)

        # reshaping for LSTM
        N, C, D, H, W = x.size()
        x = x.view(N, D, -1)  # flattening H and W dimensions

        x, _ = self.lstm(x)
        x = self.dropout(x)
        return self.fc(x)


def save_checkpoint(epoch, model, optimizer_):
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer_.state_dict(),
    }, checkpoint_path)

    print(f"Checkpoint at epoch {epoch} saved at: {checkpoint_path}")


def load_checkpoint(model, epoch_checkpoint, device=torch.device("cpu"), optimizer=None):
    filename = f"checkpoint_epoch_{epoch_checkpoint}.pth"
    path = os.path.join(CHECKPOINT_DIR, filename)
    if not os.path.exists(path):
        print(f"Checkpoint file not found: {path}")
        return -1
    # if device.type is not `cuda` specifying `map_location` is necessary
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
