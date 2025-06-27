import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import MultiFeatureCNN_BiLSTM
from utils import SpoofDataset
import matplotlib.pyplot as plt


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.show()


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for (mfcc, mel, cqt, cqcc), labels in val_loader:
            mfcc, mel, cqt, cqcc, labels = mfcc.to(device), mel.to(device), cqt.to(device), cqcc.to(device), labels.to(device)

            outputs = model(mfcc, mel, cqt, cqcc).squeeze(1)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total * 100
    return total_loss / len(val_loader), val_acc


def train(model, train_loader, dev_loader, optimizer, criterion, device, epochs=10, save_path="best_model.pth"):
    best_val_loss = float('inf')

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        # Progress bar
        progress_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")

        for (mfcc, mel, cqt, cqcc), labels in progress_bar:
            mfcc, mel, cqt, cqcc = mfcc.to(device), mel.to(device), cqt.to(device), cqcc.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(mfcc, mel, cqt, cqcc).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = (outputs >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total * 100
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation step
        val_loss, val_acc = validate(model, dev_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Epoch result summary
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Always save best model based on val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("✅ Saved best model!\n")

    # Final metric plotting
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

