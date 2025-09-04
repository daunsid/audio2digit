
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
import mlflow
import mlflow.pytorch
import dagshub
from ai.datasets.datasets import SpokenDigitDataset
from ai.models.baseline_cnn import AudioToDigitModel
from ai.preprocessing import Preprocess, collate_fn
from ai.utils import accuracy_sc
from ai.utils import save_model
from src import settings


mlruns_path = os.path.join(os.getcwd(), "mlruns")
mlflow.set_tracking_uri(f"file://{mlruns_path}")
tracking_uri = mlflow.get_tracking_uri()
print(f"Current tracking uri: {tracking_uri}")

experiment = mlflow.get_experiment_by_name(settings.EXPERIMENT_NAME)
if experiment is None:
    mlflow.create_experiment(settings.EXPERIMENT_NAME)
mlflow.set_experiment(settings.EXPERIMENT_NAME)


def main():
    best_acc = 0.0
    is_best = False
    ds = load_dataset(settings.DATASET_NAME)
    train_valid_ds = ds['train'].train_test_split(
        test_size=(1-settings.TRAIN_SPLIT)
    )
    train_ds = train_valid_ds['train']
    valid_ds = train_valid_ds['test']
    preprocess = Preprocess()
    train_data = SpokenDigitDataset(train_ds, transform=preprocess)
    valid_data = SpokenDigitDataset(valid_ds, transform=preprocess)

    train_loader = DataLoader(
        train_data,
        batch_size=settings.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=settings.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = AudioToDigitModel()
    model.to(settings.DEVICE)

    criterion = nn.CrossEntropyLoss().to(settings.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.LR)

    # Scheduler
    if settings.LR_SCHEDULER == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            patience=2
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.7
        )

    # Start MLflow run
    with mlflow.start_run():
        mlflow.log_param("batch_size", settings.BATCH_SIZE)
        mlflow.log_param("learning_rate", settings.LR)

        for epoch in range(settings.EPOCHS):
            train_loss, train_acc = train_epoch(
                train_loader,
                model,
                optimizer,
                criterion,
                settings.DEVICE
            )
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)

            val_loss, val_acc = validate_epoch(
                valid_loader,
                model,
                criterion,
                settings.DEVICE
            )
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            # Log model checkpoint
            if best_acc < val_acc:
                best_acc = val_acc
                is_best = True
            save_model(
                {
                    'state_dict': model.state_dict(),
                },
                is_best=is_best
            )

            # Scheduler step
            if settings.LR_SCHEDULER == 'ReduceLROnPlateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

    return model


def train_epoch(
    train_loader: DataLoader,
    model: AudioToDigitModel,
    optimizer: torch.optim.SGD,
    criterion: nn.CrossEntropyLoss,
    device
):
    running_loss = 0.0
    accuracy = 0.0

    model.train()
    for i, (mels, labels) in enumerate(train_loader):

        mels = mels.to(device)
        labels = labels.to(device)

        # compute output
        output = model(mels)
        loss = criterion(output, labels)

        # measure accuracy and record loss
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        accuracy += accuracy_sc(output, labels)

    avg_loss = running_loss / len(train_loader)
    accuracy = accuracy / len(train_loader)
    return avg_loss, accuracy


def validate_epoch(
    valid_loader: DataLoader,
    model: AudioToDigitModel,
    criterion,
    device: str
):
    model.eval()
    running_loss = 0.0
    accuracy = 0.0

    with torch.no_grad():
        for i, (mels, labels) in enumerate(valid_loader):
            mels, labels = mels.to(device), labels.to(device)

            # compute output
            output = model(mels)
            loss = criterion(output, labels)

            running_loss += loss.item()
            accuracy += accuracy_sc(output, labels)

    avg_loss = running_loss / len(valid_loader)
    accuracy = accuracy / len(valid_loader)
    return avg_loss, accuracy


if __name__ == "__main__":
    main()