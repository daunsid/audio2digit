import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, classification_report
from ai.datasets.datasets import SpokenDigitDataset
from src.ai.models.baseline_cnn import AudioToDigitModel
from ai.preprocessing import Preprocess, collate_fn
from ai.utils import load_state
from src import settings


experiment = mlflow.get_experiment_by_name(settings.EXPERIMENT_NAME)
if experiment is None:
    mlflow.create_experiment(settings.EXPERIMENT_NAME)
mlflow.set_experiment(settings.EXPERIMENT_NAME)


def main():

    ds = load_dataset(settings.DATASET_NAME)

    test_ds = ds['test']
    preprocess = Preprocess()
    test_data = SpokenDigitDataset(test_ds, transform=preprocess)
    
    test_loader = DataLoader(
        test_data,
        batch_size=settings.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = load_state(model_path=settings.MODEL_PATH)
    model.to(settings.DEVICE)

    # Evaluate
    accuracy, report = evaluate(test_loader, model, settings.DEVICE)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)

    # Log metrics to MLflow
    with mlflow.start_run():
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_text(report, "classification_report.txt")


def evaluate(loader: DataLoader, model: AudioToDigitModel, device: str):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for mels, labels in loader:
            mels, labels = mels.to(device), labels.to(device)

            # compute output
            outputs = model(mels)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    return acc, report


if __name__ == "__main__":
    main()
