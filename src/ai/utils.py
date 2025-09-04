import torch
from ai.models.baseline_cnn import AudioToDigitModel

def accuracy_sc(outputs: torch.Tensor, labels: torch.Tensor):
    indices = torch.argmax(outputs, dim=1)
    correct_preds = (indices == labels).sum().item()
    return correct_preds/labels.size(0)


import os
import torch
import mlflow


def save_model(state, model_name="digit_cnn", dir_path="model_checkpoint", is_best=False):
    """
    Save a PyTorch model locally and optionally log it as an artifact in MLflow.

    Args:
        state (dict): The training state.
        model_name (str): Name of the model file (without extension).
        dir_path (str): Directory to save the model.
        is_best (bool): Whether this is the current best model.
    
    Returns:
        str: Path to the saved model file.
    """
    os.makedirs(dir_path, exist_ok=True)

    # full save path
    model_path = os.path.join(dir_path, f"{model_name}.pth")

    # save state dict
    torch.save(state, model_path)
    print(f"[INFO] Model saved to {model_path}")

    # log as artifact in MLflow
    if is_best:
        mlflow.log_artifact(model_path, artifact_path="model")
        print(f"[INFO] Model logged to MLflow at artifact_path='model'")

    return model_path

def load_state(model_path=None, run_id=None, device=None):
    """
    Load a PyTorch model either from a local file or from an MLflow run.

    Args:
        model_class (torch.nn.Module): The model class (not an instance). Example: DigitCNN (not DigitCNN()).
        model_path (str, optional): Path to the saved .pth file. If provided, loads from local.
        run_id (str, optional): MLflow run ID to load from artifacts.
        device (str, optional): Device to map the model to ('cpu' or 'cuda').
    
    Returns:
        torch.nn.Module: The loaded model instance.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AudioToDigitModel()

    if model_path:
        # Load from local .pth file
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict["state_dict"])
        print(f"[INFO] Loaded model from {model_path}")

    elif run_id:
        # Load from MLflow artifacts
        model_uri = f"runs:/{run_id}/model/digit_cnn.pth"
        local_path = mlflow.artifacts.download_artifacts(model_uri)
        state_dict = torch.load(local_path, map_location=device)
        model.load_state_dict(state_dict["state_dict"])
        print(f"[INFO] Loaded model from MLflow run {run_id}")

    else:
        raise ValueError("You must provide either model_path or run_id")

    return model