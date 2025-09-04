
import torch
import torchaudio
from datasets import load_dataset, load_from_disk
from ai.utils import load_state
from ai.models.baseline_cnn import AudioToDigitModel
from ai.datasets.datasets import SpokenDigitDataset
from ai.datasets.datasets import load_and_preprocess_audio


def predict_digit(model_path=None, run_id=None, device=None, audio_path=None, audio_byte=None, transform=None):
    """
    Load a trained model and perform inference on an audio file.

    Args:
        model_class (torch.nn.Module): Model architecture class.
        model_path (str, optional): Path to .pth model file.
        run_id (str, optional): MLflow run ID to fetch model from.
        device (str, optional): 'cpu' or 'cuda'.
        audio_path (str, optional): Path to audio file to predict.
        transform (callable, optional): Preprocessing transform (e.g., MelSpectrogram + normalization).
    
    Returns:
        int: Predicted digit (0-9).
    """

    model = load_state(model_path=model_path, run_id=run_id, device=device)

    tens = load_and_preprocess_audio(
        audio_path=audio_path,
        audio_byte=audio_byte
    )
    
    with torch.no_grad():
        outputs = model(tens)
        predicted = torch.argmax(outputs, dim=1).item()

    return predicted
