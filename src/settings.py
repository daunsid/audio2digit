
# Dataset
DATASET_NAME = "mteb/free-spoken-digit-dataset"
TRAIN_SPLIT = 0.7

# mel spectrogram
SAMPLE_RATE = 8000
N_FFT = 256
HOP_LENGTH = 128
N_MELS = 40

# dataset
BATCH_SIZE = 32

# model
DEVICE = "cpu"
EPOCHS = 10
LR = 1e-3
LR_SCHEDULER = "ReduceLROnPlateau"
EARLY_STOPING_PATIENCE = 5
CHECK_POINT_DIR = "/kaggle/working/"
MODEL_PATH="model_checkpoint/digit_cnn.pth"
TRACKING_PATH="file:////mlruns"
EXPERIMENT_NAME = "audio2digit"
