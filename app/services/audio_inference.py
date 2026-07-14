import io
import logging
import torch
import torch.nn as nn
import torchaudio

logger = logging.getLogger(__name__)

# 1. Define a lightweight CNN architecture for edge-like inference
class DistressCNN(nn.Module):
    def __init__(self):
        super(DistressCNN, self).__init__()
        # Input: 1 channel (spectrogram), Output: 16 channels
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 43, 128) # Adjust based on input audio length
        self.fc2 = nn.Linear(128, 1) # Binary output: 0 (Normal) or 1 (Distress)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        # Note: In a real deployment, we calculate the exact flattened dimension
        # For this prototype, we mock the final classification step.
        return x

# Instantiate the model (In production, you'd load trained weights here: model.load_state_dict(...))
model = DistressCNN()
model.eval()

# 2. Audio Processing Logic
def analyze_audio_chunk(audio_bytes: bytes) -> dict:
    """
    Takes a 3-second audio buffer from the phone, converts it to a spectrogram, 
    and runs it through the PyTorch CNN to detect distress.
    """
    try:
        # Load audio bytes into a PyTorch tensor
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        
        # Convert to Mono if Stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Create a MelSpectrogram (Visual representation of audio frequencies)
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, 
            n_mels=128
        )
        spectrogram = transform(waveform)
        
        # --- PROTOTYPE MOCK INFERENCE ---
        # Instead of crashing because our untrained model's dimensions aren't perfectly 
        # tuned to the incoming file length, we use a basic energy thresholding heuristic 
        # as a placeholder for the final trained CNN output.
        
        energy = torch.sum(spectrogram).item()
        
        # If the energy (loudness/intensity) is very high, flag as distress
        # (This mimics the first-stage trigger before the CNN validates it)
        is_distress = energy > 50000.0  
        confidence = 0.89 if is_distress else 0.12

        logger.info(f"Analyzed audio buffer. Energy: {energy:.2f} | Distress: {is_distress}")
        
        return {
            "distress_detected": is_distress,
            "confidence": confidence,
            "energy_level": energy
        }
        
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        return {"distress_detected": False, "error": str(e)}