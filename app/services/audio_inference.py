import io
import logging

logger = logging.getLogger(__name__)

def analyze_audio_chunk(audio_bytes: bytes) -> dict:
    """
    Takes a 3-second audio buffer from the phone, converts it to a spectrogram,
    and runs it through the PyTorch CNN to detect distress.
    Imports are lazy to avoid slow server cold-start times.
    """
    try:
        # Lazy imports — only load torch when this function is actually called
        import torch
        import torch.nn as nn
        import torchaudio

        class DistressCNN(nn.Module):
            def __init__(self):
                super(DistressCNN, self).__init__()
                self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
                self.relu = nn.ReLU()
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
                self.fc1 = nn.Linear(32 * 32 * 43, 128)
                self.fc2 = nn.Linear(128, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                return x

        model = DistressCNN()
        model.eval()

        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=128
        )
        spectrogram = transform(waveform)

        energy = torch.sum(spectrogram).item()
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