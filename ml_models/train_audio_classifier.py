import os
import glob
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

# Hyperparameters
SAMPLE_RATE = 16000
CHUNK_LENGTH_SEC = 3
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH_SEC
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001

class AudioDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.is_train = is_train
        self.file_paths = []
        self.labels = []
        
        # Expecting directory structure:
        # data/
        #   distress/ (label 1)
        #   noise/ (label 0)
        
        distress_files = glob.glob(os.path.join(data_dir, 'distress', '*.wav'))
        noise_files = glob.glob(os.path.join(data_dir, 'noise', '*.wav'))
        
        for f in distress_files:
            self.file_paths.append(f)
            self.labels.append(1)
            
        for f in noise_files:
            self.file_paths.append(f)
            self.labels.append(0)
            
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        waveform, sr = torchaudio.load(path)
        
        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = T.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
            
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Pad or truncate to exact chunk length
        if waveform.shape[1] > CHUNK_SAMPLES:
            max_start = waveform.shape[1] - CHUNK_SAMPLES
            start = random.randint(0, max_start)
            waveform = waveform[:, start:start+CHUNK_SAMPLES]
        else:
            padding = CHUNK_SAMPLES - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        # Data Augmentation (Train only)
        if self.is_train:
            # 1. Add white noise
            if random.random() > 0.5:
                noise = torch.randn_like(waveform) * 0.005
                waveform = waveform + noise
            # 2. Random gain
            if random.random() > 0.5:
                gain = random.uniform(0.5, 1.5)
                waveform = waveform * gain
                
        # Convert to Mel-Spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)
        
        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
        
        return mel_spec, torch.tensor(label, dtype=torch.float32)

class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        
        # Calculate fully connected input size based on MelSpec dimensions
        # MelSpec shape for 3s @ 16kHz with 512 hop length -> 1x64x94
        # After 3 pooling layers (div by 8) -> 8x11
        # 64 channels * 8 * 11 = 5632
        self.fc1 = nn.Linear(64 * 8 * 11, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Ensure data directory exists
    os.makedirs('data/distress', exist_ok=True)
    os.makedirs('data/noise', exist_ok=True)
    
    # Normally you would split into train/val. For simplicity we assume all in 'data'
    train_dataset = AudioDataset('data', is_train=True)
    
    if len(train_dataset) == 0:
        print("No audio files found in 'data/distress' or 'data/noise'. Please add .wav files.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = AudioCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total * 100
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
    # Save the trained model
    torch.save(model.state_dict(), 'audio_classifier.pth')
    print("Model saved to 'audio_classifier.pth'")

if __name__ == '__main__':
    train()
