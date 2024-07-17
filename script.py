import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import logging
import multiprocessing
from functools import partial
import soundfile as sf
import librosa

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


def process_audio(row, base_path, max_length, mfcc_transform):
    try:
        audio_path = os.path.join(base_path, row["path"])

        if not os.path.exists(audio_path):
            logging.error(f"Audio file not found: {audio_path}")
            return None

        audio, sr = librosa.load(audio_path, sr=16000, duration=max_length)

        if len(audio) > 16000 * max_length:
            audio = audio[: 16000 * max_length]
        else:
            audio = np.pad(audio, (0, 16000 * max_length - len(audio)))

        audio = torch.from_numpy(audio).float().unsqueeze(0)

        mfcc = mfcc_transform(audio)

        return mfcc.squeeze(0).transpose(0, 1).numpy()

    except Exception as e:
        logging.error(f"Error processing audio file {audio_path}: {str(e)}")
        return None


class AudioDataset(Dataset):
    def __init__(
        self,
        csv_path,
        base_path,
        max_length=10,
        use_subset=True,
        subset_fraction=1,
        cache_dir=None,
    ):
        self.csv_path = csv_path
        self.base_path = base_path
        self.cache_dir = cache_dir
        self.df = pd.read_csv(csv_path)

        if use_subset:
            self.df = self.df.sample(frac=subset_fraction, random_state=42)

        self.max_length = max_length
        self.label_encoder = LabelEncoder()

        self.df = self.df[self.df["labels"].str.startswith("female")]
        self.df["emotion"] = self.df["labels"].apply(lambda x: x.split("_")[1])
        self.df["encoded_emotion"] = self.label_encoder.fit_transform(
            self.df["emotion"]
        )

        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=40,
            melkwargs={"n_mels": 80},
        )

        logging.info("Starting preprocessing of dataset")
        self.preprocessed_data = self.preprocess_dataset()

        self.df = self.df.iloc[: len(self.preprocessed_data)]

        logging.info(
            f"Final number of valid samples in dataset: {len(self.preprocessed_data)}"
        )

    def preprocess_dataset(self):
        if self.cache_dir and os.path.exists(
            os.path.join(self.cache_dir, "preprocessed_data.npy")
        ):
            return np.load(
                os.path.join(self.cache_dir, "preprocessed_data.npy"), allow_pickle=True
            )

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            preprocessed_data = list(
                tqdm(
                    pool.imap(
                        partial(
                            process_audio,
                            base_path=self.base_path,
                            max_length=self.max_length,
                            mfcc_transform=self.mfcc_transform,
                        ),
                        [row for _, row in self.df.iterrows()],
                    ),
                    total=len(self.df),
                    desc="Preprocessing audio files",
                )
            )

        preprocessed_data = [data for data in preprocessed_data if data is not None]

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(
                os.path.join(self.cache_dir, "preprocessed_data.npy"), preprocessed_data
            )

        return preprocessed_data

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        mfcc = torch.from_numpy(self.preprocessed_data[idx]).float()
        label = torch.tensor(self.df.iloc[idx]["encoded_emotion"], dtype=torch.long)
        return mfcc, label


class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out


def train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs
):
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            labels = labels.long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_bar.set_postfix(
                {
                    "loss": f"{train_loss/train_total:.4f}",
                    "acc": f"{train_correct/train_total:.4f}",
                }
            )

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_bar.set_postfix(
                    {
                        "loss": f"{val_loss/val_total:.4f}",
                        "acc": f"{val_correct/val_total:.4f}",
                    }
                )

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_female_emotion_model.pth")
            print("Saved best model")

        print()


def predict_emotion(audio_path, model, label_encoder, max_length=10):
    model.eval()
    try:
        logging.info(f"Predicting emotion for file: {audio_path}")
        audio, sr = torchaudio.load(audio_path)

        if sr != 16000:
            audio = torchaudio.transforms.Resample(sr, 16000)(audio)

        if audio.shape[1] > 16000 * max_length:
            audio = audio[:, : 16000 * max_length]
        else:
            audio = torch.nn.functional.pad(
                audio, (0, 16000 * max_length - audio.shape[1])
            )

        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000, n_mfcc=40, melkwargs={"n_mels": 80}
        )
        mfcc = mfcc_transform(audio)
        mfcc = mfcc.squeeze(0).transpose(0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(mfcc)
            _, predicted = torch.max(output, 1)
            predicted_label = label_encoder.inverse_transform(predicted.cpu().numpy())

        confidence = torch.softmax(output, dim=1).squeeze().cpu().numpy()

        logging.info(f"Prediction completed. Predicted emotion: {predicted_label[0]}")
        return predicted_label[0], confidence
    except Exception as e:
        logging.error(f"Error predicting emotion: {str(e)}")
        return None, None


def main():
    try:
        print("Starting main function")

        BASE_PATH = ""
        CACHE_DIR = "cache"

        print(f"BASE_PATH: {BASE_PATH}")
        print(f"CACHE_DIR: {CACHE_DIR}")

        csv_path = os.path.join(BASE_PATH, "Data_path.csv")
        print(f"CSV path: {csv_path}")
        print("Creating AudioDataset")
        dataset = AudioDataset(
            csv_path,
            BASE_PATH,
            max_length=10,
            use_subset=True,
            subset_fraction=1,
            cache_dir=CACHE_DIR,
        )

        if len(dataset) == 0:
            print("Dataset is empty. Cannot proceed with training.")
            return

        print(f"Total samples in dataset: {len(dataset)}")

        print("Splitting data into train and validation sets")
        train_indices, val_indices = train_test_split(
            range(len(dataset)),
            test_size=0.2,
            random_state=42,
            stratify=dataset.df["encoded_emotion"],
        )

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        print("Creating DataLoaders")
        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=32, shuffle=False, num_workers=4
        )

        print("Creating model")
        input_dim = 40
        hidden_dim = 128
        num_classes = len(dataset.label_encoder.classes_)
        model = EmotionClassifier(input_dim, hidden_dim, num_classes).to(device)

        print("Defining loss function and optimizer")
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=5, min_lr=1e-6
        )

        print("Starting model training")
        train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            num_epochs=50,
        )

        best_model_path = "best_female_emotion_model.pth"
        print(f"Loading best model from: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        test_folder = os.path.join(BASE_PATH, "ALL")
        print(f"Test folder: {test_folder}")
        if not os.path.exists(test_folder):
            print(f"Test folder not found: {test_folder}")
            return

        print(f"Testing model on audio files in {test_folder}")
        for audio_file in os.listdir(test_folder)[:10]:
            audio_path = os.path.join(test_folder, audio_file)
            predicted_emotion, confidence = predict_emotion(
                audio_path, model, dataset.label_encoder
            )
            if predicted_emotion is not None:
                print(
                    f"File: {audio_file}, Predicted emotion: {predicted_emotion}, Confidence: {confidence.max():.4f}"
                )
            else:
                print(f"Failed to predict emotion for {audio_file}")

    except Exception as e:
        print(f"An error occurred in main: {str(e)}")
        raise


if __name__ == "__main__":
    print("Script started")
    main()
    print("Script finished")
