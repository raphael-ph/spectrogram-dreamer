# Generating MEL spectogram using torchaudio
# torch imports
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torchaudio.utils import download_asset

# viz imports
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from IPython.display import Audio

# general
from pathlib import Path

# internal imports
from ..utils.logger import get_logger

# Logging setup
_logger = get_logger("spectrogram")

class AudioFile:
    def __init__(self, waveform_path: str, 
                 n_fft: int, 
                 win_length: int,
                 hop_length: int,
                 n_mels: int,
                 ):
        permitted_extensions = [".mp3", ".wav"]
        file_path = Path(waveform_path)

        if file_path.suffix not in permitted_extensions:
            _logger.error(f"'{file_path.suffix}' not allowed.")
            raise TypeError(f"'{file_path.suffix}' not allowed.")
        
        self.name = file_path.stem
        self.waveform, self.sample_rate = torchaudio.load(str(file_path))
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def view_spectrogram(self, title=None, ylabel="Mel bins", ax=None, save_path=None):
        """Display and optionally save a Mel spectrogram.

        Args:
            title (str): Plot title.
            ylabel (str): Label for the Y-axis.
            ax (matplotlib.axes.Axes, optional): Axis to draw on.
            save_path (str or Path, optional): If provided, saves the figure as an image.
        """
        mel = self.mel_spectrogram
        mel_db = T.AmplitudeToDB("power", 80.0)(mel)
        mel_db = mel_db.squeeze(0).detach().cpu().numpy()

        # Normalize 0–1 for consistency across samples
        mel_db -= mel_db.min()
        mel_db /= mel_db.max() + 1e-8

        # Plot without axes or whitespace
        plt.figure(figsize=(40, 10), dpi=300)
        plt.axis("off")
        plt.imshow(mel_db, origin="lower", aspect="auto", cmap="magma")

        if save_path is not None:
            path = Path(save_path).with_suffix(".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, bbox_inches="tight", pad_inches=0)
            _logger.info(f"Saved spectrogram: {path}")

    @property
    def mel_spectrogram(self):
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=self.n_mels,
            mel_scale="htk",
        )

        mel_spec = mel_spectrogram(self.waveform)

        return mel_spec

if __name__ == "__main__":
    audio_path = "/Users/rcss/Documents/master/spectrogram-dreamer/data/new-clip/common_voice_en_157.mp3"
    config = {
                "n_fft": 512, 
                "win_length": None,
                "hop_length": 160,
                "n_mels": 32,
    }
    audio_file = AudioFile(audio_path, **config)

    audio_file.view_spectrogram("test_spectrogram", save_path=f"./images/{audio_file.name}")