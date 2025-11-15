"""Applies the complete preprocessing pipeline to generate our dataset for training the model.

This pipeline uses `DatasetCleaner` to prepare validated audio (if desired) and
`generate_spectrogram.AudioFile` to compute mel-spectrograms, segment them,
save each segment as `.npy` and compute global normalization statistics.
"""
# standard
import os
from pathlib import Path
from typing import Optional

# third-party
import torch
import numpy as np
from tqdm import tqdm

# internal
from .dataset_cleaner import DatasetCleaner
from .generate_spectrogram import AudioFile
from ..utils.logger import get_logger

_logger = get_logger("pipeline")


class Pipeline:
    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 file_extension: str = "mp3",
                 # spectrogram params (defaults align with existing notebook)
                 n_fft: int = 512,
                 win_length: Optional[int] = 20,
                 hop_length: Optional[int] = 10,
                 n_mels: int = 64,
                 f_min: int = 50,
                 f_max: int = 7600,
                 segment_duration: float = 0.1,
                 overlap: float = 0.5,
                 ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.file_extension = file_extension
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.segment_duration = segment_duration
        self.overlap = overlap

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_dataset_cleaner(self, metadata_file: str, clips_dir: str, min_votes: int = 2):
        """Run `DatasetCleaner` to copy validated audio to `input_dir`.

        Args:
            metadata_file: path to metadata TSV
            clips_dir: directory with original clips
            min_votes: minimum up-votes to consider validated
        """
        cleaner = DatasetCleaner(metadata_file_path=metadata_file,
                                 clips_dir_path=clips_dir,
                                 output_dir=str(self.input_dir),
                                 min_votes=min_votes)
        cleaner.run()

    def process(self):
        """Process all audio files in `input_dir`, compute mel segments and global stats.

        Saves per-audio segments at `output_dir/<basename>/<basename>_NNN.npy` and
        global normalization at `output_dir/mel_stats.npz`.
        """
        files = sorted(self.input_dir.glob(f"*.{self.file_extension}"))
        all_means = []
        all_stds = []

        _logger.info(f"Processing {len(files)} files from {self.input_dir}...")

        for f in tqdm(files):
            try:
                audio = AudioFile(str(f),
                                  n_fft=self.n_fft,
                                  win_length=self.win_length,
                                  hop_length=self.hop_length,
                                  n_mels=self.n_mels,
                                  f_min=self.f_min,
                                  f_max=self.f_max,
                                  segment_duration=self.segment_duration,
                                  overlap=self.overlap)

                mel = audio.mel_spectrogram  # torch tensor

                # Keep tensors: move to CPU and detach
                mel = mel.detach().cpu()

                # mel expected shape: [channels, n_mels, time]
                if mel.dim() == 3 and mel.shape[0] == 1:
                    mel_2d = mel.squeeze(0)  # [n_mels, time]
                elif mel.dim() == 2:
                    mel_2d = mel
                else:
                    # try to reshape conservatively
                    mel_2d = mel.reshape(mel.shape[-2], mel.shape[-1])

                # Skip very short audio
                if mel_2d.numel() == 0 or mel_2d.shape[1] == 0:
                    _logger.warning(f"Empty mel for file: {f.name}")
                    continue

                # per-band mean/std across time axis (tensors)
                mean_per_band = mel_2d.mean(dim=1)
                std_per_band = mel_2d.std(dim=1)

                all_means.append(mean_per_band)
                all_stds.append(std_per_band)

                # Segment using L and step from AudioFile and save each segment as .pt
                L = audio.L
                step = max(1, int(L * (1 - self.overlap))) if hasattr(audio, 'overlap') else audio.step
                segments = []
                for start in range(0, mel_2d.shape[1] - L + 1, step):
                    segments.append(mel_2d[:, start:start + L])
                if len(segments) == 0 and mel_2d.shape[1] > 0:
                    segments.append(mel_2d[:, :min(mel_2d.shape[1], L)])

                base = f.stem
                audio_out_dir = self.output_dir / base
                audio_out_dir.mkdir(parents=True, exist_ok=True)

                for i, seg in enumerate(segments):
                    # seg is a torch tensor already
                    out_path = audio_out_dir / f"{base}_{i:03d}.pt"
                    torch.save(seg, str(out_path))

            except Exception as e:
                _logger.error(f"Error processing {f.name}: {e}")

        if len(all_means) > 0:
            mean_global = torch.stack(all_means).mean(dim=0)
            std_global = torch.stack(all_stds).mean(dim=0)
            stats_path = self.output_dir / "mel_stats.pt"
            torch.save({"mean": mean_global, "std": std_global}, str(stats_path))
            _logger.info(f"Normalization saved to {stats_path}")
        else:
            _logger.warning("No statistics generated (no valid mels).")


if __name__ == "__main__":
    # Example usage (adjust paths as needed)
    p = Pipeline(input_dir="data/1_validated-audio/",
                 output_dir="data/2_mel-spectrograms/",
                 file_extension="mp3",
                 n_fft=512,
                 win_length=20,
                 hop_length=10,
                 n_mels=64,
                 f_min=50,
                 f_max=7600,
                 segment_duration=0.1,
                 overlap=0.5)

    # If you need to run the dataset cleaner first, uncomment and set paths:
    # p.run_dataset_cleaner(metadata_file='data/common-voice-dataset-version-4/data-file/validated.tsv',
    #                       clips_dir='data/common-voice-dataset-version-4/new-clip/',
    #                       min_votes=2)

    p.process()

    