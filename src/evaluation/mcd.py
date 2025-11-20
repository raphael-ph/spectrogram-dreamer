# --- Mel Cepstral Distance (MCD) ---
# 
# The MCD is inspired on the classic Cdpstral Distance (CD) metric and
# is essentially an implementation of CD on the Mel frequency scale.
# This aims to better measure the quality of the generated audio based on the spectrogram.
#
# To our implementation, calculating MCD is a measure of how good the generated spectrogram 
# from the world model is and, therefore, how good is the World Model itself (autoencoder).
# 
# Original paper: https://ieeexplore.ieee.org/document/407206
# For this implementation, we'll be using this library: 

# general
import os
from typing import Literal, Optional
from pathlib import Path

# mel cepstral distance imports
from mel_cepstral_distance import compare_audio_files

# internal imports
from base import BaseEvaluator
from ..utils.logger import get_logger

# set up logging
_logger = get_logger("fad_evaluator", level="DEBUG")


class MCD(BaseEvaluator):
    def __init__(self, 
                 path_to_ground_audio: Path | str,
                 path_to_generated_audio: Path | str):
        """
        MCD measures the distance between the Mel-frequency cepstral coefficients (MFCCs)
        of the generated audio and the reference audio. It effectively quantifies how 
        different the "timbre" or spectral envelope of the generated speech is from 
        the ground truth.

        Parameters:
            path_to_ground_audio (Path | str): Path to the ground truth audio
            path_to_generated_audio (Path | str): Path to the generated audio
        """
        # loading audios
        if os.path.exists(path_to_ground_audio):
            self.path_to_ground_audio = path_to_ground_audio
        else:
            _logger.error(f"Path '{path_to_ground_audio}' not found")
            raise FileNotFoundError(f"Path '{path_to_ground_audio}' not found")

        if os.path.exists(path_to_generated_audio):
            self.path_to_generated_audio = path_to_generated_audio
        else:
            _logger.error(f"Path '{path_to_generated_audio}' not found")
            raise FileNotFoundError(f"Path '{path_to_generated_audio}' not found")

    def evaluate(self) -> tuple:
        """
        Compute MCD between two inputs: audio files, amplitude spectrograms, Mel spectrograms, or MFCCs.
        Calculate an alignment penalty (PEN) as an additional metric to indicate the extent of alignment applied.

        Returns:
            Tuple: mcd and penalty
        """
        
        mcd, penalty = compare_audio_files(
            self.path_to_ground_audio,
            self.path_to_ground_audio
        )

        _logger.info(f"Calculated MCD is: {mcd:.2f}. Calculated penalty (PEN): {penalty:.4f}")

        return mcd, penalty
