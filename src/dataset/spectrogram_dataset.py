# general
import os
from typing import List, Tuple
from pathlib import Path

# torch imports
import torch
from torch.utils.data import Dataset

# internal imports
from ..utils.logger import get_logger

# set up logging
_logger = get_logger("dataset", level="DEBUG")

class SpectrogramDataset(Dataset):
    """Generates the spectogram dataset
    
    The dataset is composed by the generated spectograms broken down into 
    sequential time patches. Each spectrogram (audio file) corresponds to one episode. 
    The actions are given by the style vectors.

    This mimics the process of sampling sequences from episodes from the orginal paper.
    """
    def __init__(self, spectrograms_dir_path: str, style_vectors_dir_path: str):
        super().__init__()
        self.DEFAULT_EXT = (".pt") # Allowed extensions
        self.spec_files = self._find_files(Path(spectrograms_dir_path))
        self.style_files = self._find_files(Path(style_vectors_dir_path))

        # loading files
        self.samples = self._pair_files(self.spec_files, self.style_files)

    # --- Dataset methods ---
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        spec_path, style_path = self.samples[index]
        
        spec_tensor = self._load_torch(spec_path)
        style_tensor = self._load_torch(style_path)

        _logger.debug(f"Spec Tensor shape: {spec_tensor.shape}, Spec Tensor Path: {spec_path}")
        _logger.debug(f"Style Tensor shape: {style_tensor.shape}, Style Tensor Path: {style_path}")

        sample = {"spectrogram": spec_tensor, "style": style_tensor}

        return sample

    # --- Helper functions ---
    def _load_torch(self, path: Path):
        return torch.load(path, map_location="cpu")

    def _find_files(self, path: Path) -> List[Path]:
        """Finds the extracted tensor files

        Args:
            path (Path): Directory path of the vector files

        Returns:
            List[Paths]: all the found paths inside the dir        
        """
        if not path.exists():
            _logger.error(f"Path '{path}' was not found.")
            raise FileNotFoundError(f"Path '{path}' was not found.")
        if path.is_file():
            _logger.error(f"Path is '{path}' is a file. Expected folder path.")
            raise Exception(f"Path is '{path}' is a file. Expected folder path.")
        files: List[Path] = []
        for root, _, filenames in os.walk(path):
            for fn in filenames:
                if fn.lower().endswith(self.DEFAULT_EXT):
                    files.append(Path(root) / fn)
        
        _logger.info(f"Found {len(files)} at '{path}'")
        return files
    
    def _pair_files(self, spec_files: List[Path], style_files: List[Path]) -> List[Tuple[Path, Path]]:
        """Pair the found files by file name
        
        Args:
            spec_files (List[Path]): List of spectrogram files
            style_files (List[Path]): List of style vector files
        
        Returns:
            List[Tuple[Path, Path]]: matched spectrogram with style vector
        """
        style_map = {y.stem: y for y in style_files}
        samples: List[Tuple[Path, Path]] = []

        _logger.info("Initiating files pairing...")

        for i, spec in enumerate(spec_files):
            style = style_map.get(spec.stem)
            samples.append((spec, style))

        if any(style is None for _, style in samples):
            _logger.warning(f"Not all spectrograms have matching style vectors.")

        _logger.info(f"Total pairs found: {len(samples)}")    
        return samples




    

