import h5py
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset
from ..utils.logger import get_logger

_logger = get_logger("spectrogram_hdf5_dataset", level="INFO")


class SpectrogramH5Dataset(Dataset):
    """Reads consolidated HDF5 dataset created by create_consolidated_dataset.py

    Exposes the same interface as `SpectrogramDataset` (returns sequence dicts with
    'observation', 'action', 'rewards'). It indexes episodes by `file_id` and `seg_idx`
    stored as datasets in the HDF5 file.
    """

    def __init__(self, h5_path: str):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 dataset not found: {h5_path}")

        # Open the file in read-only mode; actual file handle is opened lazily in worker
        self.h5 = h5py.File(str(self.h5_path), 'r')

        # datasets
        self.specs = self.h5['spectrograms']  # (N, n_mels, T)
        self.styles = self.h5['styles']       # (N, style_dim)
        self.file_ids = self.h5['file_ids']   # (N,)
        self.seg_idx = self.h5['seg_idx']     # (N,)

        # group by episode name (file_id)
        self.episode_map = self._group_by_file_id()

        # sequence settings (match SpectrogramDataset)
        self.SEQUENCE_LENGTH = 50
        self.OVERLAP = 25
        self.OVERLAP_STEP = self.SEQUENCE_LENGTH - self.OVERLAP

        # precompute sequence indices
        self.sequence_indices = self._generate_seq_idx()

    def __len__(self):
        return len(self.sequence_indices)

    def __getitem__(self, idx):
        episode_name, start_idx = self.sequence_indices[idx]
        segment_indices = self.episode_map[episode_name]

        # gather a contiguous block of segment entries
        selected = segment_indices[start_idx: start_idx + self.SEQUENCE_LENGTH]

        specs = []
        styles = []
        for i in selected:
            specs.append(torch.from_numpy(self.specs[i]))
            styles.append(torch.from_numpy(self.styles[i]))

        spec_sequence = torch.stack(specs).unsqueeze(1)  # (T, C=1, n_mels, time)
        style_sequence = torch.stack(styles)

        rewards = torch.zeros(self.SEQUENCE_LENGTH, dtype=torch.float32)

        return {
            'observation': spec_sequence,
            'action': style_sequence,
            'rewards': rewards
        }

    def _group_by_file_id(self) -> Dict[str, List[int]]:
        mapping: Dict[str, List[int]] = {}
        for i in range(self.specs.shape[0]):
            raw = self.file_ids[i]
            # h5py may return bytes or str depending on dtype; normalize to str
            if isinstance(raw, (bytes, bytearray)):
                file_id = raw.decode('utf-8')
            else:
                file_id = str(raw)

            if file_id not in mapping:
                mapping[file_id] = []
            mapping[file_id].append(i)

        # sort by seg_idx within each file
        for fid in mapping:
            mapping[fid].sort(key=lambda idx: int(self.seg_idx[idx]))

        _logger.info(f"Found {len(mapping)} episodes in {self.h5_path}")
        return mapping

    def _generate_seq_idx(self) -> List[Tuple[str, int]]:
        indices = []
        for episode_name, indices_list in self.episode_map.items():
            total_frames = len(indices_list)
            for start_idx in range(0, total_frames - self.SEQUENCE_LENGTH + 1, self.OVERLAP_STEP):
                indices.append((episode_name, start_idx))
        _logger.info(f"Generated {len(indices)} sequence indices from HDF5 dataset")
        return indices

    def close(self):
        try:
            self.h5.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
