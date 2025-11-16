# Preprocessing Module

The preprocessing module is responsible for preparing audio data for the spectrogram dreamer model. It handles dataset validation, audio processing, and mel-spectrogram generation with segmentation for training.

## Overview

This module implements a complete preprocessing pipeline that:

1. **Validates audio datasets** using crowdsourced voting metadata
2. **Generates mel-spectrograms** from audio files using torchaudio
3. **Segments spectrograms** into fixed-length windows for model training
4. **Computes global normalization statistics** for consistent data preprocessing

## Module Components

### 1. `dataset_cleaner.py` - DatasetCleaner

Filters and validates audio files based on crowdsourced voting metadata.

#### Class: `DatasetCleaner`

**Purpose:** Filters audio files from a metadata file based on voting criteria and copies validated files to an output directory.

**Constructor Parameters:**
- `metadata_file_path` (str): Path to the TSV metadata file containing voting information
- `clips_dir_path` (str): Directory containing the original audio clips
- `output_dir` (str): Directory where validated audio files will be copied
- `min_votes` (int, default=2): Minimum number of up-votes required to consider an audio clip as validated

**Key Methods:**

- `run()`: Executes the complete validation pipeline
  - Loads metadata from TSV file
  - Filters clips based on voting criteria (more up-votes than down-votes, meets minimum votes)
  - Copies validated audio files to output directory
  - Saves filtered metadata as `validated_metadata.tsv`

- `_load_dataframe()`: Loads and normalizes the metadata TSV file
- `_filter_data(df)`: Applies filtering logic to identify validated clips
- `_copy_validated_files(validated_clips)`: Copies validated audio files and saves metadata

**Example Usage:**
```python
from dataset_cleaner import DatasetCleaner

cleaner = DatasetCleaner(
    metadata_file_path='data/data-file/validated.tsv',
    clips_dir_path='data/new-clip/',
    output_dir='data/1_validated-audio/',
    min_votes=2
)
cleaner.run()
```

---

### 2. `generate_spectrogram.py` - AudioFile

Processes audio files and generates mel-spectrograms with segmentation capabilities.

#### Class: `AudioFile`

**Purpose:** Loads audio files, computes mel-spectrograms, and provides methods for visualization and segmentation.

**Constructor Parameters:**

Audio Parameters:
- `waveform_path` (str): Path to audio file (.mp3 or .wav)

Spectrogram Parameters (with defaults aligned to Henrique's notebook):
- `n_fft` (int, default=1024): FFT size for STFT computation
- `win_length` (int or None, default=20): Window length in milliseconds (automatically converted to samples)
- `hop_length` (int or None, default=10): Hop length in milliseconds (automatically converted to samples)
- `n_mels` (int, default=64): Number of mel frequency bins
- `f_min` (int, default=50): Minimum frequency in Hz
- `f_max` (int, default=7600): Maximum frequency in Hz

Segmentation Parameters:
- `segment_duration` (float, default=0.1): Duration of each segment in seconds
- `overlap` (float, default=0.5): Overlap ratio between consecutive segments (0.0-1.0)

**Key Properties & Methods:**

- `mel_spectrogram` (property): Returns the computed mel-spectrogram as a PyTorch tensor
  - Shape: `[channels, n_mels, time_frames]`
  - Uses Slaney normalization and HTK mel scale

- `segment_spectrogram()`: Segments the mel-spectrogram based on `segment_duration` and `overlap`.
  - Returns: List of mel-spectrogram segments as tensors

- `view_spectrogram(title, ylabel, ax, save_path)`: Visualizes the mel-spectrogram
  - Converts to dB scale with 80dB dynamic range
  - Normalizes to 0-1 range for consistency
  - Optionally saves as high-resolution PNG image

**Internal Processing:**

- Automatically adjusts `n_fft` to satisfy the STFT constraint: `n_fft >= win_length`
- Converts time-based parameters (win_length, hop_length) from milliseconds to samples
- Supports both mono and multi-channel audio (squeezes to 2D if needed)

**Example Usage:**
```python
from generate_spectrogram import AudioFile

audio = AudioFile(
    'path/to/audio.mp3',
    n_fft=512,
    win_length=20,
    hop_length=10,
    n_mels=64,
    f_min=50,
    f_max=7600,
    segment_duration=0.1,
    overlap=0.5
)

# Get the mel-spectrogram
mel = audio.mel_spectrogram
print(mel.shape)  # torch.Size([1, 64, time_frames])

# Segment it
segments = audio.segment_spectrogram()

# Visualize and save
audio.view_spectrogram('My Audio', save_path='./images/audio')
```

---

### 3. `pipeline.py` - Pipeline

Orchestrates the complete preprocessing workflow combining dataset cleaning and spectrogram generation.

#### Class: `Pipeline`

**Purpose:** Provides a high-level interface to run the entire preprocessing pipeline: validate datasets and generate segmented spectrograms with normalization statistics.

**Constructor Parameters:**

Directory & Format:
- `input_dir` (str): Directory containing audio files to process
- `output_dir` (str): Directory where processed spectrograms will be saved
- `file_extension` (str, default="mp3"): Audio file extension to process

Spectrogram Parameters (same as AudioFile):
- `n_fft`, `win_length`, `hop_length`, `n_mels`, `f_min`, `f_max`
- `segment_duration`, `overlap`

**Key Methods:**

- `run_dataset_cleaner(metadata_file, clips_dir, min_votes)`: Runs the dataset validation step
  - Filters validated audio and copies to `input_dir`
  - Parameters match `DatasetCleaner` constructor

- `process()`: Processes all audio files in `input_dir`
  - For each audio file:
    - Generates mel-spectrogram
    - Extracts overlapping segments
    - Saves each segment as `<audio_name>_NNN.pt` (PyTorch tensor format)
    - Computes per-band mean and std
  - Saves global normalization statistics to `output_dir/mel_stats.pt`
  - Output structure:
    ```
    output_dir/
    ├── mel_stats.pt              # Global normalization stats
    ├── audio1/
    │   ├── audio1_000.pt
    │   ├── audio1_001.pt
    │   └── ...
    └── audio2/
        ├── audio2_000.pt
        └── ...
    ```

**Example Usage:**
```python
from pipeline import Pipeline

# Create pipeline instance
p = Pipeline(
    input_dir='data/1_validated-audio/',
    output_dir='data/2_mel-spectrograms/',
    file_extension='mp3',
    n_fft=512,
    win_length=20,
    hop_length=10,
    n_mels=64,
    f_min=50,
    f_max=7600,
    segment_duration=0.1,
    overlap=0.5
)

# Step 1: Validate and filter dataset (optional)
p.run_dataset_cleaner(
    metadata_file='data/data-file/validated.tsv',
    clips_dir='data/new-clip/',
    min_votes=2
)

# Step 2: Process all audio files
p.process()
```

---

### 4. `launch.py` - Quick Start Script

A ready-to-use script for running the complete pipeline with a single command.

**Configuration:**
The script is pre-configured with sensible defaults aligned with Henrique's notebook. Modify the parameters in the `__main__` block to customize:

```python
p = Pipeline(
    input_dir="data/1_validated-audio/",
    output_dir="data/2_mel-spectrograms/",
    file_extension="mp3",
    n_fft=512,
    win_length=20,
    hop_length=10,
    n_mels=64,
    f_min=50,
    f_max=7600,
    segment_duration=0.1,
    overlap=0.5
)
```

**Running:**
```bash
python src/preprocessing/launch.py
```

---

## Data Flow

```
Raw Audio Files
       ↓
[dataset_cleaner.py]
       ↓
Validated Audio Files
       ↓
[generate_spectrogram.py]
       ↓
Mel-Spectrograms
       ↓
[Pipeline.process()]
       ↓
Segmented Spectrograms + Normalization Stats
```

## Output Files

### Segmented Spectrograms
- **Format:** PyTorch tensor (.pt) files
- **Location:** `output_dir/<audio_name>/<audio_name>_NNN.pt`
- **Shape:** `[n_mels, time_frames]` (2D tensor)
- **NNN:** Zero-padded segment index (000, 001, etc.)

### Normalization Statistics
- **File:** `output_dir/mel_stats.pt`
- **Contents:** Dictionary with keys:
  - `"mean"`: Per-band global mean (shape: `[n_mels]`)
  - `"std"`: Per-band global std (shape: `[n_mels]`)
- **Usage:** For normalizing input data during model training

### Validation Metadata (from DatasetCleaner)
- **File:** `output_dir/validated_metadata.tsv`
- **Contents:** Filtered metadata for validated clips

---

## Key Parameters Explained

### Spectrogram Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_fft` | 512 | FFT size; determines frequency resolution |
| `win_length` | 20 ms | Window size for STFT; larger = better frequency resolution but slower |
| `hop_length` | 10 ms | Frame advancement; smaller = more temporal resolution (100 fps at 10ms) |
| `n_mels` | 64 | Number of mel frequency bins; more bins = more spectral detail |
| `f_min` | 50 Hz | Minimum frequency to capture (filters out very low frequencies) |
| `f_max` | 7600 Hz | Maximum frequency to capture (typical for speech/music) |

### Segmentation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `segment_duration` | 0.1 s | Length of each segment (100ms = ~10 frames at 100fps) |
| `overlap` | 0.5 | Overlap ratio between segments; 0.5 = 50% overlap |

---

## Dependencies

- `torch`: PyTorch for tensor operations
- `torchaudio`: Audio loading and mel-spectrogram computation
- `pandas`: Data manipulation for metadata
- `numpy`: Numerical operations
- `tqdm`: Progress bars
- `matplotlib`: Visualization (for `view_spectrogram()`)

---

## Error Handling

The pipeline includes robust error handling:

- **Invalid file formats:** Only `.mp3` and `.wav` are accepted
- **Missing files:** Gracefully skips files that don't exist or can't be loaded
- **Empty spectrograms:** Skips audio files that produce empty spectrograms
- **Path creation:** Automatically creates output directories if they don't exist
- **Logging:** All operations logged to help debug issues

---

## Logging

The module uses a custom logger accessible via `from utils.logger import get_logger`. Log output includes:

- Dataset statistics (completion %, validation %, file counts)
- Processing progress (file-by-file status with error messages)
- Parameter adjustments (e.g., when `n_fft` is auto-adjusted)
- File I/O operations (save paths, file counts)

---

## Example Complete Workflow

```python
from src.preprocessing.pipeline import Pipeline

# Initialize pipeline with custom parameters
pipeline = Pipeline(
    input_dir='data/1_validated-audio/',
    output_dir='data/2_mel-spectrograms/',
    file_extension='mp3',
    n_fft=512,
    win_length=20,
    hop_length=10,
    n_mels=64,
    f_min=50,
    f_max=7600,
    segment_duration=0.1,
    overlap=0.5
)

# Step 1: Clean dataset (if raw data needs validation)
pipeline.run_dataset_cleaner(
    metadata_file='data/data-file/validated.tsv',
    clips_dir='data/new-clip/',
    min_votes=2
)

# Step 2: Generate spectrograms and segments
pipeline.process()

# Now spectrograms are ready for model training!
```

---

## Notes

- All spectrogram parameters are inspired by and aligned with Henrique's original notebook
- The module uses **Slaney normalization** and **HTK mel scale** for consistency
- Segments are saved as PyTorch tensors for efficient loading during training
- Global normalization statistics should be used to normalize all segments before feeding to the model
- The `overlap` parameter enables training data augmentation through sliding window segmentation
