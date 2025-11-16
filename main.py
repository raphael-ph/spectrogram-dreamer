from src.preprocessing.launch import launch
from src.dataset.spectrogram_dataset import SpectrogramDataset

# run preprocessing pipeline
# launch()

# testing Dataset
spec_path = "/Users/rcss/Documents/master/spectrogram-dreamer/data/2_mel-spectrograms"
style_path = "/Users/rcss/Documents/master/spectrogram-dreamer/data/3_style-vectors"
ds = SpectrogramDataset(spec_path, style_path)

print(ds[0])
print(ds[20])