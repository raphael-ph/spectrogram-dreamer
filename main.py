from src.preprocessing.generate_spectrogram import AudioFile

# Testing
audio_path = "/Users/rcss/Documents/master/spectrogram-dreamer/data/new-clip/common_voice_en_157.mp3"
audio_file = AudioFile(audio_path)

# show full spectrogram image (existing behaviour)
audio_file.view_spectrogram("test_spectrogram", save_path=f"./images/{audio_file.name}")

# New segmentation test: print number of segments and a few shapes
try:
    segments = audio_file.segment_spectogram()
    print(f"Segment frames L={audio_file.L}, step={audio_file.step}, overlap={audio_file.overlap}")
    print(f"Number of segments: {len(segments)}")
    for i, seg in enumerate(segments[:5]):
        # each segment is a tensor of shape (n_mels, L)
        print(f" segment {i}: shape={tuple(seg.shape)}")
except Exception as e:
    print("Segmentation test failed:", e)