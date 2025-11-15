"""This file is used to run the complete pipeline with a single command"""

from pipeline import Pipeline

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
    p.run_dataset_cleaner(metadata_file='data/data-file/validated.tsv',
                          clips_dir='data/new-clip/',
                          min_votes=2)

    p.process()