import os
import argparse
import pandas as pd
import jiwer
import time
from pathlib import Path
from tqdm import tqdm
from groq import Groq

# Internal imports
from src.utils.logger import get_logger

_logger = get_logger("wer_evaluator")

class WEREvaluator:
    def __init__(self, tsv_path: str, generated_dir: str):
        self.client = Groq()
        self.tsv_path = Path(tsv_path)
        self.generated_dir = Path(generated_dir)
        
        # Transformation pipeline for normalization
        # We lowercase and remove punctuation to compare "content" only
        self.transform = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ])

    def load_ground_truth(self):
        """
        Loads the TSV and creates a map: {file_stem: sentence}
        Matches 'common_voice_en_123.mp3' -> 'common_voice_en_123'
        """
        _logger.info(f"Loading metadata from {self.tsv_path}...")
        try:
            # 1. Load TSV
            # quoting=3 (QUOTE_NONE) helps prevent issues with quotes inside sentences
            df = pd.read_csv(self.tsv_path, sep="\t", on_bad_lines='skip', quoting=3)
            
            # 2. Clean Columns (Critical Fix for KeyError)
            # This removes hidden spaces like " path" or "path "
            df.columns = df.columns.str.strip()
            
            # 3. Validate Schema
            if 'path' not in df.columns or 'sentence' not in df.columns:
                _logger.error(f"Columns found: {df.columns.tolist()}")
                raise KeyError("TSV must contain 'path' and 'sentence' columns")

            # 4. Build Map
            ground_truth = {}
            for _, row in df.iterrows():
                # Extract filename from path (e.g., "clips/common_voice_en_1.mp3")
                full_path = str(row['path'])
                sentence = str(row['sentence'])
                
                # We use the stem (no extension, no folder) as the unique key
                # "common_voice_en_1.mp3" -> "common_voice_en_1"
                stem = Path(full_path).stem
                
                ground_truth[stem] = sentence
                
            _logger.info(f"Loaded {len(ground_truth)} ground truth entries.")
            return ground_truth

        except Exception as e:
            _logger.error(f"Failed to load TSV: {e}")
            raise

    def get_transcription(self, audio_path: str) -> str:
        """
        Transcribes audio using Groq Whisper API.
        """
        try:
            with open(audio_path, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(os.path.basename(audio_path), file.read()),
                    model="whisper-large-v3",
                    temperature=0.0, # Deterministic result
                    response_format="json",
                )
            return transcription.text
        except Exception as e:
            _logger.error(f"Groq API Error on {audio_path}: {e}")
            return ""

    def run(self, output_csv: str = "wer_results.csv"):
        # 1. Load Reference Data
        ground_truth_map = self.load_ground_truth()
        
        # 2. Find Generated Files
        generated_files = sorted(list(self.generated_dir.glob("*.wav")))
        if not generated_files:
            _logger.error(f"No .wav files found in {self.generated_dir}")
            return

        _logger.info(f"Found {len(generated_files)} generated files. Calculating WER...")
        
        results = []
        
        for audio_file in tqdm(generated_files):
            # Match strictly by stem
            # Generated: "/path/to/generated/common_voice_en_1.wav" -> "common_voice_en_1"
            file_stem = audio_file.stem
            
            # Check if this file exists in our metadata
            if file_stem not in ground_truth_map:
                # Optional: Log if you have generated files missing metadata
                # _logger.warning(f"Metadata not found for {file_stem}")
                continue
                
            ref_sentence = ground_truth_map[file_stem]
            
            # Transcribe
            hyp_sentence = self.get_transcription(str(audio_file))
            
            # Normalize
            ref_norm = self.transform(ref_sentence)
            hyp_norm = self.transform(hyp_sentence)
            
            # Calculate WER
            # Handle empty edge cases
            if not ref_norm and not hyp_norm:
                wer = 0.0
            elif not ref_norm:
                wer = 1.0 # Should not happen with valid data
            elif not hyp_norm:
                wer = 1.0 # Transcription failed or silence
            else:
                wer = jiwer.wer(ref_norm, hyp_norm)
            
            results.append({
                "filename": file_stem,
                "reference": ref_norm,
                "hypothesis": hyp_norm,
                "raw_ref": ref_sentence,
                "raw_hyp": hyp_sentence,
                "wer": wer
            })
            
            # Slight sleep to respect API rate limits if necessary
            # time.sleep(0.1)

        if not results:
            _logger.error("No valid pairs matched between Generated Files and TSV.")
            return

        # 3. Export Results
        df_res = pd.DataFrame(results)
        
        avg_wer = df_res["wer"].mean()
        
        print("\n" + "="*40)
        print(f"WER EVALUATION SUMMARY")
        print("="*40)
        print(f"Files Matched:  {len(df_res)}")
        print(f"Average WER:    {avg_wer:.4f} ({avg_wer*100:.2f}%)")
        print("="*40)
        
        df_res.to_csv(output_csv, index=False)
        _logger.info(f"Saved results to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", required=True, help="Path to validated.tsv")
    parser.add_argument("--generated_dir", required=True, help="Path to generated .wav folder")
    parser.add_argument("--output", default="wer_results.csv", help="Output CSV path")
    
    args = parser.parse_args()
    
    if "GROQ_API_KEY" not in os.environ:
        _logger.error("Please set GROQ_API_KEY environment variable.")
        exit(1)

    evaluator = WEREvaluator(args.tsv, args.generated_dir)
    evaluator.run(args.output)