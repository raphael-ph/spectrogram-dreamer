import torch
import argparse
import soundfile as sf
from pathlib import Path

# Adjust import path as needed for your project structure
from src.inference.inference import SpectrogramDreamerInference
from src.utils.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the trained .pt model file")
    parser.add_argument("--input", required=True, help="Path to input audio file (seed or target)")
    parser.add_argument("--actions", help="Folder with .pt actions (optional for recon)")
    parser.add_argument("--mode", default="recon", choices=["recon", "dream", "loopback"])
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to dream")
    parser.add_argument("--use_log", action="store_true", default=True, help="Must be True if model was trained with Log-Mel") 
    
    # Temperature: 1.0 is mathematically correct for Log inversion. 
    # < 1.0 (e.g. 0.8) reduces the dynamic range (quieter background, less contrast).
    parser.add_argument("--temp", type=float, default=1.0, help="Inversion temperature. Default 1.0.")
    
    args = parser.parse_args()

    out_dir = Path("output_stabilized")
    out_dir.mkdir(exist_ok=True)
    
    # Initialize Engine
    engine = SpectrogramDreamerInference(args.model, use_log=args.use_log)
    
    print(f"\n⚙️  Processing Mode: {args.mode.upper()}")
    print(f"⚙️  Sample Rate: {engine.processor.sr} Hz")
    
    if args.mode == "loopback":
        # Test the audio processor without the model (sanity check)
        raw = engine.processor.audio_to_segments(args.input).to(engine.device)
        audio = engine.processor.segments_to_audio(raw, temperature=args.temp)
        fname = "loopback.wav"
        
    elif args.mode == "recon":
        # Reconstruct audio using the model's autoencoder
        audio = engine.reconstruct(args.input, args.actions, out_dir, temperature=args.temp)
        fname = "recon.wav"
        
    elif args.mode == "dream":
        # Generate new audio from the latent space
        audio = engine.dream(args.input, args.steps, out_dir, temperature=args.temp)
        fname = "dream.wav"
        
    # Write output
    # CRITICAL FIX: Use the processor's sample rate (likely 48000), not hardcoded 16000
    out_path = out_dir / fname
    sf.write(str(out_path), audio.squeeze().cpu().numpy(), engine.processor.sr, subtype='PCM_16')
    
    print(f"\n✨ Done. Saved to: {out_path}")