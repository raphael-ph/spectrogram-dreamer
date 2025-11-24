import torch
import argparse
import soundfile as sf
from pathlib import Path

from src.inference.inference import SpectrogramDreamerInference
from src.utils.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--actions", help="Folder with .pt actions")
    parser.add_argument("--mode", default="recon", choices=["recon", "dream", "loopback"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--use_log", action="store_true", help="Use Log-Mel instead of Power")
    parser.add_argument("--noise_gate", type=float, default=0.05, help="Silence threshold")
    
    args = parser.parse_args()

    out_dir = Path("output_final_global")
    out_dir.mkdir(exist_ok=True)
    
    engine = SpectrogramDreamerInference(args.model, use_log=args.use_log)
    
    if args.mode == "loopback":
        raw = engine.processor.audio_to_segments(args.input).to(engine.device)
        audio = engine.processor.segments_to_audio(raw, noise_gate=args.noise_gate)
        fname = "loopback.wav"
    elif args.mode == "recon":
        audio = engine.reconstruct(args.input, args.actions, out_dir, noise_gate=args.noise_gate)
        fname = "recon.wav"
    elif args.mode == "dream":
        audio = engine.dream(args.input, args.steps, out_dir, noise_gate=args.noise_gate)
        fname = "dream.wav"
        
    sf.write(str(out_dir / fname), audio.squeeze().cpu().numpy(), 16000, subtype='PCM_16')
    print(f"\n✨ Done. Check {out_dir}")