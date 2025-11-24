import torch
import argparse
import soundfile as sf
from pathlib import Path

from src.inference.inference import SpectrogramDreamerInference
from src.utils.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spectrogram Dreamer Inference - ALWAYS uses Log-Mel for quality"
    )
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--input", required=True, help="Input audio file path")
    parser.add_argument("--actions", help="Folder with .pt action vectors (optional)")
    parser.add_argument("--mode", default="recon", choices=["recon", "dream", "loopback"],
                       help="Inference mode: recon=reconstruct, dream=imagine, loopback=test")
    parser.add_argument("--steps", type=int, default=100, help="Dream steps (for dream mode)")
    parser.add_argument("--use_log", action="store_true", default=True,
                       help="Use Log-Mel spectrograms (DEFAULT: True, REQUIRED for quality)")
    parser.add_argument("--noise_gate", type=float, default=0.05, 
                       help="Silence threshold for noise reduction")
    
    args = parser.parse_args()

    out_dir = Path("output_final_global")
    out_dir.mkdir(exist_ok=True)
    
    # CRITICAL: Always use Log-Mel mode for inference
    # Training data is in Log-Mel space, so inference must match
    if not args.use_log:
        logger.warning("=" * 80)
        logger.warning("WARNING: You disabled Log-Mel mode!")
        logger.warning("Training uses Log-Mel spectrograms. Power mode will produce:")
        logger.warning("  - Metallic noise artifacts")
        logger.warning("  - Loss of speech intelligibility")  
        logger.warning("  - Hissing background noise")
        logger.warning("=" * 80)
    
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