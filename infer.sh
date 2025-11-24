python3 infer.py \
  --model checkpoints/dreamer_20251123_040134/best_model.pt \
  --input data/1_validated-audio/common_voice_en_10047.mp3 \
  --actions data/3_style-vectors/common_voice_en_10047 \
  --mode recon \
  --noise_gate 0.02