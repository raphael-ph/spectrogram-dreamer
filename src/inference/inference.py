import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# Fix for plotting on headless servers/remote machines
import matplotlib
matplotlib.use('Agg')

# Try importing the model from your source code

from ..models.dreamer import DreamerModel
from ..utils.logger import get_logger

logger = get_logger(__name__)


# ==========================================
# 0. GLOBAL TRAINING STATISTICS (HARDCODED)
# ==========================================
# Derived from your "Atributos globais"
GLOBAL_MEAN = np.array([
    0.84685516, 0.9143231, 2.714284, 1.5906031, 1.7550404, 2.821859, 1.9366663, 1.7264678,
    1.7281687, 1.8439455, 2.0116918, 2.1360044, 1.6936872, 1.3126792, 1.0042206, 0.81750506,
    0.61177546, 0.4548031, 0.34468323, 0.27048427, 0.24242267, 0.20218888, 0.15644568, 0.16400875,
    0.12357886, 0.12595014, 0.10007602, 0.10497459, 0.10379053, 0.09889296, 0.09445571, 0.08885168,
    0.0764387, 0.05807441, 0.04466241, 0.03723266, 0.03544522, 0.0338206, 0.03495762, 0.03216488,
    0.02812481, 0.02229181, 0.02055656, 0.02065849, 0.02087944, 0.02081459, 0.02066852, 0.01855635,
    0.0174177, 0.01613207, 0.0161212, 0.01663797, 0.01736451, 0.0182647, 0.01779571, 0.01712709,
    0.0156159, 0.0145717, 0.01334624, 0.0119815, 0.0106809, 0.00944819, 0.00815711, 0.00682004
], dtype=np.float32)

GLOBAL_STD = np.array([
    5.5214205, 5.961305, 15.275354, 7.769694, 8.661216, 14.8847, 11.466088, 9.196149,
    9.289526, 10.504267, 12.2525835, 13.698682, 12.2135315, 9.788832, 8.073704, 7.320181,
    6.163135, 5.118304, 4.2523026, 3.6003978, 3.3463945, 2.745384, 2.1611617, 2.254761,
    1.7603103, 1.7119865, 1.3363085, 1.3531126, 1.4040065, 1.3299215, 1.2334577, 1.2426051,
    1.0945596, 0.756417, 0.5526683, 0.4651862, 0.42074946, 0.39075106, 0.38851878, 0.36128923,
    0.33291787, 0.2782033, 0.28322837, 0.28833774, 0.2953352, 0.3122883, 0.31489885, 0.30362788,
    0.28635618, 0.26723883, 0.2821316, 0.30205503, 0.3232866, 0.33916867, 0.32270235, 0.30671856,
    0.28506985, 0.27385435, 0.24366298, 0.21762806, 0.20564279, 0.20396613, 0.20264256, 0.18845226
], dtype=np.float32)


# ==========================================
# 1. AUDIO PROCESSOR (POWER MODE DEFAULT)
# ==========================================
class AudioProcessor:
    def __init__(self, use_log=False, sample_rate=16000, n_fft=1024, win_length=20, hop_length=10, 
                 n_mels=64, f_min=50, f_max=7600, segment_duration=0.1, overlap=0.5):
        
        self.sr = sample_rate
        self.n_fft = n_fft 
        self.n_mels = n_mels
        self.use_log = use_log 
        
        self.hop_len = int(sample_rate * (hop_length / 1000))
        self.win_len = int(sample_rate * (win_length / 1000))
        self.frames_per_seg = int((segment_duration * sample_rate) / self.hop_len)
        self.overlap = overlap

        # 1. MEL TRANSFORM
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=self.win_len,
            hop_length=self.hop_len, n_mels=n_mels, f_min=f_min, f_max=f_max,
            center=True, pad_mode="reflect", power=2.0, normalized=False,
            norm="slaney", mel_scale="htk"
        )
        
        # 2. INVERSE MEL
        self.inv_mel_transform = T.InverseMelScale(
            n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=sample_rate,
            f_min=f_min, f_max=f_max, 
            norm="slaney", mel_scale="htk"
        )
        
        # 3. GRIFFIN LIM
        self.griffin_lim = T.GriffinLim(
            n_fft=n_fft, win_length=self.win_len, hop_length=self.hop_len,
            power=1.0, n_iter=60 
        )

    def audio_to_segments(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sr: waveform = T.Resample(sr, self.sr)(waveform)
        
        # Raw Power Spec
        spec = self.mel_transform(waveform)
        
        if self.use_log:
            spec = torch.log(torch.clamp(spec, min=1e-5))
        
        # Segment
        C, H, Time = spec.shape
        W = self.frames_per_seg
        stride = int(W * (1 - self.overlap))
        segments = [spec[:, :, i : i + W] for i in range(0, Time - W + 1, stride)]
        
        if not segments: return torch.zeros(1, 1, C, H, W)
        return torch.stack(segments).unsqueeze(0)

    def segments_to_audio(self, segments_tensor, noise_gate=0.0):
        if segments_tensor.dim() == 5: segments_tensor = segments_tensor.squeeze(0)
        T_seq, C, H, W = segments_tensor.shape
        stride = int(W * (1 - self.overlap))
        
        total_len = (T_seq - 1) * stride + W
        full_spec = torch.zeros((C, H, total_len), device=segments_tensor.device)
        counts = torch.zeros((C, H, total_len), device=segments_tensor.device)
        
        for t in range(T_seq):
            start = t * stride
            full_spec[:, :, start:start+W] += segments_tensor[t]
            counts[:, :, start:start+W] += 1.0
            
        full_spec = full_spec / torch.clamp(counts, min=1.0)
        
        # --- INVERSION & DENOISING ---
        if self.use_log:
            mel_power = torch.exp(full_spec)
        else:
            mel_power = full_spec
            
            # --- NOISE GATE ---
            if noise_gate > 0:
                peak = torch.max(mel_power)
                threshold = peak * noise_gate
                mel_power[mel_power < threshold] = 0.0

        # Inverse Mel
        linear_power = self.inv_mel_transform(mel_power)
        
        # Sqrt to get Magnitude
        linear_mag = torch.sqrt(torch.clamp(linear_power, min=0.0))
        
        # Griffin Lim
        waveform = self.griffin_lim(linear_mag)
        
        # --- ROBUST VOLUME NORMALIZATION ---
        robust_max = torch.quantile(torch.abs(waveform), 0.999) 
        if robust_max > 0:
            waveform = waveform / (robust_max + 1e-6)
            
        return waveform

# ==========================================
# 2. INFERENCE ENGINE (WITH GLOBAL STATS)
# ==========================================
class SpectrogramDreamerInference:
    def __init__(self, model_path, use_log, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.processor = AudioProcessor(use_log=use_log)
        self.use_log = use_log
        
        # PREPARE GLOBAL STATS FOR BROADCASTING
        # Shape: (1, 1, 1, 64, 1) -> (Batch, Time, Channel, Height, Width)
        self.mean = torch.tensor(GLOBAL_MEAN, device=device).view(1, 1, 1, -1, 1)
        self.std = torch.tensor(GLOBAL_STD, device=device).view(1, 1, 1, -1, 1)
        
        logger.info(f"🔄 Loading model... (Log Mode: {use_log})")
        logger.info(f"📊 Global Stats Loaded. Mean Range: {self.mean.min():.2f}-{self.mean.max():.2f}")
        
        self.model = DreamerModel(
            h_state_size=200, z_state_size=30, action_size=53, 
            embedding_size=256, in_channels=1, cnn_depth=32
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def normalize(self, spec):
        return (spec - self.mean) / (self.std + 1e-6)

    def denormalize(self, spec):
        return (spec * (self.std + 1e-6)) + self.mean

    def save_spectrogram_plot(self, tensor, filename, title="Spectrogram"):
        try:
            if tensor.dim() == 5: tensor = tensor.squeeze(0).squeeze(1)
            if tensor.dim() == 3: full_spec = torch.cat([t for t in tensor], dim=1)
            else: full_spec = tensor
                
            spec_np = full_spec.detach().cpu().numpy()
            
            # Always visualize in dB
            if not self.use_log:
                # Power -> dB
                spec_db = 10 * np.log10(np.maximum(spec_np, 1e-10))
            else:
                spec_db = spec_np
            
            spec_db = spec_db - spec_db.max()
            
            plt.figure(figsize=(10, 4))
            plt.imshow(spec_db, aspect='auto', origin='lower', cmap='magma', interpolation='nearest')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"{title}")
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            logger.info(f"🖼️  Saved plot to {filename}")
        except Exception as e:
            logger.info(f"⚠️ Failed to plot: {e}")

    def load_actions(self, folder, num_needed):
        folder = Path(folder)
        files = sorted(list(folder.glob("*.pt")))
        try: files.sort(key=lambda x: int(''.join(filter(str.isdigit, x.name))))
        except: pass
        
        loaded = []
        for i in range(min(len(files), num_needed)):
            vec = torch.load(files[i], map_location=self.device)
            if vec.dim() == 1: vec = vec.unsqueeze(0)
            loaded.append(vec)
        while len(loaded) < num_needed: loaded.append(loaded[-1])
        return torch.stack(loaded, dim=1)

    def print_stats(self, name, tensor):
        mean = tensor.mean().item()
        std = tensor.std().item()
        mi = tensor.min().item()
        ma = tensor.max().item()
        print(f"📊 {name:15s} | Mean: {mean:8.4f} | Std: {std:8.4f} | Range: [{mi:.4f}, {ma:.4f}]")

    def reconstruct(self, audio_path, action_path, out_dir, noise_gate=0.0):
        logger.info("\n🏗️  RECONSTRUCTION")
        raw = self.processor.audio_to_segments(audio_path).to(self.device)
        self.print_stats("Raw Input", raw)
        self.save_spectrogram_plot(raw, out_dir / "input.png", "Raw Input")
        
        # USE GLOBAL NORMALIZATION
        norm = self.normalize(raw)
        self.print_stats("Normalized In", norm)
        
        B, T = norm.shape[:2]
        if action_path:
            actions = self.load_actions(action_path, T).to(self.device)
        else:
            actions = torch.randn(B, T, 53).to(self.device)

        with torch.no_grad():
            out = self.model(norm, actions, compute_loss=False)
            norm_recon = out['reconstructed']
            
        self.print_stats("Model Output", norm_recon)
        
        # USE GLOBAL DENORMALIZATION
        raw_recon = self.denormalize(norm_recon)
        
        if not self.use_log:
            raw_recon = torch.clamp(raw_recon, min=0.0)
        
        self.print_stats("Final Spec", raw_recon)
        self.save_spectrogram_plot(raw_recon, out_dir / "recon.png", "Reconstruction")
        
        return self.processor.segments_to_audio(raw_recon, noise_gate)

    def dream(self, seed_path, steps, out_dir, noise_gate=0.0):
        logger.info(f"\n🌙 DREAMING ({steps} steps)")
        raw = self.processor.audio_to_segments(seed_path).to(self.device)
        
        # USE GLOBAL NORMALIZATION
        norm = self.normalize(raw)
        
        actions = torch.zeros(1, norm.shape[1], 53).to(self.device)
        with torch.no_grad():
            out = self.model(norm, actions, compute_loss=False)
            h = out['h_states'][:, -1]
            z = out['z_states'][:, -1]
            
        style = torch.randn(1, 53).to(self.device)
        class FixedActor(torch.nn.Module):
            def __init__(self, s): super().__init__(); self.s = s
            def forward(self, h, z): return self.s.expand(h.shape[0], -1)
            
        with torch.no_grad():
            imagined = self.model.rssm.imagine(FixedActor(style), h, z, steps)
            norm_dream = self.model.decoder(imagined['h_states'], imagined['z_states'])
            
        # USE GLOBAL DENORMALIZATION
        raw_dream = self.denormalize(norm_dream)
        
        if not self.use_log:
            raw_dream = torch.clamp(raw_dream, min=0.0)
            
        self.print_stats("Dream Output", raw_dream)
        self.save_spectrogram_plot(raw_dream, out_dir / "dream.png", "Dream")
        
        return self.processor.segments_to_audio(raw_dream, noise_gate)