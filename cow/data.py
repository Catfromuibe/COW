"""Protocol loading aligned with HyperPotter's Default_dataset.

HyperPotter protocols are whitespace-separated. The fields used here are the
same columns as data_utils.Default_dataset:

    column 4: bonafide / spoof
    column 5: audio path
    column 8: split in {train, dev, eval, test, -}

HyperPotter maps spoof->0 and bonafide->1. This experiment follows the paper:
genuine y=0, manipulated y=1. Only that mapping is changed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


GENUINE_TOKENS = {"bonafide", "bona", "genuine", "real", "0"}
SPOOF_TOKENS = {"spoof", "fake", "manipulated", "1"}
SPLITS = {"train", "dev", "eval", "test", "-"}


@dataclass
class Record:
    path: str
    label: int
    split: str
    utt: str


def _label_from_token(token: str) -> int | None:
    key = token.strip().lower()
    if key in GENUINE_TOKENS:
        return 0
    if key in SPOOF_TOKENS:
        return 1
    return None


def parse_protocol_line(line: str, audio_root: str | None = None) -> Record | None:
    """Parse one HyperPotter line, with a fallback for official ASVspoof files."""
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None
    parts = raw.split()

    label = path = split = utt = None
    if len(parts) >= 9 and _label_from_token(parts[4]) is not None:
        label = _label_from_token(parts[4])
        path = parts[5]
        split = parts[8].strip()
        utt = parts[1]
    elif len(parts) >= 5 and _label_from_token(parts[4]) is not None and audio_root:
        # speaker utt - attack label
        label = _label_from_token(parts[4])
        utt = parts[1]
        split = "train"
        root = Path(audio_root)
        candidates = [
            root / f"{utt}.flac",
            root / f"{utt}.wav",
            root / utt,
        ]
        path = str(next((c for c in candidates if c.exists()), candidates[0]))
    elif len(parts) >= 2 and _label_from_token(parts[1]) is not None:
        path = parts[0]
        label = _label_from_token(parts[1])
        split = parts[2] if len(parts) >= 3 else "train"
        utt = Path(path).stem
    else:
        return None

    if split not in SPLITS:
        split = "train"
    return Record(path=path, label=label, split=split, utt=utt or Path(path).stem)


def read_protocol(path: str, splits: list[str], audio_root: str | None = None) -> list[Record]:
    wanted = set(splits)
    records = []
    missing = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            rec = parse_protocol_line(line, audio_root=audio_root)
            if rec is None or rec.split not in wanted:
                continue
            if not Path(rec.path).exists():
                missing += 1
                continue
            records.append(rec)
    if missing:
        print(f"discarded {missing} missing files from {path}")
    if not records:
        raise FileNotFoundError(
            f"No usable recordings in {path} for splits {sorted(wanted)}. "
            "Check the protocol columns and that column 6 is an existing path."
        )
    return records


def load_wave(path: str, sample_rate: int = 16000) -> torch.Tensor:
    """Load mono audio at the target rate. Prefer the same librosa path HyperPotter uses."""
    wave = None
    sr = None
    try:
        import librosa

        audio, sr = librosa.load(path, sr=sample_rate, mono=True, dtype=np.float32)
        wave = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        return wave.contiguous()
    except Exception:
        pass

    try:
        import torchaudio

        audio, sr = torchaudio.load(path)
        wave = audio.mean(dim=0)
    except Exception:
        try:
            import soundfile as sf

            audio, sr = sf.read(path, always_2d=False)
            wave = torch.as_tensor(audio, dtype=torch.float32)
            if wave.ndim > 1:
                wave = wave.mean(dim=-1)
        except Exception:
            wave, sr = _read_wav_stdlib(path)

    if sr != sample_rate:
        try:
            import torchaudio

            wave = torchaudio.functional.resample(wave.unsqueeze(0), sr, sample_rate).squeeze(0)
        except Exception:
            new_len = max(1, int(round(wave.numel() * sample_rate / sr)))
            wave = torch.nn.functional.interpolate(
                wave.view(1, 1, -1), size=new_len, mode="linear", align_corners=False
            ).view(-1)
    return wave.contiguous().float()


def fit_length(wave: torch.Tensor, length: int, random_crop: bool) -> torch.Tensor:
    if wave.numel() < length:
        reps = length // wave.numel() + 1
        wave = wave.repeat(reps)[:length]
    elif wave.numel() > length:
        start = random.randint(0, wave.numel() - length) if random_crop else 0
        wave = wave[start : start + length]
    return wave


def length_preserving_process(wave: torch.Tensor, kind: str) -> torch.Tensor:
    """Label-preserving processors that keep the temporal grid aligned."""
    if kind == "identity":
        return wave
    if kind == "gain":
        gain = 10 ** (random.uniform(-6.0, 6.0) / 20.0)
        return (wave * gain).clamp(-1.0, 1.0)
    if kind == "noise":
        snr_db = random.uniform(15.0, 30.0)
        noise = torch.randn_like(wave)
        sig = wave.pow(2).mean().clamp_min(1e-8)
        noi = noise.pow(2).mean().clamp_min(1e-8)
        scale = torch.sqrt(sig / (noi * (10 ** (snr_db / 10.0))))
        return (wave + noise * scale).clamp(-1.0, 1.0)
    if kind == "lowpass":
        kernel = torch.tensor([0.25, 0.5, 0.25], dtype=wave.dtype)
        padded = torch.nn.functional.pad(wave.view(1, 1, -1), (1, 1), mode="reflect")
        return torch.nn.functional.conv1d(padded, kernel.view(1, 1, 3)).view(-1).clamp(-1.0, 1.0)
    raise KeyError(kind)


class RecordingDataset(Dataset):
    def __init__(
        self,
        records: list[Record],
        sample_rate: int = 16000,
        segment_time: float = 4.0,
        train: bool = False,
        processed_view: bool = False,
        processors: tuple[str, ...] = ("identity", "gain", "noise", "lowpass"),
    ):
        self.records = list(records)
        self.sample_rate = sample_rate
        self.segment_length = int(segment_time * sample_rate)
        self.train = train
        self.processed_view = processed_view
        self.processors = processors

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        wave = load_wave(rec.path, self.sample_rate)
        wave = fit_length(wave, self.segment_length, random_crop=self.train)
        if not self.processed_view:
            return wave, rec.label, rec.path

        kind = random.choice(self.processors) if self.train else "identity"
        paired = length_preserving_process(wave, kind)
        # Length-preserving views have a complete one-to-one neighborhood map.
        matched = torch.tensor(1, dtype=torch.long)
        return wave, paired, rec.label, matched, rec.path


def split_reference(records: list[Record], ref_ratio: float, seed: int) -> tuple[list[Record], list[Record]]:
    """Disjoint genuine reference subset and labeled detector subset."""
    genuine = [r for r in records if r.label == 0]
    spoof = [r for r in records if r.label == 1]
    rng = random.Random(seed)
    rng.shuffle(genuine)
    n_ref = max(1, int(round(len(genuine) * ref_ratio)))
    n_ref = min(n_ref, max(1, len(genuine) - 1)) if len(genuine) > 1 else len(genuine)
    ref = genuine[:n_ref]
    detector = genuine[n_ref:] + spoof
    if not detector:
        raise RuntimeError("Detector split is empty. Lower --ref_ratio or add manipulated recordings.")
    return ref, detector


def balanced_sampler(records: list[Record]) -> WeightedRandomSampler:
    labels = [r.label for r in records]
    counts = {0: max(1, labels.count(0)), 1: max(1, labels.count(1))}
    weights = [1.0 / counts[y] for y in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def collate_train(batch):
    batch = [b for b in batch if b is not None]
    waves, paired, labels, matched, paths = zip(*batch)
    return (
        torch.stack(waves),
        torch.stack(paired),
        torch.tensor(labels, dtype=torch.float32),
        torch.stack([m.float().reshape(()) for m in matched]),
        list(paths),
    )


def collate_eval(batch):
    batch = [b for b in batch if b is not None]
    waves, labels, paths = zip(*batch)
    return torch.stack(waves), torch.tensor(labels, dtype=torch.float32), list(paths)


def write_debug_protocol(root: Path, n_genuine: int = 6, n_spoof: int = 6, seconds: float = 2.0) -> Path:
    """Tiny local protocol in HyperPotter's column layout, for a CPU smoke run."""
    wav_dir = root / "debug_wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    protocol = root / "debug_protocol.txt"
    sr = 16000
    n = int(seconds * sr)
    t = torch.linspace(0, seconds, n)
    lines = []
    for i in range(n_genuine + n_spoof):
        genuine = i < n_genuine
        if genuine:
            wave = 0.2 * torch.sin(2 * torch.pi * (140 + 15 * i) * t)
            wave = wave * (0.5 + 0.5 * torch.sin(2 * torch.pi * 3 * t))
            label = "bonafide"
        else:
            wave = 0.05 * torch.randn(n)
            label = "spoof"
        path = wav_dir / f"{'g' if genuine else 's'}{i:03d}.wav"
        _write_wav(path, wave, sr)
        # indices: 0 speaker, 1 utt, 2 -, 3 -, 4 label, 5 path, 6 -, 7 -, 8 split
        lines.append(f"spk {path.stem} - - {label} {path.as_posix()} - - train")
    protocol.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return protocol


def _read_wav_stdlib(path: str) -> tuple[torch.Tensor, int]:
    import wave as wave_mod

    with wave_mod.open(path, "rb") as handle:
        sr = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        raw = handle.readframes(handle.getnframes())
    if width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"unsupported wav width {width}: {path}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return torch.from_numpy(audio.copy()), sr


def _write_wav(path: Path, wave: torch.Tensor, sr: int) -> None:
    audio = wave.detach().cpu().clamp(-1, 1).numpy().astype(np.float32)
    try:
        import soundfile as sf

        sf.write(str(path), audio, sr)
        return
    except Exception:
        pass
    import wave as wave_mod

    pcm = (audio * 32767.0).astype(np.int16)
    with wave_mod.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())
