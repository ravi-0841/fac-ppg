from typing import Optional

import torch
import torch.nn as nn


class STFT(nn.Module):
    def __init__(
        self,
        win_size: int = 400,  # 512,
        hop_size: int = 320,  # 128,
        fft_size: Optional[int] = 512,  # None,
        win_type: Optional[str] = "hann",
    ) -> None:
        super(STFT, self).__init__()

        self.win_size: int = win_size
        self.hop_size: int = hop_size
        self.fft_size: int = fft_size if fft_size is not None else win_size

        assert win_type in {
            None,
            "hamming",
            "hann",
        }, f"Unsupported window type {win_type}"
        if win_type == "hamming":
            window = torch.hamming_window(win_size, periodic=False)
        elif win_type == "hann":
            window = torch.hann_window(win_size, periodic=False)
        else:
            window = torch.ones(win_size)

        self.register_buffer("win", window)

    def forward(
        self,
        x: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        assert mode in {"stft", "istft"}
        if mode == "stft":
            out = x
            out = torch.stft(
                out,
                self.fft_size,
                self.hop_size,
                self.win_size,
                window=self.win,
                pad_mode="constant",
            )
            out = out.transpose(1, 3).contiguous()
        else:
            out = x
            out = out.transpose(1, 3).contiguous()
            out = torch.istft(
                out,
                self.fft_size,
                self.hop_size,
                self.win_size,
                window=self.win,
            )
        return out
