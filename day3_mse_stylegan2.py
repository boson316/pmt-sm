import math
import os
from io import BytesIO
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image


# 路徑與裝置設定
ROOT = Path(__file__).resolve().parent
STYLEGAN2_DIR = ROOT / "stylegan2-ada-pytorch"
PRETRAINED_DIR = ROOT / "pretrained"
FFHQ_PKL = PRETRAINED_DIR / "ffhq.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 1. 載入 StyleGAN2-ADA-PyTorch G_ema（FFHQ 預訓）
def load_generator():
    if not FFHQ_PKL.exists():
        raise FileNotFoundError(
            f"Pretrained FFHQ model not found at {FFHQ_PKL}.\n"
            "Please run day3_stylegan2_ffhq.py or download ffhq.pkl manually."
        )
    if not STYLEGAN2_DIR.exists():
        raise FileNotFoundError(
            f"stylegan2-ada-pytorch repo not found at {STYLEGAN2_DIR}.\n"
            "Please git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git stylegan2-ada-pytorch"
        )

    import sys

    sys.path.insert(0, str(STYLEGAN2_DIR))
    try:
        import legacy  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "Failed to import legacy from stylegan2-ada-pytorch. "
            "Check that the repo was cloned correctly."
        ) from e

    print(f"Loading G_ema from: {FFHQ_PKL}")
    with open(FFHQ_PKL, "rb") as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)
    G.eval()
    return G


# 2. 讀取圖片（本地或 URL），轉成 StyleGAN2 輸入格式 [-1,1]
def load_image_for_projection(path_or_url: str, size: int = 1024) -> torch.Tensor:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        print(f"Downloading image: {path_or_url}")
        resp = requests.get(path_or_url, timeout=15)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        img_path = Path(path_or_url)
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {path_or_url}")
        img = Image.open(img_path).convert("RGB")

    img = img.resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    arr = arr.transpose(2, 0, 1)  # CHW
    arr = arr[np.newaxis, ...]  # NCHW
    arr = arr / 127.5 - 1.0  # [-1,1]
    return torch.from_numpy(arr).to(device)


# 3. 透過 MSE 對 W 空間做簡單投影（AdaIN 風格向量）
def project_to_w(
    G,
    target_img: torch.Tensor,
    num_steps: int = 200,
    w_lr: float = 0.05,
) -> torch.Tensor:
    """
    使用像素 MSE，對單張影像做簡單 W 投影：
    - 初始化 w 為 w_avg
    - 反向傳遞優化 w，使 G(w) 接近 target_img
    回傳 shape: [num_ws, w_dim] 的 W 風格向量。
    """
    assert target_img.shape[0] == 1, "Only batch size 1 is supported."

    # 取得 W 空間資訊
    w_avg = G.mapping.w_avg  # [w_dim]
    num_ws = G.num_ws
    w_dim = w_avg.shape[-1]

    # 將 w_avg 擴展成 [1, num_ws, w_dim] 作為初始值
    w_opt = w_avg.unsqueeze(0).repeat(num_ws, 1).unsqueeze(0).detach().clone()
    w_opt.requires_grad_(True)

    optimizer = torch.optim.Adam([w_opt], lr=w_lr)

    print(
        f"Starting W projection: num_steps={num_steps}, "
        f"num_ws={num_ws}, w_dim={w_dim}"
    )

    for step in range(num_steps):
        optimizer.zero_grad()
        synth = G.synthesis(w_opt, noise_mode="const")  # [1,3,H,W]

        loss = F.mse_loss(synth, target_img)
        loss.backward()
        optimizer.step()

        if (step + 1) % max(1, num_steps // 10) == 0 or step == num_steps - 1:
            print(f"  step {step+1:4d}/{num_steps}, MSE={loss.item():.6f}")

    return w_opt.detach().squeeze(0)  # [num_ws,w_dim]


def w_to_image(G, w: torch.Tensor) -> torch.Tensor:
    """給定 W 向量 [num_ws,w_dim]，產生對應影像 [1,3,H,W]（不做截斷）。"""
    if w.ndim == 2:
        w = w.unsqueeze(0)
    with torch.no_grad():
        img = G.synthesis(w.to(device), noise_mode="const")
    return img


def tensor_to_numpy_image(img: torch.Tensor) -> np.ndarray:
    """將 [-1,1] tensor [1,3,H,W] 轉成 HWC uint8。"""
    img = (img.clamp(-1, 1) + 1) * 0.5  # [0,1]
    img = img[0].permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255).round().astype("uint8")
    return img


def demo_mse_style_projection(
    G,
    source_path_or_url: str,
    target_path_or_url: str,
    num_steps: int = 200,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 讀取兩張臉照
    src_img_t = load_image_for_projection(source_path_or_url)
    tgt_img_t = load_image_for_projection(target_path_or_url)

    print("Projecting source image to W...")
    w_src = project_to_w(G, src_img_t, num_steps=num_steps)

    print("Projecting target image to W...")
    w_tgt = project_to_w(G, tgt_img_t, num_steps=num_steps)

    print(
        f"Source W shape: {tuple(w_src.shape)}, mean={w_src.mean().item():.4f}, "
        f"std={w_src.std().item():.4f}"
    )
    print(
        f"Target W shape: {tuple(w_tgt.shape)}, mean={w_tgt.mean().item():.4f}, "
        f"std={w_tgt.std().item():.4f}"
    )

    # 風格插值（W 空間線性插值）
    alpha = 0.5
    w_interp = alpha * w_src + (1.0 - alpha) * w_tgt

    img_src = tensor_to_numpy_image(w_to_image(G, w_src))
    img_tgt = tensor_to_numpy_image(w_to_image(G, w_tgt))
    img_interp = tensor_to_numpy_image(w_to_image(G, w_interp))

    # 可視化與保存 PNG
    out_path = ROOT / "day3_mse_interp.png"
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_src)
    plt.title("Source style (W_src)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img_interp)
    plt.title("Interpolated (0.5*W_src+0.5*W_tgt)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img_tgt)
    plt.title("Target style (W_tgt)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved style interpolation PNG to: {out_path}")

    return w_src, w_tgt


def main():
    G = load_generator()

    # 預設 demo：請自行替換為實際來源 / 目標臉照（本地或 URL）
    # 建議：可將 day2_mfl.py 中的 me.png 當作 source，本地路徑例如 "me.png"
    source = "me.png"  # 或 http(s) URL
    target = "me.png"  # demo: 用同一張，方便先確認流程

    try:
        w_src, w_tgt = demo_mse_style_projection(
            G,
            source_path_or_url=source,
            target_path_or_url=target,
            num_steps=20,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] Style projection failed: {e}")
        return

    # 儲存 W 空間向量為 numpy 檔便於後續使用
    np.save(ROOT / "day3_w_source.npy", w_src.detach().cpu().numpy())
    np.save(ROOT / "day3_w_target.npy", w_tgt.detach().cpu().numpy())
    print("Saved W-space vectors: day3_w_source.npy, day3_w_target.npy")
    print("StyleGAN2 MSE-based W-space style extraction completed.")


if __name__ == "__main__":
    main()

