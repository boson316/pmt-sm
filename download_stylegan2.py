import os
import hashlib
import pickle
from pathlib import Path

import requests
import torch
from PIL import Image

URL = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
SHA256 = "a205a346e86a9ddaae702e118097d014b7b8bd719491396a162cca438f2f524c"
PRETRAINED_DIR = Path("pretrained")
PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
PTH = PRETRAINED_DIR / "ffhq.pkl"


def download():
    if PTH.exists():
        print(f"Found existing: {PTH}")
        return
    print(f"Downloading FFHQ pretrained model from: {URL}")
    try:
        with requests.get(URL, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            sha = hashlib.sha256()
            downloaded = 0
            with open(PTH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    f.write(chunk)
                    sha.update(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        print(f"\rDownloaded: {downloaded} / {total} bytes ({pct}%)", end="")
            print()
        digest = sha.hexdigest()
        print(f"File SHA256: {digest}")
        if digest.lower() != SHA256.lower():
            raise ValueError("SHA256 mismatch, please delete ffhq.pkl and retry.")
    except Exception as e:  # noqa: BLE001
        print(f"Download failed: {e}")
        raise


def generate_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if not PTH.exists():
        raise FileNotFoundError(f"{PTH} not found, run download() first.")

    print(f"Loading generator from: {PTH}")
    with open(PTH, "rb") as f:
        data = pickle.load(f)
    # 兼容常見 key 命名：G_ema 或 'G_ema' 在 dict 內
    if isinstance(data, dict) and "G_ema" in data:
        G = data["G_ema"]
    else:
        G = data
    G = G.to(device).eval()

    z_dim = getattr(G, "z_dim", 512)
    c_dim = getattr(G, "c_dim", 0)
    with torch.no_grad():
        z = torch.randn(1, z_dim, device=device)
        c = torch.zeros(1, c_dim, device=device) if c_dim > 0 else None
        img = G(z, c, truncation_psi=0.7, noise_mode="const")

    img = (img.clamp(-1, 1) + 1) * 0.5
    img = img[0].permute(1, 2, 0).cpu().numpy()
    img = (img * 255).round().astype("uint8")
    Image.fromarray(img, "RGB").save("download_stylegan2_demo.png")
    print("Saved download_stylegan2_demo.png")


if __name__ == "__main__":
    download()
    generate_demo()
