import hashlib
import os
import sys
from pathlib import Path

import requests
import torch
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


REPO_DIR = Path(__file__).resolve().parent
# 使用 PyTorch 版官方 repo 資料夾名稱，避免與 TensorFlow 版混淆
STYLEGAN2_DIR = REPO_DIR / "stylegan2-ada-pytorch"
PRETRAINED_DIR = REPO_DIR / "pretrained"
FFHQ_URL = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
FFHQ_PATH = PRETRAINED_DIR / "ffhq.pkl"

# 來源自官方 ffhq.pkl（stylegan2-ada-pytorch）已知 SHA256
FFHQ_SHA256 = "a205a346e86a9ddaae702e118097d014b7b8bd719491396a162cca438f2f524c"


def ensure_stylegan2_repo():
    if not STYLEGAN2_DIR.exists():
        raise FileNotFoundError(
            f"stylegan2-ada-pytorch repo not found at {STYLEGAN2_DIR}. "
            "Please git clone NVlabs/stylegan2-ada-pytorch into ./stylegan2-ada-pytorch first."
        )


def download_with_progress(url: str, dst_path: Path, expected_sha256: str | None = None):
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading pretrained FFHQ model from:\n  {url}")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            hash_obj = hashlib.sha256()
            with open(dst_path, "wb") as f, tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                desc=dst_path.name,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    f.write(chunk)
                    hash_obj.update(chunk)
                    pbar.update(len(chunk))
    except requests.RequestException as e:  # noqa: BLE001
        raise RuntimeError(f"Download failed: {e}") from e

    digest = hash_obj.hexdigest()
    print(f"Downloaded file SHA256: {digest}")
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        raise ValueError(
            f"SHA256 mismatch for {dst_path}.\n"
            f"  expected: {expected_sha256}\n"
            f"  actual:   {digest}\n"
            "Please delete the file and try again."
        )


def ensure_ffhq_pkl():
    if FFHQ_PATH.exists():
        print(f"Found existing pretrained model at: {FFHQ_PATH}")
        # 簡單驗證 hash（如失敗會強制要求重下）
        print("Verifying existing file SHA256...")
        hash_obj = hashlib.sha256()
        with open(FFHQ_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)
        digest = hash_obj.hexdigest()
        print(f"Existing file SHA256: {digest}")
        if digest.lower() != FFHQ_SHA256.lower():
            raise ValueError(
                "Existing ffhq.pkl SHA256 mismatch. "
                "Please delete it and rerun this script to re-download."
            )
        return

    download_with_progress(FFHQ_URL, FFHQ_PATH, expected_sha256=FFHQ_SHA256)


def load_generator():
    """
    載入 FFHQ 預訓 StyleGAN2 G_ema（透過 legacy 相容方式）。
    """
    sys.path.insert(0, str(STYLEGAN2_DIR))
    try:
        import legacy  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            f"Failed to import legacy from {STYLEGAN2_DIR}. "
            "Make sure you cloned the correct NVLabs stylegan2 repo."
        ) from e

    print(f"Loading generator from: {FFHQ_PATH}")
    with open(FFHQ_PATH, "rb") as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(DEVICE)

    G.eval()
    return G


def generate_demo_face(G, out_path: Path):
    """
    使用預訓 StyleGAN2 產生 1024x1024 臉部圖片 demo。
    """
    z_dim = G.z_dim if hasattr(G, "z_dim") else 512
    label_dim = getattr(G, "c_dim", 0)

    with torch.no_grad():
        z = torch.randn(1, z_dim, device=DEVICE)
        if label_dim > 0:
            c = torch.zeros(1, label_dim, device=DEVICE)
        else:
            c = None

        img = G(z, c, truncation_psi=0.7, noise_mode="const")

    # img: [N, C, H, W], N=1, C=3, H=W=1024 (預設)
    img = (img.clamp(-1, 1) + 1) * 0.5  # \in [0,1]
    img = img[0].permute(1, 2, 0).cpu().numpy()  # HWC
    img = (img * 255).round().astype("uint8")

    from PIL import Image

    Image.fromarray(img, "RGB").save(out_path)
    print(f"Saved demo face image to: {out_path}")


def write_download_script():
    """
    在專案目錄自動生成一支簡化版 download_stylegan2.py，
    方便單獨下載 ffhq.pkl 並用 pickle 載入生成 demo。
    """
    script_path = REPO_DIR / "download_stylegan2.py"
    content = f'''import os
import hashlib
import pickle
from pathlib import Path

import requests
import torch
from PIL import Image

URL = "{FFHQ_URL}"
SHA256 = "{FFHQ_SHA256}"
PRETRAINED_DIR = Path("pretrained")
PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
PTH = PRETRAINED_DIR / "ffhq.pkl"


def download():
    if PTH.exists():
        print(f"Found existing: {{PTH}}")
        return
    print(f"Downloading FFHQ pretrained model from: {{URL}}")
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
                        print(f"\\rDownloaded: {{downloaded}} / {{total}} bytes ({{pct}}%)", end="")
            print()
        digest = sha.hexdigest()
        print(f"File SHA256: {{digest}}")
        if digest.lower() != SHA256.lower():
            raise ValueError("SHA256 mismatch, please delete ffhq.pkl and retry.")
    except Exception as e:  # noqa: BLE001
        print(f"Download failed: {{e}}")
        raise


def generate_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {{device}}")
    if not PTH.exists():
        raise FileNotFoundError(f"{{PTH}} not found, run download() first.")

    print(f"Loading generator from: {{PTH}}")
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
'''
    script_path.write_text(content, encoding="utf-8")
    print(f"Helper script generated: {script_path}")


def main():
    try:
        ensure_stylegan2_repo()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    try:
        ensure_ffhq_pkl()
    except (RuntimeError, ValueError) as e:
        print(f"[ERROR] Failed to prepare pretrained model:\n{e}")
        return

    try:
        G = load_generator()
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] Failed to load StyleGAN2 generator:\n{e}")
        return

    out_path = REPO_DIR / "day3_stylegan2_ffhq_demo.png"
    try:
        generate_demo_face(G, out_path)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] Failed to generate demo image:\n{e}")
        return

    print("StyleGAN2 FFHQ demo completed successfully.")

    # 自動生成簡化版下載 + demo 腳本
    write_download_script()


if __name__ == "__main__":
    main()

