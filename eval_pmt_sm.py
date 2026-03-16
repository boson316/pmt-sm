"""
Day7 PMT-SM 評估腳本（範本版）

功能概述：
- 載入已訓練好的 PMT-SM 編碼器與 StyleGAN2-ADA G
- 載入一組「源臉／目標臉」資料（簡化為同一資料夾中的圖片對）
- 產生妝容轉移結果（這裡假設你已有對應的轉換函式，先留 TODO）
- 評估：
    - ArcFace 相似度（insightface）
    - SSIM（torchmetrics）
    - LPIPS
    - FID（簡易實作或 torchmetrics）
- 輸出 aggregated metrics 並將逐張結果存成 CSV

注意：
- BeautyGAN / PSGAN baseline、真實 BSR 使用者研究模擬、完整 FID pipeline
  需要額外模型或資料集，這裡先留清楚的 TODO hook，讓你之後可以補上。
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from train_pmt_sm import PMTSMNet, load_stylegan2, PRETRAINED_FFHQ, ROOT, device


class FacePairDataset(Dataset):
    """
    簡化版資料集：
    - root/src/*.png  作為「源臉」
    - root/tgt/*.png  作為「目標臉」（與 src 按檔名排序配對）
    - 姿態／landmarks 先用 dummy，後續可由 MediaPipe/pose estimator 補上
    """

    def __init__(self, root: Path, img_size: int = 256):
        super().__init__()
        self.root = root
        self.src_dir = root / "src"
        self.tgt_dir = root / "tgt"

        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        self.src_paths = sorted(
            [p for p in self.src_dir.glob("*") if p.suffix.lower() in exts]
        )
        self.tgt_paths = sorted(
            [p for p in self.tgt_dir.glob("*") if p.suffix.lower() in exts]
        )
        assert len(self.src_paths) > 0, f"No images in {self.src_dir}"
        assert len(self.src_paths) == len(
            self.tgt_paths
        ), "src / tgt image count mismatch"

        self.transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.src_paths)

    def __getitem__(self, idx):
        src_path = self.src_paths[idx]
        tgt_path = self.tgt_paths[idx]

        src_img = Image.open(src_path).convert("RGB")
        tgt_img = Image.open(tgt_path).convert("RGB")

        src_t = self.transform(src_img)
        tgt_t = self.transform(tgt_img)

        # TODO: 這裡目前用 dummy landmarks (全 0)，之後可接 MediaPipe
        num_landmarks = 468
        lm_src = torch.zeros(num_landmarks, 2, dtype=torch.float32)
        lm_tgt = torch.zeros(num_landmarks, 2, dtype=torch.float32)

        return src_t, tgt_t, lm_src, lm_tgt, str(src_path.name)


def load_arcface(device: torch.device):
    """
    載入 insightface 的 ArcFace 模型，用於臉部相似度評估。
    """
    try:
        import insightface
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] insightface not installed: {e}")
        return None

    model = insightface.app.FaceAnalysis(
        name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    model.prepare(ctx_id=0 if device.type == "cuda" else -1)
    return model


def arcface_cosine_sim(
    arcface_app, img_a: np.ndarray, img_b: np.ndarray
) -> float:
    """
    使用 insightface 的 embedding 做 cosine similarity。
    img_*: HWC, BGR, uint8
    """
    if arcface_app is None:
        return float("nan")

    fa = arcface_app.get(img_a)
    fb = arcface_app.get(img_b)
    if len(fa) == 0 or len(fb) == 0:
        return float("nan")

    emb_a = fa[0].embedding
    emb_b = fb[0].embedding
    num = (emb_a * emb_b).sum()
    den = np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8
    return float(num / den)


def build_metrics(device: torch.device):
    """
    建立 torchmetrics / LPIPS 等度量工具。
    """
    try:
        import torchmetrics

        ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(
            device
        )
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] torchmetrics not available ({e}), SSIM disabled.")
        ssim_metric = None

    try:
        import lpips

        lpips_metric = lpips.LPIPS(net="vgg").to(device)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] LPIPS not available ({e}), LPIPS disabled.")
        lpips_metric = None

    # TODO: FID 建議用 torchmetrics.image.FrechetInceptionDistance
    fid_metric = None

    return ssim_metric, lpips_metric, fid_metric


def evaluate_pmt_sm(
    ckpt_path: Path,
    pair_root: Path,
    batch_size: int = 2,
    img_size: int = 256,
):
    print(f"[Eval] Using device: {device}")

    # 1) 載入 StyleGAN2 與 PMT-SM 模型
    print("[Eval] Loading StyleGAN2-ADA FFHQ...")
    G = load_stylegan2(PRETRAINED_FFHQ)
    print("[Eval] Building PMTSMNet...")
    model = PMTSMNet(
        stylegan_G=G,
        num_landmarks=468,
        fpn_out_ch=256,
        vit_freeze=True,
    ).to(device)

    # 只載入 encoder/fusion 相關權重
    print(f"[Eval] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # 2) 準備資料集
    dataset = FacePairDataset(pair_root, img_size=img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 3) 準備 metrics
    arcface_app = load_arcface(device)
    ssim_metric, lpips_metric, fid_metric = build_metrics(device)

    all_results = []

    for src_t, tgt_t, lm_src, lm_tgt, names in loader:
        src_t = src_t.to(device)
        tgt_t = tgt_t.to(device)
        lm_src = lm_src.to(device)
        # 目前只示範：源臉 → 預測 latent → 用 G 生成；目標臉只用來比相似度

        with torch.no_grad():
            # encode 源臉 + landmarks
            pred_w, _, _ = model(src_t, lm_src, w_gt=None, use_decode=False)

            # decode 成影像（注意：這一步在 Windows 可能較慢）
            recon = model.decode(pred_w)
            recon_clamped = recon.clamp(0, 1)

        # 評估每一張
        for i in range(src_t.size(0)):
            name = names[i]
            src_img = src_t[i : i + 1]
            tgt_img = tgt_t[i : i + 1]
            rec_img = recon_clamped[i : i + 1]

            # SSIM
            if ssim_metric is not None:
                ssim_val = float(ssim_metric(rec_img, tgt_img).item())
            else:
                ssim_val = float("nan")

            # LPIPS
            if lpips_metric is not None:
                lp = lpips_metric(
                    (rec_img * 2 - 1), (tgt_img * 2 - 1)
                ).item()  # LPIPS 期待 [-1,1]
                lpips_val = float(lp)
            else:
                lpips_val = float("nan")

            # ArcFace 相似度：先轉回 HWC BGR uint8
            def to_bgr_uint8(t: torch.Tensor) -> np.ndarray:
                x = t[0].permute(1, 2, 0).detach().cpu().numpy()  # HWC, [0,1]
                x = (x * 255.0).clip(0, 255).astype("uint8")
                x_bgr = x[:, :, ::-1]
                return x_bgr

            if arcface_app is not None:
                tgt_bgr = to_bgr_uint8(tgt_img)
                rec_bgr = to_bgr_uint8(rec_img)
                arcface_sim = arcface_cosine_sim(arcface_app, tgt_bgr, rec_bgr)
            else:
                arcface_sim = float("nan")

            all_results.append(
                {
                    "name": name,
                    "ssim": ssim_val,
                    "lpips": lpips_val,
                    "arcface_sim": arcface_sim,
                }
            )
            print(
                f"[Eval] {name}: SSIM={ssim_val:.3f}, LPIPS={lpips_val:.3f}, ArcFace={arcface_sim:.3f}"
            )

    # TODO: FID、BSR A/B test、BeautyGAN/PSGAN baseline 比較可在這裡接上

    # 匯出 CSV
    import csv  # noqa: PLC0415

    out_csv = ROOT / "pmt_sm_eval_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "ssim", "lpips", "arcface_sim"]
        )
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"[Eval] Saved per-image metrics to: {out_csv}")


if __name__ == "__main__":
    # 預設使用 step50 checkpoint 與 demo 資料夾 pairs_demo/src, pairs_demo/tgt
    ckpt = ROOT / "pmt_sm_ckpt" / "pmt_sm_step50.pt"
    pair_root = ROOT / "pairs_demo"
    evaluate_pmt_sm(ckpt, pair_root, batch_size=2, img_size=256)

