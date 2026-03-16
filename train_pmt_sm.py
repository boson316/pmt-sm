import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

import timm
import torchvision.transforms as T


"""
Day6 PMT-SM training script

Pipeline:
  ViT (timm) -> multi-scale features -> FPN -> global feature
  + MediaPipe-like facial landmarks (MFL) embedding
  -> encoder predicts StyleGAN2 latent w
  -> StyleGAN2 generator reconstructs face image

Loss:
  - L_latent: MSE(pred_w, w_gt)
  - L_img:    MSE(recon_img_small, target_img_small)

Metrics (validation):
  - PSNR
  - SSIM (簡化版實作)
  - FID (可選，需 torchvision Inception v3，且只做 demo 等級)

Dataset:
  預設用 StyleGAN2-ADA FFHQ 預訓模型「自我生成」MT-Dataset：
    - 隨機取樣 z -> G -> img, 取出其中間 latent w
    - 使用 MediaPipe (若安裝) 或 dummy 產生 MFL (landmarks)

依賴：
  - timm
  - torchvision
  - stylegan2-ada-pytorch (clone 在專案目錄)
  - pretrained/ffhq.pkl

這支腳本偏教學 / 範本性質，可依實際專案調整資料來源與網路結構。
"""


ROOT = Path(__file__).resolve().parent
STYLEGAN2_DIR = ROOT / "stylegan2-ada-pytorch"
PRETRAINED_FFHQ = ROOT / "pretrained" / "ffhq.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[PMT-SM] Using device: {device}")


# -----------------------------------------------------------------------------
# ViT + FPN 模組
# -----------------------------------------------------------------------------


class ViTFPN(nn.Module):
    """
    FPN over ViT patch features.
    假設從 ViT 抽到的 patch grid 特徵為 [B, 768, 14, 14]。
    我們建立三個尺度:
      P3: 14x14, C=768
      P4:  7x7, C=384
      P5:  4x4, C=192
    然後用 1x1 conv 對齊通道，再自上而下 FPN 融合，最後輸出最高解析度 feature map。
    """

    def __init__(self, out_ch: int = 256):
        super().__init__()
        self.out_ch = out_ch

        # 固定把 768 維 ViT grid 映射到三個尺度/通道
        self.conv_768_to_384 = nn.Conv2d(768, 384, kernel_size=1)
        self.conv_768_to_192 = nn.Conv2d(768, 192, kernel_size=1)

        # 對應 channels=(192,384,768) -> 統一成 out_ch
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(192, out_ch, kernel_size=1),
                nn.Conv2d(384, out_ch, kernel_size=1),
                nn.Conv2d(768, out_ch, kernel_size=1),
            ]
        )
        self.smooth_convs = nn.ModuleList(
            [nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1) for _ in range(3)]
        )

    def build_multi_scale(self, feat_grid: torch.Tensor):
        """
        feat_grid: [B,768,14,14] 從 ViT patch token 轉回的 2D grid。
        回傳由小到大: [P5 (4x4,192), P4 (7x7,384), P3 (14x14,768)]
        """
        b, c, h, w = feat_grid.shape
        assert c == 768 and h == 14 and w == 14, "Expect [B,768,14,14] from ViT base"

        p3 = feat_grid
        p3_384 = self.conv_768_to_384(p3)
        p3_192 = self.conv_768_to_192(p3)

        p4 = nn.functional.avg_pool2d(p3_384, kernel_size=2, stride=2)  # [B,384,7,7]
        p5 = nn.functional.adaptive_avg_pool2d(p3_192, output_size=(4, 4))  # [B,192,4,4]

        return [p5, p4, p3]

    def forward(self, feat_grid: torch.Tensor):
        feats = self.build_multi_scale(feat_grid)  # [P5,P4,P3]
        assert len(feats) == 3

        laterals = [l(f) for l, f in zip(self.lateral_convs, feats)]

        p5 = laterals[0]
        p4 = laterals[1] + nn.functional.interpolate(
            p5, size=laterals[1].shape[-2:], mode="bilinear", align_corners=False
        )
        p3 = laterals[2] + nn.functional.interpolate(
            p4, size=laterals[2].shape[-2:], mode="bilinear", align_corners=False
        )

        p5 = self.smooth_convs[0](p5)
        p4 = self.smooth_convs[1](p4)
        p3 = self.smooth_convs[2](p3)
        return p3  # [B,out_ch,14,14]


def vit_forward_features_grid(vit: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    將 timm ViT (vit_base_patch16_224, num_classes=0) 的 forward_features 結果
    轉回 patch grid:
      feat_grid: [B,768,14,14]
      feat_tokens: [B,197,768]
    """
    feat_tokens = vit.forward_features(x)  # [B,197,768]
    b, n, c = feat_tokens.shape
    assert n == 197 and c == 768

    patch_tok = feat_tokens[:, 1:]  # [B,196,768]
    h = w = int(patch_tok.shape[1] ** 0.5)
    assert h * w == 196
    feat_grid = patch_tok.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    return feat_grid, feat_tokens


def create_vit_backbone():
    vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    return vit


# -----------------------------------------------------------------------------
# MFL (facial landmarks) encoder
# -----------------------------------------------------------------------------


class LandmarkEncoder(nn.Module):
    """
    將 2D landmarks (N,L,2) 壓成 embedding。
    """

    def __init__(self, num_landmarks: int = 68, emb_dim: int = 256):
        super().__init__()
        in_dim = num_landmarks * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, emb_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, lm: torch.Tensor) -> torch.Tensor:
        # lm: [B,L,2] 或 [B,2L]
        if lm.dim() == 3:
            b, l, d = lm.shape
            assert d == 2
            lm = lm.reshape(b, l * 2)
        return self.net(lm)


def try_mediapipe_landmarks(img: Image.Image, num_landmarks: int = 468) -> np.ndarray:
    """
    嘗試使用 MediaPipe FaceMesh 產生 landmarks。
    若 mediapipe 未安裝、API 異常（例如無 solutions）或偵測失敗，回傳全零陣列，讓流程不中斷。
    """
    try:
        import mediapipe as mp
        face_mesh_cls = getattr(mp, "solutions", None)
        if face_mesh_cls is None:
            return np.zeros((num_landmarks, 2), dtype=np.float32)
        face_mesh_cls = getattr(face_mesh_cls, "face_mesh", None)
        if face_mesh_cls is None:
            return np.zeros((num_landmarks, 2), dtype=np.float32)

        with face_mesh_cls(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            img_rgb = np.array(img.convert("RGB"))
            results = face_mesh.process(img_rgb)
            if not results.multi_face_landmarks:
                return np.zeros((num_landmarks, 2), dtype=np.float32)

            h, w, _ = img_rgb.shape
            pts = []
            for lm in results.multi_face_landmarks[0].landmark[:num_landmarks]:
                pts.append([lm.x * w, lm.y * h])
            pts = np.array(pts, dtype=np.float32)
            pts[:, 0] /= w
            pts[:, 1] /= h
            return pts
    except Exception:
        return np.zeros((num_landmarks, 2), dtype=np.float32)


# -----------------------------------------------------------------------------
# StyleGAN2 載入與 PMT-SM 模型
# -----------------------------------------------------------------------------


def load_stylegan2(pretrained_pkl: Path):
    if not STYLEGAN2_DIR.exists():
        raise FileNotFoundError(
            f"stylegan2-ada-pytorch not found at {STYLEGAN2_DIR}. "
            "Please git clone NVlabs/stylegan2-ada-pytorch into this folder."
        )
    if not pretrained_pkl.exists():
        raise FileNotFoundError(
            f"Pretrained ffhq.pkl not found at {pretrained_pkl}. "
            "Please download it first (see Day3 script)."
        )

    sys.path.insert(0, str(STYLEGAN2_DIR))
    import legacy  # type: ignore
    import training.networks  # 確保 networks 已載入，之後 patch 的才會被 unpickle 的 class 用到

    # 無 CUDA kernel 時 ref 的 upsample2d 會產出錯誤尺寸。直接替換成 F.interpolate，確保 synthesis 通過。
    _upf = training.networks.upfirdn2d
    _orig_up2d = _upf.upsample2d

    def _safe_upsample2d(x, f, up=2, padding=0, flip_filter=False, gain=1, impl="cuda"):
        upx, upy = (up, up) if isinstance(up, int) else (up[0], up[1])
        _, _, h, w = x.shape
        return torch.nn.functional.interpolate(
            x, size=(h * upy, w * upx), mode="bilinear", align_corners=False
        )

    _upf.upsample2d = _safe_upsample2d

    with open(pretrained_pkl, "rb") as f:
        data = legacy.load_network_pkl(f)
    G = data["G_ema"].to(device).eval()
    return G


class PMTSMNet(nn.Module):
    """
    完整 PMT-SM encoder:
      image -> ViT -> FPN -> global feature
      + landmarks -> LandmarkEncoder -> emb
      -> fusion -> MLP -> predicted StyleGAN2 w

    並內建 StyleGAN2 G 以重建影像 (for loss & metric)。
    """

    def __init__(
        self,
        stylegan_G: nn.Module,
        num_landmarks: int = 468,
        fpn_out_ch: int = 256,
        vit_freeze: bool = True,
    ):
        super().__init__()
        self.G = stylegan_G
        self.vit = create_vit_backbone()
        self.fpn = ViTFPN(out_ch=fpn_out_ch)
        self.lm_enc = LandmarkEncoder(num_landmarks=num_landmarks, emb_dim=256)

        if vit_freeze:
            for p in self.vit.parameters():
                p.requires_grad_(False)

        # 將 FPN feature 做 global pooling -> 256
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # StyleGAN2 mapping 常用 w_dim (512)，這裡檢查一下
        self.w_dim = getattr(self.G.mapping, "w_dim", 512)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fpn_out_ch + 256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.w_dim),
        )
        self._cached_num_ws: Optional[int] = None

    def encode(self, x: torch.Tensor, lm: torch.Tensor) -> torch.Tensor:
        # x: [B,3,224,224], lm: [B,L,2] (已正規化到 [0,1])
        feat_grid, _ = vit_forward_features_grid(self.vit, x)
        fpn_feat = self.fpn(feat_grid)  # [B,fpn_ch,14,14]
        pooled = self.global_pool(fpn_feat).flatten(1)  # [B,fpn_ch]

        lm_emb = self.lm_enc(lm)  # [B,256]
        fusion = torch.cat([pooled, lm_emb], dim=1)
        w = self.fusion_mlp(fusion)  # [B,w_dim]
        return w

    def _get_num_ws(self) -> int:
        """用 G.mapping 輸出維度決定 num_ws，與實際 G(z,c) 一致（FFHQ 1024 為 18）。"""
        if self._cached_num_ws is not None:
            return self._cached_num_ws
        dev = next(self.G.parameters()).device
        with torch.no_grad():
            z = torch.randn(1, self.G.mapping.z_dim, device=dev)
            c = torch.zeros(1, getattr(self.G, "c_dim", 0), device=dev)
            ws = self.G.mapping(z, c)
        self._cached_num_ws = int(ws.shape[1])
        return self._cached_num_ws

    def decode(self, w: torch.Tensor) -> torch.Tensor:
        """
        將 encoder 預測或 GT 的 w 輸入 StyleGAN2 G.synthesis 生成影像。
        """
        num_ws = self._get_num_ws()
        w = w.unsqueeze(1).expand(-1, num_ws, -1)
        w = w.to(torch.float32)

        img = self.G.synthesis(w, noise_mode="const", force_fp32=True)
        # 範圍 [-1,1] -> [0,1]
        img = (img.clamp(-1, 1) + 1) * 0.5
        return img

    def forward(
        self,
        x: torch.Tensor,
        lm: torch.Tensor,
        w_gt: Optional[torch.Tensor] = None,
        use_decode: bool = True,
    ):
        """
        訓練時：
          - use_decode=True：回傳 pred_w, recon_pred, recon_gt（會呼叫 G.synthesis，需 CUDA kernel 或相容環境）
          - use_decode=False：只回傳 pred_w，不呼叫 synthesis（僅 latent 訓練，Windows 無 MSVC 可跑）
        """
        pred_w = self.encode(x, lm)
        if not use_decode:
            return pred_w, None, None
        recon_pred = self.decode(pred_w)
        if w_gt is not None:
            recon_gt = self.decode(w_gt)
            return pred_w, recon_pred, recon_gt
        return pred_w, recon_pred, None


# -----------------------------------------------------------------------------
# Dataset: 使用 StyleGAN2 自行合成 MT-Dataset
# -----------------------------------------------------------------------------


class SyntheticPMTDataset(Dataset):
    """
    使用 StyleGAN2-ADA FFHQ generator 自行產生 (img, w, landmarks)。
    適合 demo / 原型，不適合作為最終高品質訓練資料。
    """

    def __init__(
        self,
        G: nn.Module,
        length: int = 2000,
        img_size: int = 224,
        num_landmarks: int = 468,
    ):
        super().__init__()
        self.G = G
        self.length = length
        self.img_size = img_size
        self.num_landmarks = num_landmarks

        self.to_tensor = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        # StyleGAN Mapping module
        self.mapping = self.G.mapping
        self.z_dim = getattr(self.mapping, "z_dim", 512)
        self.c_dim = getattr(self.G, "c_dim", 0)

    def __len__(self):
        return self.length

    @torch.no_grad()
    def _sample_pair(self, idx: int):
        # 為了可重現性，用 idx 固定 random seed（簡易做法）
        rng = torch.Generator(device=device)
        rng.manual_seed(idx)

        z = torch.randn(1, self.z_dim, generator=rng, device=device)
        if self.c_dim > 0:
            c = torch.zeros(1, self.c_dim, device=device)
        else:
            c = None

        # 使用完整 G(z, c, ...) 產生影像，並用 mapping(z, c) 取得對應 w
        w_all = self.mapping(z, c)  # [1,num_ws,w_dim]
        w_avg = w_all.mean(dim=1)  # [1,w_dim]

        img = self.G(z, c, truncation_psi=1.0, noise_mode="const")
        img = (img.clamp(-1, 1) + 1) * 0.5  # [0,1]

        img_np = (
            img[0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )  # HWC, float32
        img_np = (img_np * 255.0).astype(np.uint8)
        pil = Image.fromarray(img_np, "RGB")

        # landmarks (使用 mediapipe，若失敗則全零)
        lm = try_mediapipe_landmarks(pil, num_landmarks=self.num_landmarks)
        if lm.shape[0] != self.num_landmarks:
            lm_pad = np.zeros((self.num_landmarks, 2), dtype=np.float32)
            lm_pad[: lm.shape[0]] = lm
            lm = lm_pad

        img_t = self.to_tensor(pil)  # [3,H,W] in [-1,1] after Normalize
        lm_t = torch.from_numpy(lm).float()  # [L,2]

        return img_t, lm_t, w_avg.squeeze(0).detach().cpu()

    def __getitem__(self, idx):
        img_t, lm_t, w_avg = self._sample_pair(idx)
        return img_t, lm_t, w_avg


# -----------------------------------------------------------------------------
# Metrics: PSNR / SSIM (簡化) / FID (可選)
# -----------------------------------------------------------------------------


def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    # img in [0,1], shape [B,3,H,W]
    mse = nn.functional.mse_loss(img1, img2)
    if mse == 0:
        return torch.tensor(99.0, device=img1.device)
    return 20 * torch.log10(max_val) - 10 * torch.log10(mse)


def ssim_simple(
    img1: torch.Tensor,
    img2: torch.Tensor,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """
    簡化版 SSIM (不做多尺度)，僅做 demo，非嚴格標準實作。
    """
    # 轉成灰階 [B,1,H,W]
    if img1.shape[1] == 3:
        img1 = 0.299 * img1[:, 0] + 0.587 * img1[:, 1] + 0.114 * img1[:, 2]
        img2 = 0.299 * img2[:, 0] + 0.587 * img2[:, 1] + 0.114 * img2[:, 2]
        img1 = img1.unsqueeze(1)
        img2 = img2.unsqueeze(1)

    mu1 = nn.functional.avg_pool2d(img1, 7, 1, 0)
    mu2 = nn.functional.avg_pool2d(img2, 7, 1, 0)

    sigma1_sq = nn.functional.avg_pool2d(img1 * img1, 7, 1, 0) - mu1 ** 2
    sigma2_sq = nn.functional.avg_pool2d(img2 * img2, 7, 1, 0) - mu2 ** 2
    sigma12 = nn.functional.avg_pool2d(img1 * img2, 7, 1, 0) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


@torch.no_grad()
def fid_simple(
    real_imgs: torch.Tensor,
    fake_imgs: torch.Tensor,
    device: torch.device,
) -> Optional[float]:
    """
    極簡 FID 估計：
      - 使用 torchvision Inception-v3 pool3 特徵
      - 僅做單 batch demo，非嚴謹實作
    若 torchvision 或 Inception 下載失敗，回傳 None。
    """
    try:
        from torchvision.models import inception_v3
    except Exception:
        print("[WARN] torchvision.models.inception_v3 not available, skip FID.")
        return None

    model = inception_v3(pretrained=True, transform_input=False, aux_logits=False)
    model.fc = nn.Identity()
    model.to(device).eval()

    def _preprocess(x: torch.Tensor) -> torch.Tensor:
        # x in [0,1], resize to 299x299
        x = nn.functional.interpolate(
            x, size=(299, 299), mode="bilinear", align_corners=False
        )
        return x

    real = _preprocess(real_imgs)
    fake = _preprocess(fake_imgs)

    def _feat(m, img):
        return m(img).detach()

    f_r = _feat(model, real)
    f_f = _feat(model, fake)

    mu_r = f_r.mean(dim=0)
    mu_f = f_f.mean(dim=0)
    sigma_r = torch.cov(f_r.T)
    sigma_f = torch.cov(f_f.T)

    diff = mu_r - mu_f
    diff_sq = diff.dot(diff)
    # sqrt of product of covariance matrices (approx，用 trace)
    covmean_trace = torch.trace(
        torch.linalg.sqrtm((sigma_r @ sigma_f).cpu().numpy()).real
    )
    fid = diff_sq + torch.trace(sigma_r + sigma_f) - 2 * covmean_trace
    return float(fid.cpu().item())


# -----------------------------------------------------------------------------
# 訓練主迴圈
# -----------------------------------------------------------------------------


def train_pmt_sm(
    total_steps: int = 1000,
    batch_size: int = 4,
    lr: float = 1e-4,
    img_size: int = 224,
    num_landmarks: int = 468,
    use_amp: bool = True,
    use_synthesis_in_training: bool = False,
):
    # 抑制 StyleGAN2 在無 MSVC 時重複的 CUDA kernel fallback 警告
    if not use_synthesis_in_training:
        print("[PMT-SM] use_synthesis_in_training=False: 僅 latent MSE，不呼叫 G.synthesis（Windows 無 MSVC 可跑）")
    warnings.filterwarnings(
        "ignore",
        message=".*(upfirdn2d|bias_act|Falling back to slow).*",
        category=UserWarning,
    )
    print("[PMT-SM] Loading StyleGAN2-ADA FFHQ...")
    G = load_stylegan2(PRETRAINED_FFHQ)

    print("[PMT-SM] Building synthetic dataset...")
    dataset = SyntheticPMTDataset(
        G=G,
        length=total_steps * batch_size,
        img_size=img_size,
        num_landmarks=num_landmarks,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    print("[PMT-SM] Building PMTSMNet...")
    model = PMTSMNet(
        stylegan_G=G,
        num_landmarks=num_landmarks,
        fpn_out_ch=256,
        vit_freeze=True,
    ).to(device)

    # 僅訓練 encoder / fusion，StyleGAN G 固定
    for p in model.G.parameters():
        p.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=lr, betas=(0.9, 0.99))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = GradScaler(enabled=use_amp)

    out_dir = ROOT / "pmt_sm_ckpt"
    out_dir.mkdir(exist_ok=True)

    step = 0
    model.train()

    for epoch in range(999999):  # 以 step 控制結束
        for img_t, lm_t, w_gt in loader:
            if step >= total_steps:
                break
            step += 1

            img_t = img_t.to(device, non_blocking=True)
            lm_t = lm_t.to(device, non_blocking=True)
            w_gt = w_gt.to(device, non_blocking=True)

            # StyleGAN 產的 img 是 [0,1]，但 Dataset 先 Normalize 到 [-1,1]，
            # 因此這裡需要還原回 [0,1] 再與重建圖比較。
            target_img = (img_t * 0.5) + 0.5  # [-1,1] -> [0,1]

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                pred_w, recon_pred, recon_gt = model(
                    img_t, lm_t, w_gt=w_gt, use_decode=use_synthesis_in_training
                )

                latent_loss = nn.functional.mse_loss(pred_w, w_gt)
                if use_synthesis_in_training and recon_pred is not None and recon_gt is not None:
                    recon_pred_small = nn.functional.interpolate(
                        recon_pred, size=(img_size, img_size), mode="area"
                    )
                    recon_gt_small = nn.functional.interpolate(
                        recon_gt, size=(img_size, img_size), mode="area"
                    )
                    img_loss_pred = nn.functional.mse_loss(recon_pred_small, target_img)
                    img_loss_gt = nn.functional.mse_loss(recon_gt_small, target_img)
                    loss = latent_loss + img_loss_pred + 0.5 * img_loss_gt
                else:
                    loss = latent_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % 10 == 0:
                lr_now = scheduler.get_last_lr()[0]
                if use_synthesis_in_training and recon_pred is not None:
                    with torch.no_grad():
                        recon_s = nn.functional.interpolate(
                            recon_pred, size=(img_size, img_size), mode="area"
                        )
                        psnr_val = psnr(recon_s, target_img).item()
                        ssim_val = ssim_simple(recon_s, target_img).item()
                    print(
                        f"[step {step}/{total_steps}] loss={loss.item():.4f} latent={latent_loss.item():.4f} "
                        f"PSNR={psnr_val:.2f} SSIM={ssim_val:.3f} lr={lr_now:.2e}"
                    )
                else:
                    print(
                        f"[step {step}/{total_steps}] loss={loss.item():.4f} latent={latent_loss.item():.4f} lr={lr_now:.2e}"
                    )

            if step % 200 == 0 or step == total_steps:
                ckpt_path = out_dir / f"pmt_sm_step{step}.pt"
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"[PMT-SM] Saved checkpoint: {ckpt_path}")

        if step >= total_steps:
            break

    print("[PMT-SM] Training finished.")


if __name__ == "__main__":
    # 這裡給一組預設超參數，可依需要在命令列換成 argparse
    train_pmt_sm(
        total_steps=300,
        batch_size=4,
        lr=1e-4,
        img_size=224,
        num_landmarks=468,
        use_amp=True,
        use_synthesis_in_training=False,  # True 需 MSVC/CUDA kernel；False 僅 latent 訓練可跑
    )

