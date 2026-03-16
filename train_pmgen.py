import copy
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


ROOT = Path(__file__).resolve().parent
STYLEGAN2_DIR = ROOT / "stylegan2-ada-pytorch"
PRETRAINED = ROOT / "pretrained" / "ffhq.pkl"
CSV_PATH = ROOT / "faces.csv"  # img_path,arcface_emb(optional)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_ARCFACE = False  # 先關閉 torch.hub ArcFace，避免下載問題
MAX_TRAIN_RES = 256  # 省 GPU：訓練/判別只在 <=256 的尺度進行
BATCH_SIZE = 1       # 省 GPU：RTX3050 建議先 1
TOTAL_STEPS = 20     # 省 GPU：先跑通流程（可自行調大）


class FaceCSVDataset(Dataset):
    def __init__(self, csv_path: Path, image_root: Path, img_size: int = 256):
        import csv

        self.image_root = image_root
        self.img_size = img_size
        self.entries = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append(row)

        self.to_tensor = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.arcface = None
        if USE_ARCFACE:
            try:
                self.arcface = torch.hub.load(
                    "deepinsight/insightface",
                    "arcface_r100_v1",
                    pretrained=True,
                    trust_repo=True,
                ).eval().to(device)
                self.arcface_resize = T.Compose(
                    [T.Resize((112, 112)), T.ToTensor(), T.Normalize(0.5, 0.5)]
                )
                print("ArcFace loaded from torch.hub.")
            except Exception as e:  # noqa: BLE001
                print(f"Warning: ArcFace load failed, use zero embeddings. ({e})")
                self.arcface = None
        else:
            self.arcface_resize = None

    def __len__(self):
        return len(self.entries)

    @torch.no_grad()
    def _compute_arcface(self, img: Image.Image) -> np.ndarray:
        if self.arcface is None:
            # 沒有 ArcFace 模型時，回傳全零 embedding 以便流程繼續
            return np.zeros(512, dtype=np.float32)
        x = self.arcface_resize(img).unsqueeze(0).to(device)
        emb = self.arcface(x)
        emb = nn.functional.normalize(emb, dim=-1)
        return emb.cpu().numpy()[0]

    def __getitem__(self, idx):
        row = self.entries[idx]
        img_path = self.image_root / row["img_path"]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.to_tensor(img)

        if "arcface_emb" in row and row["arcface_emb"]:
            emb = np.fromstring(row["arcface_emb"], sep=" ")
        else:
            emb = self._compute_arcface(img)

        emb_tensor = torch.from_numpy(emb).float()
        return img_tensor, emb_tensor


def load_stylegan2(pretrained_pkl: Path):
    if not STYLEGAN2_DIR.exists():
        raise FileNotFoundError(
            f"stylegan2-ada-pytorch not found at {STYLEGAN2_DIR}"
        )
    if not pretrained_pkl.exists():
        raise FileNotFoundError(
            f"Pretrained ffhq.pkl not found at {pretrained_pkl}"
        )

    sys.path.insert(0, str(STYLEGAN2_DIR))
    import legacy  # type: ignore

    with open(pretrained_pkl, "rb") as f:
        data = legacy.load_network_pkl(f)
    G = data["G_ema"].to(device).float()
    return G


class SimpleDiscriminator(nn.Module):
    """
    省 GPU 的簡易判別器：支援任意解析度（透過 AdaptiveAvgPool2d）。
    用來替換 stylegan2-ada 原本只吃 1024 的 D。
    """

    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_ch * 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class CondMapping(nn.Module):
    def __init__(self, base_mapping, emb_dim=512, w_dim=512):
        super().__init__()
        self.base_mapping = base_mapping
        self.cond_fc = nn.Sequential(
            nn.Linear(emb_dim, w_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(w_dim, w_dim),
        )

    def forward(self, z, c, arcface_emb):
        w = self.base_mapping(z, c)
        cond = self.cond_fc(arcface_emb)
        cond = cond.unsqueeze(1).expand_as(w)
        return w + cond


def get_current_resolution(step: int, total_steps: int) -> int:
    min_log2, max_log2 = 2, 10
    alpha = min(1.0, step / max(1, total_steps))
    cur_log2 = min_log2 + alpha * (max_log2 - min_log2)
    res = 2 ** int(round(cur_log2))
    return min(res, MAX_TRAIN_RES)


def downsample_to(img: torch.Tensor, res: int) -> torch.Tensor:
    _, _, H, W = img.shape
    if H == res and W == res:
        return img
    return nn.functional.interpolate(img, size=(res, res), mode="area")


def d_loss_fn(D, real_img, fake_img):
    real_logits = D(real_img)
    fake_logits = D(fake_img.detach())
    loss_real = nn.functional.softplus(-real_logits).mean()
    loss_fake = nn.functional.softplus(fake_logits).mean()
    return loss_real + loss_fake


def g_loss_fn(D, fake_img):
    logits = D(fake_img)
    return nn.functional.softplus(-logits).mean()


def train_pmgen():
    G = load_stylegan2(PRETRAINED)
    D = SimpleDiscriminator().to(device).float()
    base_mapping = G.mapping
    G.mapping = CondMapping(base_mapping, emb_dim=512, w_dim=base_mapping.w_dim).to(
        device
    )
    G.requires_grad_(True)

    G_ema = copy.deepcopy(G).eval().requires_grad_(False)
    D.train()
    D.requires_grad_(True)

    dataset = FaceCSVDataset(CSV_PATH, ROOT, img_size=256)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    opt_G = optim.AdamW(G.parameters(), lr=1e-4, betas=(0.0, 0.99))
    opt_D = optim.AdamW(D.parameters(), lr=1e-4, betas=(0.0, 0.99))

    total_steps = TOTAL_STEPS
    step = 0

    (ROOT / "pmgen_ckpt").mkdir(exist_ok=True)
    (ROOT / "pmgen_demo").mkdir(exist_ok=True)

    while step < total_steps:
        for real_img, arc_emb in loader:
            if step >= total_steps:
                break
            step += 1

            real_img = real_img.to(device)
            arc_emb = arc_emb.to(device)

            cur_res = get_current_resolution(step, total_steps)
            real_small = downsample_to(real_img, cur_res)

            batch = real_img.size(0)
            z = torch.randn(batch, base_mapping.z_dim, device=device)
            c = torch.zeros(batch, getattr(G, "c_dim", 0), device=device)

            w = G.mapping(z, c, arc_emb)
            fake = G.synthesis(w)
            fake_small = downsample_to(fake, cur_res)

            opt_D.zero_grad(set_to_none=True)
            loss_D = d_loss_fn(D, real_small, fake_small)
            loss_D.backward()
            opt_D.step()

            opt_G.zero_grad(set_to_none=True)
            loss_G = g_loss_fn(D, fake_small)
            rec_loss = nn.functional.mse_loss(fake_small, real_small)
            total_G = loss_G + 10.0 * rec_loss
            total_G.backward()
            opt_G.step()

            with torch.no_grad():
                ema_beta = 0.99
                for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                    p_ema.copy_(ema_beta * p_ema + (1.0 - ema_beta) * p)

            if step % 5 == 0:
                print(
                    f"step {step}/{total_steps} res={cur_res} "
                    f"D={loss_D.item():.4f} G={loss_G.item():.4f} rec={rec_loss.item():.4f}"
                )

    print("train_pmgen finished (demo run).")


if __name__ == "__main__":
    if not CSV_PATH.exists():
        print(f"faces.csv not found at {CSV_PATH}, create it first.")
    else:
        train_pmgen()

