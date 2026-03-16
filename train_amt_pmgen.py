import os
import sys
from pathlib import Path

import imageio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from PIL import Image

import lpips  # type: ignore
import wandb  # type: ignore


ROOT = Path(__file__).resolve().parent
AMT_DIR = ROOT / "amt_gan"
STYLEGAN2_DIR = ROOT / "stylegan2-ada-pytorch"
PMGEN_CKPT_DIR = ROOT / "pmgen_ckpt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MakeupTransferDataset(Dataset):
    def __init__(self, list_file: Path, img_root: Path, img_size: int = 256):
        self.pairs = []
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                src, tgt = line.split(",")
                self.pairs.append((src, tgt))

        self.img_root = img_root
        self.to_tensor = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.pairs)

    def _load(self, rel):
        img = Image.open(self.img_root / rel).convert("RGB")
        return self.to_tensor(img)

    def __getitem__(self, idx):
        src_rel, tgt_rel = self.pairs[idx]
        return self._load(src_rel), self._load(tgt_rel)


def load_amt_gan():
    if not AMT_DIR.exists():
        raise FileNotFoundError(
            f"AMT-GAN repo not found at {AMT_DIR} "
            "(git clone https://github.com/CGCL-codes/AMT-GAN.git amt_gan)"
        )
    sys.path.insert(0, str(AMT_DIR))
    from models import networks  # type: ignore

    G = networks.define_G().to(device)
    D = networks.define_D().to(device)
    return G, D


def load_pmgen_prior():
    sys.path.insert(0, str(STYLEGAN2_DIR))
    try:
        import legacy  # type: ignore
    except Exception:
        print("Warning: cannot import legacy from stylegan2-ada-pytorch, skip PM prior.")
        return None

    ckpts = sorted(PMGEN_CKPT_DIR.glob("pmgen_step_*.pth"))
    if not ckpts:
        print("Warning: no PMGen ckpt found, skip PM prior.")
        return None
    last = ckpts[-1]
    data = torch.load(last, map_location="cpu")
    G_ema_state = data["G_ema"]

    from day3_stylegan2_ffhq import load_generator as _loadG  # type: ignore

    G_ema = _loadG().to(device)
    G_ema.load_state_dict(G_ema_state, strict=False)
    G_ema.eval().requires_grad_(False)
    return G_ema


class IdentityArcFace(nn.Module):
    def __init__(self):
        super().__init__()
        self.arc = torch.hub.load(
            "deepinsight/insightface", "arcface_r100_v1", pretrained=True
        ).eval().to(device)
        self.resize = T.Compose(
            [T.Resize((112, 112)), T.ToTensor(), T.Normalize(0.5, 0.5)]
        )

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = (img * 0.5 + 0.5)
        outs = []
        for i in range(x.size(0)):
            pil = T.ToPILImage()(x[i].cpu())
            t = self.resize(pil).unsqueeze(0).to(device)
            emb = self.arc(t)
            emb = nn.functional.normalize(emb, dim=-1)
            outs.append(emb)
        return torch.cat(outs, dim=0)


def train_amt_pmgen():
    wandb.init(project="pmt-sm-day5-amt-pmgen-demo", config={"lr": 1e-4})

    pairs_txt = ROOT / "mt_pairs.txt"
    if not pairs_txt.exists():
        raise FileNotFoundError(
            f"{pairs_txt} not found. Create it with lines: source_rel_path,target_rel_path"
        )

    dataset = MakeupTransferDataset(pairs_txt, ROOT, img_size=256)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    G, D = load_amt_gan()
    id_net = IdentityArcFace().to(device)
    lpips_net = lpips.LPIPS(net="vgg").to(device)
    pm_prior = load_pmgen_prior()

    opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.99))
    opt_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.99))

    best_lpips = None
    (ROOT / "amt_ckpt").mkdir(exist_ok=True)
    (ROOT / "amt_demo").mkdir(exist_ok=True)

    for epoch in range(1, 3):  # demo: 2 epochs
        for it, (src, tgt) in enumerate(loader, start=1):
            src = src.to(device)
            tgt = tgt.to(device)

            # --- G forward (依 AMT-GAN 介面調整，這裡用簡化版) ---
            fake = G(src, tgt)
            cycle = G(fake, src)

            # --- D update ---
            opt_D.zero_grad(set_to_none=True)
            real_logit = D(tgt)
            fake_logit = D(fake.detach())
            loss_D = (
                nn.functional.softplus(-real_logit).mean()
                + nn.functional.softplus(fake_logit).mean()
            )
            loss_D.backward()
            opt_D.step()

            # --- G update ---
            opt_G.zero_grad(set_to_none=True)
            gan_loss = nn.functional.softplus(-D(fake)).mean()
            cycle_loss = nn.functional.l1_loss(cycle, src)

            with torch.no_grad():
                id_src = id_net(src)
            id_fake = id_net(fake)
            id_loss = 1 - (id_src * id_fake).sum(dim=-1).mean()

            lp = lpips_net(fake, tgt).mean()

            pm_loss = torch.tensor(0.0, device=device)
            if pm_prior is not None:
                with torch.no_grad():
                    pm_img = pm_prior(
                        torch.randn(src.size(0), pm_prior.z_dim, device=device),
                        torch.zeros(
                            src.size(0), pm_prior.c_dim, device=device
                        )
                        if getattr(pm_prior, "c_dim", 0) > 0
                        else None,
                    )
                pm_loss = nn.functional.mse_loss(fake, pm_img)

            total_G = (
                gan_loss + 10.0 * cycle_loss + 5.0 * id_loss + 2.0 * lp + 1.0 * pm_loss
            )
            total_G.backward()
            opt_G.step()

            wandb.log(
                {
                    "epoch": epoch,
                    "loss_D": loss_D.item(),
                    "gan_loss": gan_loss.item(),
                    "cycle_loss": cycle_loss.item(),
                    "id_loss": id_loss.item(),
                    "lpips": lp.item(),
                    "pm_loss": pm_loss.item(),
                }
            )

            if it % 10 == 0:
                print(
                    f"[E{epoch} I{it}] "
                    f"D={loss_D.item():.4f} G={gan_loss.item():.4f} "
                    f"cyc={cycle_loss.item():.4f} id={id_loss.item():.4f} lp={lp.item():.4f}"
                )

        # 每個 epoch 產生 demo GIF
        with torch.no_grad():
            demo_frames = []
            for i in range(min(8, src.size(0))):
                out = fake[i]
                img = (out.clamp(-1, 1) + 1) * 0.5
                img = img.permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype("uint8")
                demo_frames.append(img)
            gif_path = ROOT / "amt_demo" / f"epoch_{epoch:03d}_demo.gif"
            imageio.mimsave(gif_path, demo_frames, fps=2)

        cur_lpips = lp.item()
        if best_lpips is None or cur_lpips < best_lpips:
            best_lpips = cur_lpips
            torch.save(
                {"G": G.state_dict(), "D": D.state_dict()},
                ROOT / "amt_ckpt" / "best_amt_pmgen.pth",
            )
            print(f"[E{epoch}] New best LPIPS={best_lpips:.4f}, saved model.")

    print("AMT-PMGen demo training completed.")


if __name__ == "__main__":
    train_amt_pmgen()

