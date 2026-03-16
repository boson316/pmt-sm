import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

from train_pmt_sm import PMTSMNet, load_stylegan2, PRETRAINED_FFHQ, ROOT, device

def main():
    # 1) 載入 StyleGAN2 G 與 PMT-SM encoder 結構
    G = load_stylegan2(PRETRAINED_FFHQ)
    model = PMTSMNet(
        stylegan_G=G,
        num_landmarks=468,
        fpn_out_ch=256,
        vit_freeze=True,
    ).to(device)

    # 2) 載入剛剛訓練的 checkpoint（step50）
    ckpt_path = ROOT / "pmt_sm_ckpt" / "pmt_sm_step50.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # 3) 準備一張臉圖（用專案目錄下的 me.png）
    img_path = ROOT / "me.png"
    if not img_path.exists():
        raise FileNotFoundError(f"{img_path} not found, 請放一張 me.png 到專案目錄。")

    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    img_t = transform(img).unsqueeze(0).to(device)  # [1,3,224,224]

    # 4) 先用 dummy landmarks（全 0），只是測試 pipeline
    num_landmarks = 468
    lm_t = torch.zeros(1, num_landmarks, 2, device=device)

    with torch.no_grad():
        pred_w, _, _ = model(img_t, lm_t, w_gt=None, use_decode=False)

    print("pred_w shape:", tuple(pred_w.shape))
    save_path = ROOT / "pmt_sm_pred_w_step50.pt"
    torch.save({"pred_w": pred_w.cpu()}, save_path)
    print(f"Saved predicted w to: {save_path}")

if __name__ == "__main__":
    main()
