"""
Day7 PMT-SM Gradio Web Demo（簡化版）

功能：
- 上傳：源臉、目標臉、（可選）姿態圖
- 透過已訓練的 PMT-SM encoder + StyleGAN2 G 產生妝容轉移結果（目前簡化為：源臉 → w → G(w)）
- 顯示源圖 / 目標圖 / 生成圖
- 計算簡單 metrics：SSIM、LPIPS（若安裝）

TODO（後續可補強）：
- 真正的「妝容轉移」邏輯（例如用 PMGen 或額外 conditioning）
- ArcFace 相似度顯示
- BeautyGAN / PSGAN baseline 結果並列
"""

from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T

import gradio as gr

from train_pmt_sm import PMTSMNet, load_stylegan2, PRETRAINED_FFHQ, ROOT, device


def load_models():
    print("[Gradio] Initializing models...")
    G = load_stylegan2(PRETRAINED_FFHQ)
    model = PMTSMNet(
        stylegan_G=G,
        num_landmarks=468,
        fpn_out_ch=256,
        vit_freeze=True,
    ).to(device)

    ckpt_path = ROOT / "pmt_sm_ckpt" / "pmt_sm_step50.pt"
    if ckpt_path.exists():
        print(f"[Gradio] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        print(f"[Gradio] WARNING: checkpoint not found at {ckpt_path}, using random weights.")

    model.eval()

    # metrics（選用）
    try:
        import torchmetrics

        ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(
            device
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Gradio] torchmetrics not available: {e}")
        ssim_metric = None

    try:
        import lpips

        lpips_metric = lpips.LPIPS(net="vgg").to(device)
    except Exception as e:  # noqa: BLE001
        print(f"[Gradio] LPIPS not available: {e}")
        lpips_metric = None

    return model, ssim_metric, lpips_metric


MODEL, SSIM_METRIC, LPIPS_METRIC = load_models()


TRANSFORM = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def run_pmt_sm(
    src_img: Image.Image,
    tgt_img: Image.Image | None,
    pose_img: Image.Image | None,
):
    if src_img is None:
        return None, None, "請先上傳源臉圖片。"

    # 1) 預處理
    src_t = TRANSFORM(src_img).unsqueeze(0).to(device)
    if tgt_img is not None:
        tgt_t = TRANSFORM(tgt_img).unsqueeze(0).to(device)
    else:
        tgt_t = None

    # TODO: 這裡 landmarks 先用 dummy，之後可從 MediaPipe / pose_img 補上
    num_landmarks = 468
    lm = torch.zeros(1, num_landmarks, 2, device=device)

    with torch.no_grad():
        pred_w, _, _ = MODEL(src_t, lm, w_gt=None, use_decode=False)
        recon = MODEL.decode(pred_w).clamp(0, 1)

    # 轉回 PIL 方便顯示
    def to_pil(t: torch.Tensor) -> Image.Image:
        x = t[0].permute(1, 2, 0).detach().cpu().numpy()
        x = (x * 255.0).clip(0, 255).astype("uint8")
        return Image.fromarray(x)

    recon_pil = to_pil(recon)

    # 2) metrics（簡易版）
    metrics_txt = []
    if tgt_t is not None and SSIM_METRIC is not None:
        with torch.no_grad():
            # 尺度對齊
            r = F.interpolate(recon, size=tgt_t.shape[-2:], mode="area")
            ssim_val = SSIM_METRIC(r, tgt_t).item()
        metrics_txt.append(f"SSIM={ssim_val:.3f}")

    if tgt_t is not None and LPIPS_METRIC is not None:
        with torch.no_grad():
            lp = LPIPS_METRIC(
                (recon * 2 - 1), (tgt_t * 2 - 1)
            ).item()  # LPIPS 期待 [-1,1]
        metrics_txt.append(f"LPIPS={lp:.3f}")

    if not metrics_txt:
        metrics_txt.append("（未安裝 torchmetrics / LPIPS，因此略過 SSIM/LPIPS）")

    return recon_pil, "\n".join(metrics_txt), "完成"


def main():
    with gr.Blocks(title="PMT-SM Makeup Transfer Demo") as demo:
        gr.Markdown(
            """
            # PMT-SM 妝容轉移 Demo（簡化版）

            - 上傳一張 **源臉**（必填）
            - 可選：上傳一張 **目標臉**（用來計算 SSIM / LPIPS）
            - （之後可加姿態圖 / landmarks / text prompt）
            """
        )

        with gr.Row():
            src_in = gr.Image(label="源臉（Source Face）", type="pil")
            tgt_in = gr.Image(label="目標臉（Target Face，可選）", type="pil")
            pose_in = gr.Image(label="姿態圖（Pose，可選，暫未使用）", type="pil")

        run_btn = gr.Button("執行 PMT-SM")

        with gr.Row():
            out_img = gr.Image(label="生成結果（Transferred）")
            out_metrics = gr.Textbox(label="Metrics", lines=4)
            out_status = gr.Textbox(label="狀態")

        run_btn.click(
            fn=run_pmt_sm,
            inputs=[src_in, tgt_in, pose_in],
            outputs=[out_img, out_metrics, out_status],
        )

    demo.launch()


if __name__ == "__main__":
    main()

