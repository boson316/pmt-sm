import argparse
from pathlib import Path

import torch
import torch.nn as nn

from train_pmt_sm import (
    ROOT,
    PRETRAINED_FFHQ,
    STYLEGAN2_DIR,
    PMTSMNet,
    load_stylegan2,
)


"""
TensorRT export script for PMT-SM

Flow:
  1) 構建與訓練時一致的 PMTSMNet 結構
  2) 載入訓練好的 checkpoint
  3) 匯出 ONNX
  4) 若安裝 torch2trt，則額外轉成 TensorRT engine (Jetson-ready)

注意：
  - Jetson 上實際執行時，建議在裝置端重跑一次 torch2trt 轉換，確保與當地
    TensorRT / CUDA / cuDNN 版本相容。
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[export_trt] Using device: {device}")


def build_model_for_export(num_landmarks: int = 468) -> nn.Module:
    print("[export_trt] Loading StyleGAN2-ADA FFHQ generator...")
    G = load_stylegan2(PRETRAINED_FFHQ)

    print("[export_trt] Building PMTSMNet for export...")
    model = PMTSMNet(
        stylegan_G=G,
        num_landmarks=num_landmarks,
        fpn_out_ch=256,
        vit_freeze=True,
    )

    # 匯出時不需要 StyleGAN 參數可訓練，只為 forward 使用
    for p in model.G.parameters():
        p.requires_grad_(False)

    return model.to(device).eval()


def load_checkpoint(model: nn.Module, ckpt_path: Path):
    print(f"[export_trt] Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=True)
    return model


def export_onnx(
    model: nn.Module,
    save_path: Path,
    img_size: int = 224,
    num_landmarks: int = 468,
):
    print(f"[export_trt] Exporting ONNX to: {save_path}")

    dummy_img = torch.randn(1, 3, img_size, img_size, device=device)
    dummy_lm = torch.zeros(1, num_landmarks, 2, device=device)

    # 我們只匯出 encode 部分 (image+landmarks -> w)，decoder/StyleGAN2 在 server 端或離線環境使用
    class EncoderWrapper(nn.Module):
        def __init__(self, net: PMTSMNet):
            super().__init__()
            self.net = net

        def forward(self, x, lm):
            w = self.net.encode(x, lm)  # [B,w_dim]
            return w

    wrapper = EncoderWrapper(model).to(device).eval()

    torch.onnx.export(
        wrapper,
        (dummy_img, dummy_lm),
        save_path.as_posix(),
        input_names=["image", "landmarks"],
        output_names=["w"],
        dynamic_axes={
            "image": {0: "batch"},
            "landmarks": {0: "batch"},
            "w": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print("[export_trt] ONNX export done.")


def export_trt(
    onnx_path: Path,
    trt_path: Path,
    img_size: int = 224,
    num_landmarks: int = 468,
):
    """
    使用 torch2trt 做簡單 TensorRT 匯出。
    注意：torch2trt 不一定依賴 ONNX；這裡只是對齊你「ONNX -> engine」的流程概念。
    """
    try:
        from torch2trt import torch2trt
    except Exception as e:  # noqa: BLE001
        print(
            f"[export_trt] torch2trt not available ({e}). "
            "Skip TensorRT engine export; you can run it on Jetson instead."
        )
        return

    print("[export_trt] Building model for TensorRT conversion...")
    # 這裡再次建立與 ONNX 相同的 EncoderWrapper
    G = load_stylegan2(PRETRAINED_FFHQ)
    model = PMTSMNet(
        stylegan_G=G,
        num_landmarks=num_landmarks,
        fpn_out_ch=256,
        vit_freeze=True,
    ).to(device).eval()

    class EncoderWrapper(nn.Module):
        def __init__(self, net: PMTSMNet):
            super().__init__()
            self.net = net

        def forward(self, x, lm):
            return self.net.encode(x, lm)

    wrapper = EncoderWrapper(model).to(device).eval()

    dummy_img = torch.randn(1, 3, img_size, img_size, device=device)
    dummy_lm = torch.zeros(1, num_landmarks, 2, device=device)

    print("[export_trt] Converting to TensorRT engine with torch2trt...")
    trt_model = torch2trt(
        wrapper,
        [dummy_img, dummy_lm],
        fp16_mode=True,
        max_batch_size=4,
    )

    print(f"[export_trt] Saving TensorRT engine weights to: {trt_path}")
    torch.save(trt_model.state_dict(), trt_path)
    print("[export_trt] TensorRT export done.")


def main():
    parser = argparse.ArgumentParser(description="Export PMT-SM encoder to ONNX/TensorRT.")
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to PMT-SM checkpoint (.pt) from train_pmt_sm.py",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT / "export"),
        help="Directory to save ONNX / TRT files.",
    )
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_landmarks", type=int, default=468)
    parser.add_argument("--no_trt", action="store_true", help="Only export ONNX, skip TensorRT.")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = out_dir / "pmt_sm_encoder.onnx"
    trt_path = out_dir / "pmt_sm_encoder_trt.pth"

    model = build_model_for_export(num_landmarks=args.num_landmarks)
    model = load_checkpoint(model, ckpt_path)

    export_onnx(
        model,
        save_path=onnx_path,
        img_size=args.img_size,
        num_landmarks=args.num_landmarks,
    )

    if not args.no_trt:
        export_trt(
            onnx_path=onnx_path,
            trt_path=trt_path,
            img_size=args.img_size,
            num_landmarks=args.num_landmarks,
        )


if __name__ == "__main__":
    main()

