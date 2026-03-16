import math
import os
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import timm
import torch
import torch.nn as nn
from PIL import Image
from timm.data import create_transform


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ViTFPN(nn.Module):
    """
    FPN over ViT patch features.
    我們從 ViT 的 patch token 產生 3 個尺度的特徵圖：
    - 14x14, C=768
    - 7x7,  C=384
    - 4x4,  C=192
    然後用 Conv1x1 側向連接 + 上採樣融合。
    """

    def __init__(self, channels=(768, 384, 192), out_ch=256):
        super().__init__()
        self.out_ch = out_ch

        # 將不同通道數的尺度轉成同一個 out_ch
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(c, out_ch, kernel_size=1) for c in channels]
        )
        # FPN 後再做平滑
        self.smooth_convs = nn.ModuleList(
            [nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1) for _ in channels]
        )

    def forward(self, feats):
        """
        feats: list of [B, C, H, W]，由小到大尺度，例如：
        [P5 (最小), P4, P3 (最大)]
        """
        assert len(feats) == 3, "Expect 3 feature maps for FPN."

        # 先做 lateral conv
        laterals = [l(f) for l, f in zip(self.lateral_convs, feats)]

        # 自上而下融合：P5 -> P4 -> P3
        # 假設 feats[0] 最小、feats[2] 最大
        p5 = laterals[0]
        p4 = laterals[1] + nn.functional.interpolate(
            p5, size=laterals[1].shape[-2:], mode="bilinear", align_corners=False
        )
        p3 = laterals[2] + nn.functional.interpolate(
            p4, size=laterals[2].shape[-2:], mode="bilinear", align_corners=False
        )

        # 平滑
        p5 = self.smooth_convs[0](p5)
        p4 = self.smooth_convs[1](p4)
        p3 = self.smooth_convs[2](p3)

        # 這裡回傳最高解析度的 p3 作為 fused feature
        return p3


def load_vit_backbone():
    # num_classes=0 代表輸出 feature 而不是分類 logits
    vit = timm.create_model(
        "vit_base_patch16_224", pretrained=True, num_classes=0
    ).to(device)
    vit.eval()
    return vit


def load_vit_classifier():
    # 另一個帶分類頭的 ViT，用於 top-5 分類 (ImageNet)
    vit_cls = timm.create_model("vit_base_patch16_224", pretrained=True).to(device)
    vit_cls.eval()
    return vit_cls


def get_transform():
    # 與你提供的一致：input_size=224, 非訓練模式
    transform = create_transform(input_size=224, is_training=False, crop_pct=1.0)
    return transform


def fetch_image_from_url(url: str, size=(224, 224)) -> Image.Image:
    """
    從 URL 抓圖；如果失敗（例如 404 或無網路），回傳一張灰色 dummy 圖，
    讓整個 ViT+FPN pipeline 仍可順利跑完。
    """
    try:
        print(f"Downloading image from: {url}")
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        if size is not None:
            img = img.resize(size, Image.BILINEAR)
        return img
    except Exception as e:  # noqa: BLE001
        print(f"Image download failed ({e}), using dummy gray image instead.")
        # 建立一張簡單的灰底圖片（中間畫一個白色圓形，方便視覺化）
        w, h = size
        img = Image.new("RGB", (w, h), (128, 128, 128))
        # 畫一個白色圓形區域（可視作「臉」）
        cx, cy = w // 2, h // 2
        radius = min(w, h) // 4
        arr = np.array(img)
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        arr[mask] = (220, 220, 220)
        return Image.fromarray(arr)


def load_local_or_url_face(
    local_basename: str = "me",
    size: tuple[int, int] = (224, 224),
) -> Image.Image:
    """
    優先嘗試讀取當前資料夾下名稱開頭為 local_basename 的圖片 (例如 me.jpg / me.png)，
    若不存在則退回 URL / dummy 圖。
    """
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    cwd = os.getcwd()
    for ext in exts:
        candidate = os.path.join(cwd, local_basename + ext)
        if os.path.isfile(candidate):
            print(f"Loading local face image: {candidate}")
            img = Image.open(candidate).convert("RGB")
            if size is not None:
                img = img.resize(size, Image.BILINEAR)
            return img

    print(f"No local '{local_basename}*' image found, falling back to URL/dummy.")
    url = "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/deeplab1.png"
    return fetch_image_from_url(url, size=size)


def vit_forward_features_grid(vit, x):
    """
    用 ViT 抽取 patch 特徵並轉成 2D grid。
    vit.forward_features(x) -> [B, num_tokens, C]，其中 num_tokens = 1 + 14*14
    回傳：
      - feat_grid: [B, C, 14, 14] (不含 CLS token)
      - feat_tokens: [B, 197, 768]
    """
    with torch.no_grad():
        feat_tokens = vit.forward_features(x)  # [B, 197, 768]
    b, n, c = feat_tokens.shape
    assert n == 197, f"Expect 197 tokens (1+14*14), got {n}"

    cls_tok = feat_tokens[:, 0:1]  # [B,1,C]，這裡如果要用可以額外處理
    patch_tok = feat_tokens[:, 1:]  # [B,196,C]

    h = w = int(math.sqrt(patch_tok.shape[1]))
    assert h * w == patch_tok.shape[1] == 196

    feat_grid = patch_tok.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    return feat_grid, feat_tokens


def build_multi_scale_from_vit_grid(feat_grid, feat_tokens):
    """
    從 ViT 的 14x14 feature grid 構造 3 個尺度：
    - P3: 14x14, C=768
    - P4:  7x7, C=384 (由 768 經 1x1 conv 降維後，再 avg pool)
    - P5:  4x4, C=192 (再降維 + 更大的 pool)
    """
    b, c, h, w = feat_grid.shape  # [B,768,14,14]
    assert (c, h, w) == (768, 14, 14), "Expect ViT base patch16 grid of [B,768,14,14]"

    # P3: 原生解析度 14x14, C=768
    p3 = feat_grid

    # 先用 1x1 conv 把通道降到 384 / 192
    conv_768_to_384 = nn.Conv2d(768, 384, kernel_size=1).to(device)
    conv_768_to_192 = nn.Conv2d(768, 192, kernel_size=1).to(device)

    with torch.no_grad():
        p3_384 = conv_768_to_384(feat_grid)  # [B,384,14,14]
        p3_192 = conv_768_to_192(feat_grid)  # [B,192,14,14]

        # P4: 7x7, C=384
        p4 = nn.functional.avg_pool2d(p3_384, kernel_size=2, stride=2)  # [B,384,7,7]

        # P5: 4x4, C=192 (14 -> 4 可以用自適應池化)
        p5 = nn.functional.adaptive_avg_pool2d(p3_192, output_size=(4, 4))  # [B,192,4,4]

    # 回傳由小到大尺度: [P5, P4, P3]
    return [p5, p4, p3]


def imagenet_top5(vit_cls, input_tensor):
    """
    對輸入圖片做 ImageNet top-5 分類。
    為了方便標籤，從 GitHub 下載 imagenet_classes.txt。
    """
    with torch.no_grad():
        logits = vit_cls(input_tensor)  # [B,1000]
        probs = torch.softmax(logits, dim=-1)[0]

    top5_prob, top5_idx = torch.topk(probs, 5)

    try:
        cls_txt_url = (
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        )
        cls_resp = requests.get(cls_txt_url, timeout=10)
        cls_resp.raise_for_status()
        classes = cls_resp.text.strip().splitlines()
        if len(classes) != 1000:
            raise ValueError("Unexpected number of ImageNet classes.")
    except Exception as e:  # noqa: BLE001
        print(f"Warning: failed to load ImageNet class names, use indices only. ({e})")
        classes = [str(i) for i in range(1000)]

    print("\nTop-5 ImageNet predictions:")
    for p, idx in zip(top5_prob.tolist(), top5_idx.tolist()):
        print(f"  {p*100:5.2f}% - [{idx:4d}] {classes[idx]}")


def visualize_and_save(img_pil, fpn_feat, out_path="day1_vit_fpn_demo.png"):
    """
    可視化輸入臉照 + FPN 融合後的特徵圖 (取第一個通道)。
    """
    img_np = np.array(img_pil)  # [H,W,3], uint8

    feat = fpn_feat[0, 0].detach().cpu().numpy()  # 取 batch=0, channel=0
    # 正規化到 0-1
    feat_min, feat_max = feat.min(), feat.max()
    if feat_max > feat_min:
        feat_norm = (feat - feat_min) / (feat_max - feat_min)
    else:
        feat_norm = np.zeros_like(feat)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title("Input Face")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(feat_norm, cmap="viridis")
    plt.title("ViT+FPN Feature Map")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nFeature map saved to: {out_path}")


def main():
    # 1. 準備模型與 transform
    vit_backbone = load_vit_backbone()
    vit_classifier = load_vit_classifier()
    transform = get_transform()

    # 2. 讀取臉部圖片：
    #    - 如果當前資料夾下有 me.jpg / me.png ...，優先使用你的照片
    #    - 否則才會嘗試從 URL 抓圖，最後退回 dummy 圖
    img_pil = load_local_or_url_face(local_basename="me", size=(224, 224))

    # 3. 前處理
    input_tensor = transform(img_pil).unsqueeze(0).to(device)  # [1,3,224,224]
    print(f"Input tensor shape: {input_tensor.shape}")

    # 4. ViT forward_features -> 14x14 patch grid
    feat_grid, feat_tokens = vit_forward_features_grid(vit_backbone, input_tensor)
    print(f"ViT patch grid shape: {feat_grid.shape}")  # [1,768,14,14]
    print(f"ViT token feature shape: {feat_tokens.shape}")  # [1,197,768]

    # 5. 構造 3 尺度特徵 (768/384/192)
    multi_scale_feats = build_multi_scale_from_vit_grid(feat_grid, feat_tokens)
    print("Multi-scale shapes:")
    for i, f in enumerate(multi_scale_feats):
        print(f"  Level {i}: {tuple(f.shape)}")

    # 6. FPN 融合
    fpn = ViTFPN(channels=(192, 384, 768), out_ch=256).to(device)
    with torch.no_grad():
        fpn_feat = fpn([f.to(device) for f in multi_scale_feats])
    print(f"FPN fused feature shape: {fpn_feat.shape}")  # [1,256,14,14]

    # 7. top-5 ImageNet 分類 (臉部圖片，但是 label 來自 ImageNet)
    imagenet_top5(vit_classifier, input_tensor)

    # 8. 可視化與儲存特徵圖
    visualize_and_save(img_pil, fpn_feat, out_path="day1_vit_fpn_demo.png")


if __name__ == "__main__":
    main()

