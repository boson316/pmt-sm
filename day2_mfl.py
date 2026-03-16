import os
import math
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
import face_alignment


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------
#  ViT backbone (延續 Day1)
# -----------------------------
def load_vit_backbone():
    vit = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=0,  # 輸出 features
    ).to(device)
    vit.eval()
    return vit


def get_vit_transform():
    return create_transform(input_size=224, is_training=False, crop_pct=1.0)


def vit_forward_features_grid(vit, x):
    """
    vit.forward_features(x) -> [B, 197, 768]
    轉成：
      - patch_tok: [B, 196, 768]
      - grid: [B, 768, 14, 14]
    """
    with torch.no_grad():
        feat_tokens = vit.forward_features(x)  # [B,197,768]
    b, n, c = feat_tokens.shape
    assert n == 197, f"Expect 197 tokens, got {n}"

    patch_tok = feat_tokens[:, 1:]  # [B,196,768] 去掉 CLS
    h = w = int(math.sqrt(patch_tok.shape[1]))
    assert h * w == patch_tok.shape[1] == 196

    grid = patch_tok.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    return patch_tok, grid


# -----------------------------
#  影像載入：優先使用本地 me.*
# -----------------------------
def fetch_image_from_url(url: str) -> Image.Image:
    print(f"Downloading image from: {url}")
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def load_local_or_url_face(
    local_basename: str = "me",
) -> Image.Image:
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    cwd = os.getcwd()
    for ext in exts:
        candidate = os.path.join(cwd, local_basename + ext)
        if os.path.isfile(candidate):
            print(f"Loading local face image: {candidate}")
            return Image.open(candidate).convert("RGB")

    print(f"No local '{local_basename}*' image found, falling back to URL.")
    # 如果沒本地圖，就用一個示例 URL，若失敗再用 dummy
    url = "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/deeplab1.png"
    try:
        return fetch_image_from_url(url)
    except Exception as e:  # noqa: BLE001
        print(f"Image download failed ({e}), using dummy gray image instead.")
        img = Image.new("RGB", (224, 224), (128, 128, 128))
        return img


# -----------------------------
#  MediaPipe: 生成妝容 mask
# -----------------------------
def build_makeup_masks(img_np: np.ndarray):
    """
    使用 face-alignment(68點) 近似取代 MediaPipe：
    - 唇部：landmarks 48–67
    - 左眼：36–41
    - 右眼：42–47
    - 臉頰：選取靠兩側與下巴的一些點
    回傳：
      lip_mask, eye_mask, cheek_mask, makeup_mask, landmarks (N,2) or None
    """
    h, w = img_np.shape[:2]

    # 初始化 face-alignment（全域只建一次以節省時間）
    global _fa_model
    if "_fa_model" not in globals() or _fa_model is None:
        # 兼容不同版本的 LandmarksType 命名
        lt = getattr(face_alignment.LandmarksType, "TWO_D", None)
        if lt is None:
            lt = getattr(face_alignment.LandmarksType, "_2D", None)
        if lt is None:
            raise RuntimeError("Unsupported face_alignment.LandmarksType enum; cannot find TWO_D or _2D.")

        _fa_model = face_alignment.FaceAlignment(
            lt,
            device="cuda" if torch.cuda.is_available() else "cpu",
            flip_input=False,
        )

    preds = _fa_model.get_landmarks(img_np)
    if preds is None or len(preds) == 0:
        print("No face detected by face-alignment.")
        return None, None, None, None, None

    landmarks = preds[0]  # (68,2)

    # 唇部：外嘴唇 48-59 + 內嘴唇 60-67
    lip_idx = list(range(48, 68))
    lip_pts = landmarks[lip_idx]

    # 眼睛：左眼 36-41，右眼 42-47
    left_eye_idx = list(range(36, 42))
    right_eye_idx = list(range(42, 48))
    eye_idx = left_eye_idx + right_eye_idx
    eye_pts = landmarks[eye_idx]

    # 臉頰：用靠兩側與下巴的一些點近似 (1,2,3,4,12,13,14,15)
    cheek_idx = [1, 2, 3, 4, 12, 13, 14, 15]
    cheek_pts = landmarks[cheek_idx]

    lip_mask = np.zeros((h, w), np.uint8)
    if len(lip_pts) > 0:
        cv2.fillConvexPoly(lip_mask, np.int32(lip_pts), 255)

    eye_mask = np.zeros((h, w), np.uint8)
    if len(eye_pts) > 0:
        cv2.fillConvexPoly(eye_mask, np.int32(eye_pts), 255)

    cheek_mask = np.zeros((h, w), np.uint8)
    if len(cheek_pts) > 0:
        cv2.fillConvexPoly(cheek_mask, np.int32(cheek_pts), 255)

    makeup_mask = cv2.bitwise_or(
        cv2.bitwise_or(lip_mask, eye_mask),
        cheek_mask,
    )

    return lip_mask, eye_mask, cheek_mask, makeup_mask, landmarks


# -----------------------------
#  ViT 特徵與妝容 mask 融合
# -----------------------------
def fuse_vit_with_mask(patch_tokens: torch.Tensor, makeup_mask: np.ndarray):
    """
    patch_tokens: [1,196,768]
    makeup_mask: [H,W], 0~255
    回傳：
      - fused_scalar: 採用 mask 權重加權後的 ViT 特徵強度
      - heatmap_2d:   14x14 的 ViT 熱圖 (已正規化到 0~1)
    """
    b, n, c = patch_tokens.shape
    assert b == 1 and n == 196

    # 每個 patch 的 L2 norm 當作「注意力強度」(也可換成別的度量)
    with torch.no_grad():
        patch_norm = torch.norm(patch_tokens, dim=-1)  # [1,196]
        patch_norm = patch_norm.view(1, 14, 14)  # [1,14,14]

    heatmap = patch_norm[0].cpu().numpy()  # [14,14]
    # 正規化到 0~1
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax > hmin:
        heatmap_norm = (heatmap - hmin) / (hmax - hmin)
    else:
        heatmap_norm = np.zeros_like(heatmap)

    # 將妝容 mask 縮到 14x14，當作權重
    mask_small = cv2.resize(
        makeup_mask.astype(np.float32) / 255.0,
        (14, 14),
        interpolation=cv2.INTER_AREA,
    )
    mask_small_t = torch.from_numpy(mask_small).to(patch_tokens.device)  # [14,14]

    with torch.no_grad():
        fused = (patch_norm[0] * mask_small_t).sum() / (mask_small_t.sum() + 1e-6)
    fused_scalar = fused.item()

    return fused_scalar, heatmap_norm


# -----------------------------
#  視覺化與主流程
# -----------------------------
def visualize_and_save(
    img_np: np.ndarray,
    landmarks: np.ndarray | None,
    lip_mask: np.ndarray,
    eye_mask: np.ndarray,
    cheek_mask: np.ndarray,
    makeup_mask: np.ndarray,
    heatmap_2d: np.ndarray,
    out_path: str = "day2_mfl_demo.png",
):
    h, w = img_np.shape[:2]

    # 將 14x14 heatmap 放大到影像尺寸
    heatmap_big = cv2.resize(
        heatmap_2d.astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_CUBIC,
    )

    plt.figure(figsize=(16, 6))

    # (1) 關鍵點 + 原圖
    plt.subplot(1, 4, 1)
    plt.imshow(img_np)
    if landmarks is not None:
        plt.scatter(landmarks[:, 0], landmarks[:, 1], c="cyan", s=3)
    plt.title("Face + Landmarks")
    plt.axis("off")

    # (2) 妝容 mask
    plt.subplot(1, 4, 2)
    plt.imshow(makeup_mask, cmap="Reds")
    plt.title("Makeup Mask (lip+eye+cheek)")
    plt.axis("off")

    # (3) ViT 熱圖
    plt.subplot(1, 4, 3)
    plt.imshow(heatmap_big, cmap="viridis")
    plt.title("ViT Patch Feature Heatmap")
    plt.axis("off")

    # (4) Mask + Heatmap + 原圖疊加
    plt.subplot(1, 4, 4)
    plt.imshow(img_np)
    plt.imshow(heatmap_big, alpha=0.6, cmap="viridis")
    plt.imshow(makeup_mask, alpha=0.3, cmap="Reds")
    plt.title("Overlay: Heatmap + Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved MediaPipe+ViT MFL demo to: {out_path}")


def main():
    # 1. 載入臉部影像（優先使用 pmt-sm-day1/me.*）
    img_pil = load_local_or_url_face(local_basename="me")
    img_np = np.array(img_pil)  # [H,W,3], RGB
    h, w = img_np.shape[:2]
    print(f"Input image size: {w}x{h}")

    # 2. MediaPipe Face Mesh → 妝容 mask
    lip_mask, eye_mask, cheek_mask, makeup_mask, landmarks = build_makeup_masks(img_np)
    if makeup_mask is None:
        print("Abort: no face / landmarks for MFL.")
        return

    # 3. ViT 特徵 (224x224)
    vit = load_vit_backbone()
    transform = get_vit_transform()

    img_pil_224 = img_pil.resize((224, 224), Image.BILINEAR)
    input_tensor = transform(img_pil_224).unsqueeze(0).to(device)  # [1,3,224,224]
    print(f"Input tensor for ViT: {tuple(input_tensor.shape)}")

    patch_tokens, _ = vit_forward_features_grid(vit, input_tensor)  # [1,196,768], [1,768,14,14]

    # 4. ViT + 妝容 mask 融合
    fused_scalar, heatmap_2d = fuse_vit_with_mask(patch_tokens, makeup_mask)
    print(f"MFL fused feature scalar: {fused_scalar:.4f}")

    # 5. 視覺化輸出
    visualize_and_save(
        img_np=img_np,
        landmarks=landmarks,
        lip_mask=lip_mask,
        eye_mask=eye_mask,
        cheek_mask=cheek_mask,
        makeup_mask=makeup_mask,
        heatmap_2d=heatmap_2d,
        out_path="day2_mfl_demo.png",
    )


if __name__ == "__main__":
    main()

