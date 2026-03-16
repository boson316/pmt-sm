# PMT-SM Architecture: ViT-FPN + MFL 深度解析

## 1. 總覽

PMT-SM 將臉部妝容轉移拆成三階段：

1. **Encoder**：ViT-FPN + Multi-Scale Facial Landmarks (MFL) → 預測 StyleGAN2 的 latent `w`
2. **Generator**：StyleGAN2-ADA FFHQ，以 `w` 解碼成臉部圖像
3. **Deployment**：ONNX / TensorRT 匯出 Encoder，達成低延遲推論

## 2. ViT-FPN Backbone

- **ViT**：使用 `timm` 的 Vision Transformer，從 224×224 輸入得到 patch 特徵 (e.g. 14×14×768)。
- **FPN**：在 patch grid 上做多尺度特徵金字塔 (P3/P4/P5)，經 1×1 conv 對齊通道後自上而下融合，輸出單一高解析度 feature map。
- **用途**：提供全域與多尺度的臉部語意特徵，供後續與 landmark 特徵融合。

## 3. MFL (Multi-Scale Facial Landmarks)

- **輸入**：68 或 468 點臉部關鍵點 (MediaPipe / dlib 等)。
- **編碼**：多尺度 embedding（對應不同臉部區域：眼、唇、輪廓等），與 ViT-FPN 特徵 concat 或相加。
- **輸出**：與 backbone 特徵一併送入 latent regression head。

## 4. Latent Regression Head

- 將融合後的特徵經 MLP / 輕量 Conv 映射到 StyleGAN2 的 `w` 空間 (通常 512 維 × num_ws)。
- 訓練目標：MSE(pred_w, w_gt)，可選加 perceptual / identity loss。

## 5. StyleGAN2-ADA Generator

- 使用 NVLabs 的 `stylegan2-ada-pytorch` 與 FFHQ 預訓練權重。
- 訓練時可選「僅 latent MSE」或「latent + 解碼圖像 loss」，後者需在支援 CUDA custom ops 的環境編譯。

## 6. 與 TensorRT / Jetson 的對接

- Encoder 匯出 ONNX 後，可用 `torch2trt` 或 NVIDIA 工具鏈轉成 TensorRT FP16 engine。
- 目標延遲：單張 512×512 臉部 < 50 ms，適用 real-time AR / 美妝鏡。
