# PMT-SM 消融實驗 (Ablation Study)

## 設計目的

釐清各模組對最終指標（ArcFace、SSIM、FID、推論延遲）的貢獻，供論文與 Repo 優化參考。

## 建議對照組

| 設定 | ViT-FPN | MFL | Latent Head | G.synthesis in training | 說明 |
|------|:------:|:---:|:-----------:|:-----------------------:|------|
| Full (PMT-SM) | ✓ | ✓ | ✓ | 可選 | 完整模型 |
| w/o MFL | ✓ | ✗ | ✓ | 同左 | 僅 ViT-FPN → w |
| w/o FPN | ViT only | ✓ | ✓ | 同左 | 單尺度 ViT 特徵 |
| w/o ViT (CNN backbone) | ✗ (e.g. ResNet) | ✓ | ✓ | 同左 | 替換 backbone |
| Latent-only training | ✓ | ✓ | ✓ | ✗ | 僅 w MSE，不跑 G.synthesis |

## 建議指標

- **ArcFace**：身份保持
- **SSIM**：結構與內容保留
- **FID**：生成品質
- **BSR**：A/B 美妝偏好
- **Inference time**：Encoder-only (ONNX/TensorRT) vs Full (Encoder+Generator)

## 紀錄格式

可於 `report.md` 或本檔以表格紀錄各設定之數值，例如：

```text
| Config        | ArcFace | SSIM | FID  | Inference (ms) |
|---------------|---------|------|------|-----------------|
| Full          | 0.94    | 0.95 | 7.93 | ~45 (Encoder)   |
| w/o MFL       | TBD     | TBD  | TBD  | TBD             |
| Latent-only   | TBD     | TBD  | TBD  | TBD             |
```

（TBD：待實測後填寫）
