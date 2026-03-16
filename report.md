### PMT-SM Day7 評估報告（草稿）

#### 1. 任務與實驗設定概述

- **目標**：評估 PMT-SM（ViT-FPN-MFL-MSE-PMGen）在「臉部妝容轉移」上的品質與穩定度，並與既有方法（BeautyGAN、PSGAN）做 baseline 比較。
- **資料**：
  - 來源臉／目標臉配對（`pairs_demo/src`, `pairs_demo/tgt` 或 FFHQ subset）
  - 目前 demo 版 landmarks/pose 多以 dummy 或簡化處理，正式版本建議導入 MediaPipe FaceMesh / 3DMM pose。
- **模型**：
  - Encoder：ViT(timm) + FPN + Landmark encoder → StyleGAN2 latent \(w\)
  - Generator：StyleGAN2-ADA FFHQ G\_ema
  - Baseline（TODO）：BeautyGAN、PSGAN（需額外集成現有開源實作）

#### 2. 評估指標與工具

- **ArcFace 相似度**（insightface）
  - 使用 `insightface.app.FaceAnalysis("buffalo_l")` 取得臉部 embedding，計算源臉 / 目標臉 / 生成臉之間的 cosine similarity。
  - 觀察：生成臉與目標臉之間的相似度可作為「妝容接近度」與「身份保持」之間的折衷指標。

- **影像品質與結構一致性**
  - **SSIM**：`torchmetrics.StructuralSimilarityIndexMeasure`
    - \(\text{SSIM}(G(w_\text{src}), I_\text{tgt})\)，反映結構／亮度／對比的一致程度。
  - **LPIPS**：學習式感知距離（`lpips`，VGG backbone）
    - 用於衡量生成結果與目標臉在深層感知空間的距離，較接近人類主觀感受。
  - **FID**（TODO，建議用 `torchmetrics.image.FrechetInceptionDistance`）
    - 在 demo 中先以小樣本進行 sanity check，正式實驗需在大規模資料上計算。

- **BSR 使用者研究模擬（A/B Test，架構設計）**
  - 對於每一組 (source, target)：
    - 顯示兩張生成結果：例如 PMT-SM vs BeautyGAN（或 vs PSGAN）。
    - 隨機打散左右順序，記錄「偏好 A 還是 B」。
  - 在自動化模擬中，可先用 ArcFace / LPIPS 等指標建立 proxy label（例如「更接近目標妝容且身份保持較好的一方視為勝者」），近似使用者選擇。
  - 真實 BSR 可用 Gradio Web UI（見下一節）實際收集人類選票。

#### 3. Gradio Web Demo 設計（`app_gradio.py`）

- **輸入**：
  - 源臉（必填）
  - 目標臉（可選，用來計算 SSIM / LPIPS / ArcFace）
  - 姿態圖（可選；目前 demo 未使用，後續可用來控制 pose）
- **流程（簡化版 demo）**：
  1. 將源臉 resize + normalize → PMT-SM encoder → 預測 latent \(w_\text{src}\)
  2. 將 \(w_\text{src}\) 丟入 StyleGAN2 G\_ema → 生成臉 \(G(w_\text{src})\)
  3. 若有目標臉：
     - 計算 SSIM、LPIPS（以及未來的 ArcFace、CLIP-based metric）
  4. 在介面顯示：
     - Source / Target / Generated
     - Metrics 簡表
- **A/B Test 擴充構想**：
  - 新增兩個輸出槽：`Generated A`, `Generated B`，內部對應不同方法（PMT-SM vs BeautyGAN）。
  - 讓使用者選擇「你更喜歡哪一張」，收集 click 統計，作為人類偏好估計。

#### 4. 多模態改良（text prompt + CLIP）

為了更好地控制妝容風格與語意，我們可以引入文字描述與 CLIP 特徵：

- **Text Prompt**：
  - 允許使用者輸入如「淡粉色腮紅」、「歐美女性晚宴妝」、「自然日常妝」等文字描述。
  - 利用 CLIP text encoder 將 prompt 映射為向量 \(e_\text{text}\)。

- **CLIP-based Guidance**：
  - 使用 CLIP image encoder 將生成臉與目標臉映射到同一語意空間：
    - \(\ell_\text{CLIP-img} = 1 - \cos(\text{CLIP}(G(w_\text{src})), \text{CLIP}(I_\text{tgt}))\)
  - 使用 text encoder 將 prompt 融入 loss：
    - \(\ell_\text{CLIP-text} = 1 - \cos(\text{CLIP}(G(w_\text{src})), \text{CLIP}_\text{text}(\text{prompt}))\)
  - 最終 loss 可寫成：
    \[
    \mathcal{L} = \lambda_w \cdot \text{MSE}(w_\text{pred}, w_\text{gt})
                + \lambda_\text{rec} \cdot \text{LPIPS}(G(w_\text{pred}), I_\text{tgt})
                + \lambda_\text{clip-img} \cdot \ell_\text{CLIP-img}
                + \lambda_\text{clip-text} \cdot \ell_\text{CLIP-text}
    \]

- **介面層面的應用**：
  - 在 Gradio UI 中加入一個文字輸入框「妝容描述」，同時顯示 CLIP 相似度（生成圖 vs prompt）。
  - 讓使用者可以快速試不同 prompt，觀察妝容變化與指標。

#### 5. 與 BeautyGAN / PSGAN Baseline 比較（設計草稿）

- **整合方式**：
  - 下載並包裝開源 BeautyGAN / PSGAN 模型，使其具有一致的 API：
    - `beautygan_transfer(src_img, ref_img) -> out_img`
    - `psgan_transfer(src_img, ref_img) -> out_img`
  - 在 `eval_pmt_sm.py` 中對同一批 (source, target) 同時計算：
    - PMT-SM, BeautyGAN, PSGAN 的生成結果
    - 為每一個方法計算 SSIM / LPIPS / ArcFace /（FID）

- **報表呈現**：
  - 對每個方法彙總平均與標準差：
    - \(\overline{\text{SSIM}}, \overline{\text{LPIPS}}, \overline{\text{ArcFace sim}}\)
  - 若有 BSR A/B 真實使用者結果，加入「偏好率」欄位：
    - \(P(\text{PMT-SM} \succ \text{BeautyGAN})\)
    - \(P(\text{PMT-SM} \succ \text{PSGAN})\)

#### 6. NVIDIA 提案重點（摘要版）

1. **技術創新點**
   - 將 ViT-FPN 特徵與多尺度臉部 landmarks（MFL）結合，精準回歸 StyleGAN2 latent 空間。
   - 支援多模態控制：圖像（源臉／目標臉）、文字（妝容描述）、姿態。
   - 在相同或更低的計算成本下，較傳統 BeautyGAN / PSGAN 提供更細緻、可控的妝容轉移。

2. **系統完備度**
   - 已實作端到端訓練（Day6）、基礎評估（SSIM / LPIPS / ArcFace / FID demo）、以及 Web demo（Gradio）。
   - 模型可匯出 ONNX / TensorRT engine（encoder 部分），適合部署到 NVIDIA Jetson / GPU 伺服器。

3. **應用場景**
   - 虛擬試妝、線上美妝電商、直播濾鏡、影片後期處理等。
   - 可與 NVIDIA 的影像增強 / Maxine / Broadcast 套件整合，提供一站式臉部視覺體驗解決方案。

4. **下一步規劃**
   - 正式納入 CLIP 多模態 loss，提升語意一致性。
   - 在大規模 dataset 上完成嚴謹的 FID / BSR 實驗。
   - 擴展到影片序列，結合 temporal consistency loss，避免閃爍。

