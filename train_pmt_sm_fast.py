"""
超快速 PMT-SM Demo 設定

用途：
- 3 分鐘內確認整條 pipeline 正常（DataLoader / ViT+FPN / MFL / encoder）
- 只做 latent MSE，不呼叫 StyleGAN2 G.synthesis（避免 Windows 無 MSVC 問題）

使用方式：
    python train_pmt_sm_fast.py
"""

from train_pmt_sm import train_pmt_sm


if __name__ == "__main__":
    # 極速 demo：步數少、batch 小、不跑 synthesis
    train_pmt_sm(
        total_steps=50,          # 可以依需要調成 50 / 100
        batch_size=2,           # 減小一點讓每步更快
        lr=1e-4,
        img_size=224,
        num_landmarks=468,
        use_amp=True,
        use_synthesis_in_training=False,  # 只訓練 latent，避免 StyleGAN2 synthesis 相關問題
    )

