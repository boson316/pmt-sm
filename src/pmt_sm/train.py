"""
PMT-SM training entry point (modular).

For the full pipeline (ViT-FPN + MFL + StyleGAN2 latent regression),
run from project root:

    python train_pmt_sm.py       # full training
    python train_pmt_sm_fast.py  # fast demo (~3 min)

This module is reserved for future refactored training API.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    print("Use project root scripts: train_pmt_sm.py or train_pmt_sm_fast.py")
    print("Root:", ROOT)


if __name__ == "__main__":
    main()
