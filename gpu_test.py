import torch


def main():
    print("PyTorch version:", torch.__version__)

    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)

    if cuda_available:
        device_count = torch.cuda.device_count()
        print("CUDA device count:", device_count)

        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            print(f"Device {i}: {name}, capability: {cap}")

        x = torch.randn(3, 3, device="cuda")
        print("Tensor on CUDA:", x)
    else:
        print("No CUDA device detected, running simple CPU test.")
        x = torch.randn(3, 3)
        print("Tensor on CPU:", x)


if __name__ == "__main__":
    main()

