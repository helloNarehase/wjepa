def main():
    import torch
    from models import WaveEncode

    conv_configs = [
        (512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), 
        (512, 3, 2), (512, 2, 2), (32, 2, 2)
    ]

    x_input = torch.randn(2, 6000)
    input_lengths = torch.tensor([6000, 3000])

    we = WaveEncode(conv_configs, 3, 32, 4, 4.0)

    features, new_lengths = we(x_input, input_lengths)
    print(f"{features.shape=}", f"{new_lengths=}",)

if __name__ == "__main__":
    main()