import torch
import torch.nn as nn

class Feature_Extractor(nn.Module):
    def __init__(self, conv_configs, dropout: float = 0.0):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        for i, (out_channels, kernel_size, stride) in enumerate(conv_configs):
            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=0)
            norm_layer = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels)
            activation = nn.GELU()
            self.conv_layers.append(nn.Sequential(conv_layer, norm_layer, activation, nn.Dropout(dropout)))
            in_channels = out_channels
        
        self.conv_configs = conv_configs

    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size).float() / stride).long() + 1
        for conv in self.conv_configs:
            input_lengths = _conv_out_length(input_lengths, conv[1], conv[2])
        return input_lengths
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        for layer in self.conv_layers:
            x = layer(x)
        new_lengths = self._get_feat_extract_output_lengths(lengths.clone())
        return x, new_lengths

