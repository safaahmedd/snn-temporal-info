import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate


def lif():
    # tau=2.0 with decay_input=False gives v = 0.5 * v + x (decay 0.5)
    return neuron.LIFNode(tau=2.0, decay_input=False,
                          v_threshold=1.0, v_reset=0.0,
                          surrogate_function=surrogate.ATan(),
                          detach_reset=True, step_mode='m')


def conv_block(cin, cout):
    return [layer.SeqToANNContainer(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout)),
            lif()]


class VGGSNN(nn.Module):
    def __init__(self, num_classes=11, in_ch=2, size=48, drop=0.25):
        super().__init__()
        cfg = [64, 128, 'A', 256, 256, 'A', 512, 512, 'A', 512, 512, 'A']
        layers, c = [], in_ch
        for v in cfg:
            if v == 'A':
                layers.append(layer.SeqToANNContainer(nn.AvgPool2d(2)))
            else:
                layers += conv_block(c, v)
                c = v
        self.features = nn.Sequential(*layers)
        s = size // 16                      # four 2x2 poolings: 48 -> 3
        self.classifier = layer.SeqToANNContainer(
            nn.Flatten(), nn.Dropout(drop), nn.Linear(c * s * s, num_classes))

    def forward(self, x):                   # x: [N, T, C, H, W]
        x = x.transpose(0, 1)               # -> [T, N, C, H, W]
        x = self.features(x)
        out = self.classifier(x)            # [T, N, classes]
        return out.mean(0)                  # average over time -> [N, classes]