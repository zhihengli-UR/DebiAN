import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_class=10, return_feat=False):
        super(MLP, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(3 * 28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU()
        )
        self.classifier = nn.Linear(100, num_class)
        self.return_feat = return_feat

    def forward(self, x):
        x = x.view(x.size(0), -1) / 255
        feat = x = self.feature(x)
        x = self.classifier(x)

        if self.return_feat:
            return x, feat
        else:
            return x


def get_simple_classifier(arch, num_class=10, return_feat=False):
    if arch == 'mlp':
        model = MLP(num_class, return_feat=return_feat)
    else:
        raise NotImplementedError

    return model
