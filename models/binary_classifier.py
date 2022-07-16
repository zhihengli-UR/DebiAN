import torch.nn as nn
from torchvision.models import resnet50, resnet18


class ReturnFeatureLinear(nn.Linear):
    def forward(self, feature):
        fc_out = super().forward(feature)
        return fc_out, feature


def get_resnet18_classifier(num_classes=10, return_feature=False):
    model = resnet18(pretrained=True)
    if not return_feature:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.fc = ReturnFeatureLinear(model.fc.in_features, num_classes)
    return model


def get_resnet50_classifier(num_classes=10, return_feature=False):
    model = resnet50(pretrained=True)
    if not return_feature:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.fc = ReturnFeatureLinear(model.fc.in_features, num_classes)
    return model
