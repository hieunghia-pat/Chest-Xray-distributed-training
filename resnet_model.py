from torch import nn
from torchvision.models.resnet import resnet101

class ResnetModel(nn.Module):
    def __init__(self):
        super(ResnetModel, self).__init__()

        instance = resnet101()
        
        self.conv1 = instance.conv1
        self.bn1 = instance.bn1
        self.relu = instance.relu
        self.maxpool = instance.maxpool
        self.layer1 = instance.layer1
        self.layer2 = instance.layer2
        self.layer3 = instance.layer3
        self.layer4 = instance.layer4
        self.avgpool = instance.avgpool
        self.fc = nn.Linear(in_features=2048, out_features=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.fc(out.squeeze())

        return self.sigmoid(out)