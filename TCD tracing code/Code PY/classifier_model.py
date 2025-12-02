import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Standard ResNet-18 1D ---

class ResNetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18_1D(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super(ResNet18_1D, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(ResNetBlock1D(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResNetBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# --- 2. Self-ONN Based Models (Simplified Implementation) ---

class SelfONN1d(nn.Module):
    """
    Self-Organizing Operational Neural Network Layer (1D).
    Approximates function using Taylor series expansion: Sum(Conv(X^q)) for q=1 to Q.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, Q=3, bias=True):
        super(SelfONN1d, self).__init__()
        self.Q = Q
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False) 
            for _ in range(Q)
        ])
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        out = 0
        for q in range(self.Q):
            # Taylor term: X^(q+1) -> q=0 is X^1, q=1 is X^2...
            # Note: Paper sometimes uses odd powers or full powers. We use full powers 1..Q.
            # Power operation should be element-wise.
            power = q + 1
            if power == 1:
                term = x
            else:
                # Use sign-preserving power for stability? Or just raw power?
                # Taylor series is on raw values.
                term = torch.pow(x, power)
            
            out += self.convs[q](term)
        
        if self.bias is not None:
            out += self.bias.view(1, -1, 1)
        return out

class SelfResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, Q=3):
        super(SelfResBlock1D, self).__init__()
        # Replace Conv1d with SelfONN1d
        self.conv1 = SelfONN1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, Q=Q, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = SelfONN1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, Q=Q, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # Shortcut can be simple Conv1d or SelfONN1d. Standard ResNet uses 1x1 Conv.
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.tanh(self.bn1(self.conv1(x))) # Paper uses Tanh for Self-ONN blocks
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.tanh(out) # Paper uses Tanh
        return out

class SelfResNet18_1D(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, Q=3):
        super(SelfResNet18_1D, self).__init__()
        self.in_channels = 64
        self.Q = Q
        
        # Initial layer
        self.conv1 = SelfONN1d(input_channels, 64, kernel_size=7, stride=2, padding=3, Q=Q, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(SelfResBlock1D(self.in_channels, out_channels, stride, Q=self.Q))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(SelfResBlock1D(out_channels, out_channels, Q=self.Q))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.tanh(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1) # Paper mentions Log-Softmax final layer

# Wrapper to choose model
def get_classifier_model(model_type='resnet18', input_length=1024, num_classes=2):
    if model_type == 'self_resnet18':
        return SelfResNet18_1D(input_channels=1, num_classes=num_classes, Q=3)
    elif model_type == 'resnet18':
        return ResNet18_1D(input_channels=1, num_classes=num_classes)
    else:
        # Fallback to simple CNN if needed or raise error
        from classifier_model import TCDClassifier # Import original simple one if kept?
        # For now return ResNet18 as default
        return ResNet18_1D(input_channels=1, num_classes=num_classes)