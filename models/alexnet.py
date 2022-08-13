import torch
import torch.nn as nn

class AlexNet(nn.Module):

    def __init_weights(layer, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, num_classes: int = 3, dropout: float = 0.5):
        super().__init__()
        self.cpu = torch.device('cpu')
        conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.features.apply(self.__init_weights)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        #Adding double the feature vectors in linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6 * 2, 4096 * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.softmax = nn.Softmax(1)

    '''
    def classify(self, batch_a_features, batch_b_features):
        all_features = torch.cat([batch_a_features, batch_b_features])
        x = self.classifier(all_features)
        return self.softmax(x)
    '''

    def forward(self, batches: torch.Tensor, batches_rgn: torch.Tensor, device):
        batches.to(device)
        batches_rgn.to(device)
        x = self.features(batches)
        x_rgn = self.features(batches_rgn)
        x = self.avgpool(x)
        x_rgn = self.avgpool(x_rgn)
        x = torch.flatten(x, 1)
        x_rgn = torch.flatten(x_rgn, 1)
        x = torch.cat([x, x_rgn], 1)
        x_rgn.to(self.cpu)
        x = self.classifier(x)
        x = self.softmax(x)
        return x