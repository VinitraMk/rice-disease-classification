import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.5):
        super().__init__()
        self.cpu = torch.device('cpu')
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
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.softmax = nn.Softmax(1)
    
    def __get_image_partitions(self, partitions, device):
        partition_features = []
        no_of_partitions = partitions.shape[0]
        feature_sum = 0
        for i in range(no_of_partitions):
            x = partitions[i].unsqueeze(0).to(device)
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x.to(self.cpu)
            if i == 0:
                feature_sum = x
            else:
                feature_sum = torch.add(feature_sum, x)
        return feature_sum
    
    def __iterate_through_batch(self, batches, device):
        batch_results = []
        for image_partitions in batches:
            feature_sum = self.__get_image_partitions(image_partitions, device)
            feature_sum.to(device)
            x = self.classifier(feature_sum)
            x = self.softmax(x)
            batch_results.append(x)
        return batch_results

    def forward(self, batches: torch.Tensor, device):
        batches.to(device)
        x = self.features(batches)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.softmax(x)
        #x = self.features(x)
        #x = self.avgpool(x)
        #print('features', x.shape)
        #x = torch.flatten(x, 1)
        #print('flatten features', x.shape)
        #x = self.classifier(x)
        return x