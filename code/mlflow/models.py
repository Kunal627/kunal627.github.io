from imports import *

class ECGClassifier(nn.Module):
    def __init__(self, sequence_length, num_classes):
        super(ECGClassifier, self).__init__()
        self.num_classes = num_classes
        self.seq_len = sequence_length
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn3   = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.bn4   = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(256 * (self.seq_len // 16), 256)  # Adjust depending on the sequence length
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = self.dropout(x)
        x = x.view(-1, 128 * (self.seq_len // 8))  # Adjust depending on the sequence length
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x