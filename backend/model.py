import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=6, hidden_size=256):
        super(CNNLSTM, self).__init__()

        # CNN 부분
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # LSTM 부분
        self.lstm = nn.LSTM(512, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)

        # 분류기
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),  # bidirectional이므로 * 2
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (batch, time, channel, height, width)
        batch_size, time_steps, C, H, W = x.size()

        # CNN 특징 추출
        features = []
        for t in range(time_steps):
            frame = x[:, t, :, :, :]  # (batch, C, H, W)
            frame_features = self.cnn(frame)  # (batch, 512, 1, 1)
            frame_features = frame_features.view(batch_size, -1)  # (batch, 512)
            features.append(frame_features)

        features = torch.stack(features, dim=1)  # (batch, time, 512)

        # LSTM
        lstm_out, _ = self.lstm(features)  # (batch, time, hidden_size * 2)
        lstm_out = self.dropout(lstm_out)

        # 마지막 시간 스텝 사용
        output = lstm_out[:, -1, :]  # (batch, hidden_size * 2)

        # 분류
        output = self.classifier(output)  # (batch, num_classes)
        return output