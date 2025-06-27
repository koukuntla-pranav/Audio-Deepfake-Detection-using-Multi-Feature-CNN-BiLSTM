import torch.nn as nn
import torch


class FeatureCNN(nn.Module):
    def __init__(self, in_channels=1):
        super(FeatureCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.cnn(x)




class MultiFeatureCNN_BiLSTM(nn.Module):
    def __init__(self):
        super(MultiFeatureCNN_BiLSTM, self).__init__()

        # 4 parallel CNNs for MFCC, Mel, CQCC, CQT
        self.cnn_mfcc = FeatureCNN()
        self.cnn_mel = FeatureCNN()
        self.cnn_cqcc = FeatureCNN()
        self.cnn_cqt = FeatureCNN()

        # Assuming input shape: (batch, 1, 20, 100) for each feature
        # After 3 layers of MaxPool2d(2), time/freq reduce by 2^3 = 8
        # e.g., (20, 100) -> (2, 12), with 128 channels
        self.lstm_input_size = 1024  # channels * freq (height)
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=64,
                            num_layers=2, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64 * 2, 1)  # BiLSTM output

    def forward(self, mfcc, mel, cqcc, cqt):
        # Each input: (batch, 1, 20, 100)
        mfcc_out = self.cnn_mfcc(mfcc)
        mel_out = self.cnn_mel(mel)
        cqcc_out = self.cnn_cqcc(cqcc)
        cqt_out = self.cnn_cqt(cqt)

        # All outputs: (batch, 128, 2, 12) → reshape for LSTM
        def reshape_for_lstm(x):
            if x.dim() == 4:
                x = x.permute(0, 3, 1, 2)  # (B, T, C, F)
                x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, T, C*F)
            return x


        mfcc_seq = reshape_for_lstm(mfcc_out)
        mel_seq = reshape_for_lstm(mel_out)
        cqcc_seq = reshape_for_lstm(cqcc_out)
        cqt_seq = reshape_for_lstm(cqt_out)

        # Concatenate along feature axis
        combined_seq = torch.cat([mfcc_seq, mel_seq, cqcc_seq, cqt_seq], dim=2)  # (batch, time, features * 4)

        lstm_out, _ = self.lstm(combined_seq)
        lstm_out = self.dropout(lstm_out)

        out = self.fc(lstm_out[:, -1, :])
        return out
