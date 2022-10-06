import torch
from torch import nn
from torch.nn import functional as fn


class ConvBnReLu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ConvBnReLu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding='same')
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(fn.relu(self.conv(x)))

    def reset_parameters(self):
        for _, child in self.named_children():
            if 'reset_parameters' in dir(child):
                child.reset_parameters()


class UTime(nn.Module):
    """This class is an implementation of U-Time.
    source: https://github.com/perslev/U-Time
    """
    def __init__(self, window_size: int, in_channels: int = 1, out_channels: int = 5):
        super(UTime, self).__init__()
        # self.bs = batch_size
        self.t = window_size
        self.l = 3000
        self.in_ch = in_channels
        self.out_ch = out_channels

        # encoders
        self.enc1 = ConvBnReLu(in_channels, 16, 5)
        self.enc2 = ConvBnReLu(16, 16, 5)

        self.ds1 = nn.MaxPool2d((10, 1))
        self.enc3 = ConvBnReLu(16, 32, 5)
        self.enc4 = ConvBnReLu(32, 32, 5)

        self.ds2 = nn.MaxPool2d((8, 1))
        self.enc5 = ConvBnReLu(32, 64, 5)
        self.enc6 = ConvBnReLu(64, 64, 5)

        self.ds3 = nn.MaxPool2d((6, 1))
        self.enc7 = ConvBnReLu(64, 128, 5)
        self.enc8 = ConvBnReLu(128, 128, 5)

        self.ds4 = nn.MaxPool2d((4, 1))
        self.enc9 = ConvBnReLu(128, 256, 5)
        self.enc10 = ConvBnReLu(256, 256, 5)

        # decoders
        self.dec1 = ConvBnReLu(256, 128, 4)
        self.us1 = nn.Upsample(size=(self.t * self.l // 10 // 8 // 6, 1))
        self.dec2 = ConvBnReLu(256, 128, 5)
        self.dec3 = ConvBnReLu(128, 128, 5)

        self.dec4 = ConvBnReLu(128, 64, 4)
        self.us2 = nn.Upsample(size=(self.t * self.l // 10 // 8, 1))
        self.dec5 = ConvBnReLu(128, 64, 5)
        self.dec6 = ConvBnReLu(64, 64, 5)

        self.dec7 = ConvBnReLu(64, 32, 4)
        self.us3 = nn.Upsample(size=(self.t * self.l // 10, 1))
        self.dec8 = ConvBnReLu(64, 32, 5)
        self.dec9 = ConvBnReLu(32, 32, 5)

        self.dec10 = ConvBnReLu(32, 16, 4)
        self.us4 = nn.Upsample(size=(self.t * self.l, 1))
        self.dec11 = ConvBnReLu(32, 16, 5)
        self.dec12 = ConvBnReLu(16, 16, 5)

        # classifier
        self.conv1 = nn.Conv2d(16, 5, kernel_size=(1, 1), padding='same')
        self.global_padding = nn.AvgPool2d((1, self.l))
        self.conv2 = nn.Conv2d(5, 5, kernel_size=(19, 1), padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: the input tensors, shape should be [bs, C, T, L]
        :return: prediction
        """
        x = x.transpose(2, 1)
        x = x[:, :, 2, :]
        #print(x.shape)
        x = x.view(x.shape[0], 1, self.t * x.shape[-1], 1)
        #print(x.shape)

        # encoder
        x1 = self.enc1(x)
        x1 = self.enc2(x1)

        x2 = self.ds1(x1.clone())
        x2 = self.enc3(x2)
        x2 = self.enc4(x2)

        x3 = self.ds2(x2.clone())
        x3 = self.enc5(x3)
        x3 = self.enc6(x3)

        x4 = self.ds3(x3.clone())
        x4 = self.enc7(x4)
        x4 = self.enc8(x4)

        # decoder
        stream = self.ds4(x4.clone())
        stream = self.enc9(stream)
        stream = self.enc10(stream)

        stream = self.dec1(stream)
        stream = torch.cat((self.us1(stream), x4), -3)
        stream = self.dec2(stream)
        stream = self.dec3(stream)

        stream = self.dec4(stream)
        stream = torch.cat((self.us2(stream), x3), -3)
        stream = self.dec5(stream)
        stream = self.dec6(stream)

        stream = self.dec7(stream)
        stream = torch.cat((self.us3(stream), x2), -3)
        stream = self.dec8(stream)
        stream = self.dec9(stream)

        stream = self.dec10(stream)
        stream = torch.cat((self.us4(stream), x1), -3)
        stream = self.dec11(stream)
        stream = self.dec12(stream)

        # classifier
        stream = torch.tanh(self.conv1(stream))
        stream = stream.view(x.shape[0], self.out_ch, self.t, self.l)
        stream = self.global_padding(stream)
        #print(stream.shape)
        stream = self.conv2(stream)  # [bs, out_ch]
        #print(stream.shape)
        return torch.squeeze(stream)

    def reset_parameters(self):
        for _, child in self.named_children():
            if 'reset_parameters' in dir(child):
                child.reset_parameters()
        print("[Model]: Parameters have been reset.")


if __name__ == '__main__':
    from torchinfo import summary

    batch_size = 12
    t = 35
    model = UTime(t)
    summary(model, input_size=(batch_size, 1, t, 3000))
