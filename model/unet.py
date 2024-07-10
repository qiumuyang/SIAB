import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""
    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU())

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), ConvBlock(in_channels, out_channels, dropout_p))

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""
    def __init__(self,
                 in_channels1,
                 in_channels2,
                 out_channels,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1,
                                         in_channels2,
                                         kernel_size=2,
                                         stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p=0.0)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert len(self.ft_chns) == 5
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0],
                                 self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1],
                               self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2],
                               self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3],
                               self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4],
                               self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert len(self.ft_chns) == 5

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3])
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2])
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1])
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0])

        self.out_conv = nn.Conv2d(self.ft_chns[0],
                                  self.n_class,
                                  kernel_size=3,
                                  padding=1)

    def forward(self, feature, feat=False):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        if feat:
            return output, x
        return output


class UNet(nn.Module):

    def __init__(self, in_chns, class_num, dropout: bool = False):
        super(UNet, self).__init__()

        dropout_on = [0.05, 0.1, 0.2, 0.3, 0.5]
        dropout_off = [0.0] * len(dropout_on)
        params = {
            'in_chns': in_chns,
            'feature_chns': [16, 32, 64, 128, 256],
            'dropout': dropout_on if dropout else dropout_off,
            'class_num': class_num,
            'bilinear': False,
        }

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x, feat=False):
        feature = self.encoder(x)

        output = self.decoder(feature, feat=feat)
        if feat:
            output, last_feature = output
            return output, feature, last_feature
        return output
