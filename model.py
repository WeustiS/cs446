import torch
import torch.nn as nn


def conv3d_bn_relu(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.ReLU(inplace=True))


def conv3dtrans_bn_relu(in_dim, out_dim):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        nn.ReLU(inplace=True)
    )

def unet_max_pool():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def unet_conv_block(in_dim, out_dim):
    return nn.Sequential(
        conv3d_bn_relu(in_dim, out_dim),
        conv3d_bn_relu(out_dim, out_dim),
        conv3d_bn_relu(out_dim, out_dim)
    )


class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters

        # Down sampling
        self.down_1 = unet_conv_block(self.in_dim, self.num_filters)
        self.pool_1 = unet_max_pool()
        self.down_2 = unet_conv_block(self.num_filters, self.num_filters * 2)
        self.pool_2 = unet_max_pool()
        self.down_3 = unet_conv_block(self.num_filters * 2, self.num_filters * 4)
        self.pool_3 = unet_max_pool()
        self.down_4 = unet_conv_block(self.num_filters * 4, self.num_filters * 8)
        self.pool_4 = unet_max_pool()
        self.down_5 = unet_conv_block(self.num_filters * 8, self.num_filters * 16)
        self.pool_5 = unet_max_pool()

        # Bridge
        self.bridge = unet_conv_block(self.num_filters * 16, self.num_filters * 32)

        # Up sampling
        self.trans_1 = conv3dtrans_bn_relu(self.num_filters * 32, self.num_filters * 32)
        self.up_1 = unet_conv_block(self.num_filters * 48, self.num_filters * 16)
        self.trans_2 = conv3dtrans_bn_relu(self.num_filters * 16, self.num_filters * 16)
        self.up_2 = unet_conv_block(self.num_filters * 24, self.num_filters * 8)
        self.trans_3 = conv3dtrans_bn_relu(self.num_filters * 8, self.num_filters * 8)
        self.up_3 = unet_conv_block(self.num_filters * 12, self.num_filters * 4)
        self.trans_4 = conv3dtrans_bn_relu(self.num_filters * 4, self.num_filters * 4)
        self.up_4 = unet_conv_block(self.num_filters * 6, self.num_filters * 2)
        self.trans_5 = conv3dtrans_bn_relu(self.num_filters * 2, self.num_filters * 2)
        self.up_5 = unet_conv_block(self.num_filters * 3, self.num_filters)

        # Output
        self.out = unet_conv_block(self.num_filters, out_dim)

    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x)       # -> [n, 4, 192, 192, 192]
        pool_1 = self.pool_1(down_1)  # -> [n, 4, 96, 96, 96]

        down_2 = self.down_2(pool_1)  # -> [n, 8, 96, 96, 96]
        pool_2 = self.pool_2(down_2)  # -> [n, 8, 48, 48, 48

        down_3 = self.down_3(pool_2)  # -> [n, 16, 48, 48, 48]
        pool_3 = self.pool_3(down_3)  # -> [n, 16, 24, 24, 24]

        down_4 = self.down_4(pool_3)  # -> [n, 32, 24, 24, 24]
        pool_4 = self.pool_4(down_4)  # -> [n, 32, 12, 12, 12]

        down_5 = self.down_5(pool_4)  # -> [n, 64, 12, 12, 12]
        pool_5 = self.pool_5(down_5)  # -> [n, 64, 6, 6, 6]

        # Bridge
        bridge = self.bridge(pool_5)  # -> [n, 128, 6, 6, 6]

        # Up sampling
        trans_1 = self.trans_1(bridge)  # -> [n, 128, 12, 12, 12]
        concat_1 = torch.cat([trans_1, down_5], dim=1)  # -> [n, (128+64), 12,12,12]
        up_1 = self.up_1(concat_1)  # -> [n, 64, 12,12,12]

        trans_2 = self.trans_2(up_1)  # -> [n, 64, 24,24,24]
        concat_2 = torch.cat([trans_2, down_4], dim=1)  # -> [n, (64+32),24,24,24]
        up_2 = self.up_2(concat_2)  # -> [n, 32, 24,24,24]

        trans_3 = self.trans_3(up_2)  # -> [n, 32, 48, 48, 48]
        concat_3 = torch.cat([trans_3, down_3], dim=1)  # -> [n, (32+16), 48, 48, 48]
        up_3 = self.up_3(concat_3)  # -> [n, 16, 48, 48, 48]

        trans_4 = self.trans_4(up_3)  # -> [n, 16, 96, 96, 96]
        concat_4 = torch.cat([trans_4, down_2], dim=1)  # -> [n, (16+8), 96, 96, 96]
        up_4 = self.up_4(concat_4)  # -> [n, 8, 96, 96, 96]

        trans_5 = self.trans_5(up_4)  # -> [n, 8, 192, 192, 192]
        concat_5 = torch.cat([trans_5, down_1], dim=1)  # -> [n, (8+4), 192, 192, 192]
        up_5 = self.up_5(concat_5)  # -> [n, 4, 192, 192, 192]

        # Output
        out = self.out(up_5)  # -> [n, out, 192, 192, 192]
        return out


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    image_size = 192
    x = torch.Tensor(1, 4, image_size, image_size, image_size)
    x.to(device)
    print("x size: {}".format(x.size()))

    model = UNet(in_dim=4, out_dim=1, num_filters=2)
    print(f'PARAMS: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    out = model(x)
    print("out size: {}".format(out.size()))