import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    """ This class implements the Position Encoder from the paper:
        https://arxiv.org/abs/1706.03762

        Ref website:
        https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
        https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
    """

    _pem = None  # positional embedding matrix

    def __init__(self, seq_len: int, emb_dim: int) -> None:
        """
        Parameters:
        out_features: the number of output features of the layer
        seq_len: the length of the sequence or total time steps,
        emb_dim: the dimension of the embedding vector
        """

        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim

        # To save memory, we will only create the positional embedding matrix once
        if self._pem is None or self._pem.shape != (seq_len, emb_dim):
            self._pem = self.create_sinusoidal_pos_emb(seq_len, emb_dim)

    @staticmethod
    def create_sinusoidal_pos_emb(seq_len: int, dim: int) -> torch.Tensor:
        """
        PE(pos, 2i) = sin(pos / 10000^(2i/dim))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))

        where:
            pos is the time position,
            i is the index of the dimension,
            dim is the dimension of the embedding vector.

        """
        pem = torch.zeros(seq_len, dim)  # shape: (seq_length, dim)
        pos = torch.arange(0, seq_len).unsqueeze(1)  # shape: (seq_length, 1)
        i = torch.arange(0, dim, 2)  # shape: (dim / 2, )
        pem[:, 0::2] = torch.sin(pos / 10000 ** (2 * i / dim))
        pem[:, 1::2] = torch.cos(pos / 10000 ** (2 * i / dim))
        return pem

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the Positional Embedding Layer
         Parameters:
            t: the time step, shape: (N, ), where N is the batch size,

         Returns:
             time embedding vectors, shape: (N, dim)
         """
        # Move the positional embedding matrix to the same device as t:
        if self._pem.device != t.device:
            self._pem = self._pem.to(t.device)

        # Get the positional embedding vector for each time step:
        return self._pem[t, :]  # shape: (N, dim)


class Block(nn.Module):
    """ This class implements the Double Convolutional Layers """

    def __init__(self, in_channels: int, out_channels: int, emb_dim: int = 256, residual=True) -> None:
        super().__init__()
        self.residual = residual

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # shape: (N, out_channels, H, W)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # Time Embedding Layer
        self.time_emb = nn.Sequential(
            nn.Linear(emb_dim, out_channels),
            nn.ReLU(),
        )

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)  # shape: (N, out_channels, H, W)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """  shape of x: (N, in_channels, H, W),
             shape of t: (N, emb_dim)
        """
        # Pass the input tensor through the first convolutional layer:
        x = self.conv1(x)  # shape: (N, in_channels, H, W) -> (N, out_channels, H, W)
        x = self.relu(x)  # shape: (N, out_channels, H, W) -> (N, out_channels, H, W)
        x = self.batchnorm1(x)  # shape: (N, out_channels, H, W) -> (N, out_channels, H, W)

        # Add the time embedding vector:
        t = self.time_emb(t)  # shape: (N, emb_dim) -> (N, out_channels)
        t = t.unsqueeze(-1).unsqueeze(-1)  # shape: (N, out_channels, 1, 1)

        # Add the time embedding vector to the input tensor:
        x = x + t  # shape: (N, out_channels, H, W)  -> (N, out_channels, H, W)

        # Pass the input tensor through the second convolutional layer:
        x = self.conv2(x)  # shape: (N, out_channels, H, W) -> (N, out_channels, H, W)

        # Add the residual:
        if self.residual:
            x = x + self.relu(x)  # shape:  (N, out_channels, H, W) -> (N, out_channels, H, W)
        else:
            x = self.relu(x)  # shape:  (N, out_channels, H, W) -> (N, out_channels, H, W)

        x = self.batchnorm2(x)  # shape:  (N, out_channels, H, W) -> (N, out_channels, H, W)

        return x  # shape: (N, out_channels, H, W)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int = 256, residual: bool = True):
        super().__init__()

        # self.down = nn.MaxPool2d(2)  # shape: (N, in_channels, H/2, W/2)
        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.conv = Block(in_channels, out_channels, emb_dim, residual)  # shape: (N, out_channels, H/2, W/2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """  shape of x: (N, in_channels, H, W),
             shape of t: (N, emb_dim)"""
        x = self.down(x)  # shape: (N, in_channels, H, W) -> (N, in_channels, H/2, W/2)
        x = self.conv(x, t)  # shape: (N, in_channels, H/2, W/2) -> (N, out_channels, H/2, W/2)
        return x # shape: (N, out_channels, H/2, W/2) + (N, out_channels, 1, 1) -> (N, out_channels, H/2, W/2)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int = 256, residual: bool = True):
        super().__init__()

        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # shape: (N, in_channels, H*2, W*2)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)  # shape: (N, in_channels, H*2, W*2)
        self.conv = Block(2 * out_channels, out_channels, emb_dim, residual)  # shape: (N, out_channels, H*2, W*2)

    def forward(self, x: torch.Tensor, skip_x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """  shape of x: (N, in_channels, H, W),
             shape of skip_x: (N, in_channels, H*2, W*2)
             shape of t: (N, emb_dim)
        """
        x = self.up(x)  # shape: (N, in_channels, H, W) -> (N, out_channels, H*2, W*2)
        x = torch.cat([x, skip_x], dim=1)  # shape: (N, in_channels + in_channels, H*2, W*2) -> (N, 2*in_channels, H*2, W*2)
        x = self.conv(x, t)  # shape: (N, 2*in_channels, H*2, W*2) -> (N, out_channels, H*2, W*2)
        return x  # shape: (N, out_channels, H*2, W*2) -> (N, out_channels, H*2, W*2)


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 seq_len: int = 1000,
                 emb_dim: int = 256):
        super().__init__()

        self.time_emb = nn.Sequential(
            PositionalEncoder(seq_len, emb_dim),  # shape: (N, dim)
            nn.Linear(emb_dim, emb_dim),  # shape: (N, dim)
            nn.ReLU(),  # shape: (N, dim)
        )  # shape: (N, dim)

        self.inc = Block(in_channels, 64, emb_dim=emb_dim)  # (N, 64, img_size, img_size)
        self.down1 = Down(64, 128, emb_dim=emb_dim)  # (N, 128, img_size/2, img_size/2)
        self.down2 = Down(128, 256, emb_dim=emb_dim)  # (N, 256, img_size/4, img_size/4)
        self.down3 = Down(256, 512, emb_dim=emb_dim)  # (N, 512, img_size/8, img_size/8)
        self.down4 = Down(512, 1024, emb_dim=emb_dim)  # (N, 512, img_size/16, img_size/16)
        self.bot1 = Block(1024, 1024, emb_dim=emb_dim)  # (N, 1024, img_size/16, img_size/16)
        self.up1 = Up(1024, 512, emb_dim=emb_dim)  # (N, 512, img_size/8, img_size/8)
        self.up2 = Up(512, 256, emb_dim=emb_dim)  # (N, 256, img_size/4, img_size/4))
        self.up3 = Up(256, 128, emb_dim=emb_dim)  # (N, 128, img_size/2, img_size/2)
        self.up4 = Up(128, 64, emb_dim=emb_dim)  # (N, 64, img_size, img_size)
        self.outc = Block(64, out_channels, emb_dim=emb_dim)  # (N, out_channels, img_size, img_size)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ shape of x (N, in_channels, H, W),
            shape of t (N, )
         """
        # Initial:
        t = self.time_emb(t)  # shape: (N, emb_dim)
        xi1 = self.inc(x, t)  # shape: (N, in_channels, H, W) -> (N, 64, H, W)

        # Down:
        xd1 = self.down1(xi1, t)  # shape: (N, 64, H, W) -> (N, 128, H/2, W/2)
        xd2 = self.down2(xd1, t)  # shape: (N, 128, H/2, W/2) -> (N, 256, H/4, W/4)
        xd3 = self.down3(xd2, t)  # shape: (N, 256, H/4, W/4) -> (N, 512, H/8, W/8)
        xd4 = self.down4(xd3, t)  # shape: (N, 512, H/8, W/8) -> (N, 512, H/16, W/16)

        # Bottom:
        xb1 = self.bot1(xd4, t)  # shape: (N, 512, H/16, W/16) -> (N, 1024, H/16, W/16)

        # Up:
        x = self.up1(xb1, skip_x=xd3, t=t)  # shape: (N, 1024, H/16, W/16) -> (N, 512, H/8, W/8)
        x = self.up2(x, skip_x=xd2, t=t)  # shape: (N, 512, H/8, W/8) -> (N, 256, H/4, W/4)
        x = self.up3(x, skip_x=xd1, t=t)  # shape: (N, 256, H/4, W/4) -> (N, 128, H/2, W/2)
        x = self.up4(x, skip_x=xi1, t=t)  # shape: (N, 128, H/2, W/2) -> (N, 64, H, W)

        # Output
        x = self.outc(x, t)  # shape: (N, 64, H, W) -> (N, out_channels, H, W)
        return x


if __name__ == '__main__':
    # Test the model:
    N = 4
    in_channels = 3
    out_channels = 3
    seq_len = 1000
    emb_dim = 256
    img_size = 64
    DEVICE ='cpu'

    # Fake data:
    x = torch.randn(N, in_channels, img_size, img_size).to(DEVICE)
    t = torch.randint(0, seq_len, (N,)).to(DEVICE)

    pos = PositionalEncoder(seq_len, emb_dim)

    # Model:
    from torchinfo import summary
    u = UNet(in_channels, out_channels, seq_len, emb_dim)
    u.to(DEVICE)
    summary(u, input_data=(x, t), verbose=1)