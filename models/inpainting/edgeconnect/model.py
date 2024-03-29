import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from skimage.color import rgb2gray
from skimage.feature import canny


class InpaintModel:
    def __init__(self, device):
        # device
        self.device = device

        # edge generator setting
        self.edge_model = EdgeGenerator()
        self.edge_model.load_state_dict(
            torch.load(
                "./models/inpainting/edgeconnect/weights/EdgeModel_gen.pth",
                map_location=torch.device(device),
            )["generator"]
        )
        self.edge_model = self.edge_model.eval()
        self.edge_model = self.edge_model.to(device)

        # inpaint generator setting
        self.inpaint_model = InpaintGenerator()
        self.inpaint_model.load_state_dict(
            torch.load(
                "./models/inpainting/edgeconnect/weights/InpaintingModel_gen.pth",
                map_location=torch.device(device),
            )["generator"]
        )
        self.inpaint_model = self.inpaint_model.eval()
        self.inpaint_model = self.inpaint_model.to(device)

    def t(self, img, grayscale=False):
        if grayscale:
            t = T.Compose(
                [
                    Image.fromarray,
                    T.Grayscale(),
                    T.ToTensor(),
                    lambda x: x.float(),
                    lambda x: x.unsqueeze(0),
                ]
            )
        else:
            t = T.Compose(
                [
                    Image.fromarray,
                    T.ToTensor(),
                    lambda x: x.float(),
                    lambda x: x.unsqueeze(0),
                ]
            )

        return t(img)

    def __call__(self, img, mask):
        # edge generating
        # preprocessing
        img_gray = self.t(img, grayscale=True)
        edge = self.t(canny(rgb2gray(img), sigma=1, mask=(1 - mask).astype(bool)))
        mask = self.t(mask)

        img_gray_masked = img_gray * (1 - mask.bool().int()) + mask * 255
        edge_masked = edge * (1 - mask.bool().int())
        input_edge = torch.cat((img_gray_masked, edge_masked, mask), dim=1)

        input_edge = input_edge.to(self.device)

        # forward
        with torch.no_grad():
            edge_gen = self.edge_model(input_edge)

        # inpainting
        # preprocessing
        img = self.t(img)

        img_masked = img * (1 - mask.bool().int()) + mask * 255
        img_masked = img_masked.to(self.device)
        input_inpaint = torch.cat((img_masked, edge_gen), dim=1)

        # forward
        with torch.no_grad():
            img_gen = self.inpaint_model(input_inpaint)[0]

        # postprocessing
        img_gen = torch.moveaxis(img_gen, 0, -1)

        return img_gen


# Codes below are from EdgeConnect
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type="normal", gain=0.02):
        """
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        """

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                if init_type == "normal":
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier":
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find("BatchNorm2d") != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class InpaintGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2

        return x


class EdgeGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, use_spectral_norm=True, init_weights=True):
        super(EdgeGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0),
                use_spectral_norm,
            ),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
            spectral_norm(
                nn.Conv2d(
                    in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1
                ),
                use_spectral_norm,
            ),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
            spectral_norm(
                nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                use_spectral_norm,
            ),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=128,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                use_spectral_norm,
            ),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
            spectral_norm(
                nn.ConvTranspose2d(
                    in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1
                ),
                use_spectral_norm,
            ),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x


class Discriminator(BaseNetwork):
    def __init__(
        self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True
    ):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=64,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels=512,
                    out_channels=1,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    padding=0,
                    dilation=dilation,
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            spectral_norm(
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    padding=0,
                    dilation=1,
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
