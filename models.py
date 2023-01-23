import torch
import math
import torch.nn as nn
import clip
import torch.nn.functional as F
import torchvision



class ImgNet(nn.Module):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_image_encode, _ = clip.load("ViT-B/16", device=self.device)   #512
        # self.vgg16 = torchvision.models.vgg16(pretrained=True)
        # self.vgg16.classifier = self.vgg16.classifier[0:-1]
        self.hash_layer = nn.Sequential(nn.Linear(512, 4096),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(4096, code_len)
                                        )
        self.alpha = 1.0

    def forward(self, x):
        with torch.no_grad():
            feat = self.clip_image_encode.encode_image(x)
            feat = feat.type(torch.float32)
        # feat = self.vgg16(x)
        hid = self.hash_layer(feat)
        code = torch.tanh(self.alpha * hid)
        return feat, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(TxtNet, self).__init__()
        self.hash_layer = nn.Sequential(nn.Linear(txt_feat_len, 4096),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(4096, code_len)
                                        )
        self.alpha = 1.0

    def forward(self, x):
        hid = self.hash_layer(x)
        code = torch.tanh(self.alpha * hid)
        return x, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class AttImgNet(nn.Module):
    def __init__(self):
        super(AttImgNet, self).__init__()
        self.fc = nn.Linear(512, 512)

    def forward(self, x):
        Mask1 = F.sigmod(self.fc(x))
        G1 = x + Mask1 * x
        return G1



class AttTexNet(nn.Module):
    def __init__(self, txt_feat_len):
        super(AttTexNet, self).__init__()
        self.fc = nn.Linear(txt_feat_len, txt_feat_len)

    def forward(self, x):
        Mask2 = F.sigmod(self.fc(x))
        G2 = x + Mask2 * x
        return G2

