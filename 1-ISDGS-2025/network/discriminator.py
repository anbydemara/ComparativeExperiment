import torch
import torch.nn as nn
import torch.nn.functional as F
import  numpy as np

class Discriminator(nn.Module):

    def __init__(self, inchannel, outchannel, num_classes, patch_size):
        super(Discriminator, self).__init__()
        dim = 512
        self.patch_size = patch_size
        self.inchannel = inchannel
        self.conv1 = nn.Conv2d(inchannel, 64, kernel_size=3, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)  # 对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self._get_final_flattened_size(), dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.relu4 = nn.ReLU(inplace=True)

        self.cls_head_src = nn.Linear(dim, num_classes)
        self.p_mu = nn.Linear(dim, outchannel, nn.LeakyReLU())
        self.pro_head = nn.Linear(dim, outchannel, nn.ReLU())

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.inchannel,
                             self.patch_size, self.patch_size))
            in_size = x.size(0)
            out1 = self.mp(self.relu1(self.conv1(x)))
            out2 = self.mp(self.relu2(self.conv2(out1)))
            out2 = out2.view(in_size, -1)
            w, h = out2.size()
            fc_1 = w * h
        return fc_1

    def forward(self, x, mode='test'):  # 前向传播默认test模式

        in_size = x.size(0)
        if mode == 'train' :
            hrand = np.random.randint(0, 100)
            wrand = np.random.randint(0, 100)
            if hrand < 5:
                Hfuzhu = torch.empty(x.size(0), x.size(1), 1, 13).to(0)
                x = torch.cat([x, Hfuzhu], dim=2)
                x = x.view(x.size(0), x.size(1), 2, 7, 13)
                x = x.transpose(2, 3).contiguous()
                x = x.view(x.size(0), x.size(1), 14, 13)
                a = torch.split(x, 6, dim=2)
                b = torch.split(a[2], 1, dim=2)
                x = torch.cat([a[1], b[0]], dim=2)
                x = torch.cat([x, a[0]], dim=2)

            if wrand < 5:
                Hfuzhu = torch.empty(x.size(0), x.size(1), 1, 13).to(0)
                x = x.transpose(2, 3)
                x = torch.cat([x, Hfuzhu], dim=2)
                x = x.view(x.size(0), x.size(1), 2, 7, 13)
                x = x.transpose(2, 3).contiguous()
                x = x.view(x.size(0), x.size(1), 14, 13)
                a = torch.split(x, 6, dim=2)
                b = torch.split(a[2], 1, dim=2)
                x = torch.cat([a[1], b[0]], dim=2)
                x = torch.cat([x, a[0]], dim=2)
                x = x.transpose(2, 3)

        out1 = self.mp(self.relu1(self.conv1(x)))
        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))

        if mode == 'test':
            clss = self.cls_head_src(out4)
            return clss
        elif mode == 'train':
            proj = F.normalize(self.pro_head(out4))
            clss = self.cls_head_src(out4)

            return clss, proj

