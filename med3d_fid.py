"""
Built on https://github.com/batmanlab/HA-GAN/tree/master
Sun, L., Chen, J., Xu, Y., Gong, M., Yu, K., & Batmanghelich, K. (2022). Hierarchical Amortized GAN for 3D High Resolution Medical Image Synthesis. IEEE Journal of Biomedical and Health Informatics, 26(8), 3966–3975. https://doi.org/10.1109/JBHI.2022.3172976
"""

import os
from component.dataset import TwoClassDataset, OneClassDataset
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
from scipy import linalg

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class upsample(nn.Module):
  def forward(self, inp):
    return F.interpolate(inp, scale_factor = 2)

def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

class Flatten(torch.nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_seg_classes=2,
                 shortcut_type='B',
                 no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.conv_seg = nn.Sequential(
            upsample(),
            nn.ConvTranspose3d(
                512 * block.expansion,
                32,
                2,
                stride=2
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            upsample(),
            nn.Conv3d(
                32,
                32,
                kernel_size=3,
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                32,
                num_seg_classes,
                kernel_size=1,
                stride=(1, 1, 1),
                bias=False),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)
        x = self.conv_seg(x)

        return x

def get_activations_from_dataloader(model, data_loader, num_samples, batch_size):

    pred_arr = np.empty((num_samples, 2048))
    for i, batch in enumerate(data_loader):
        if i % 10 == 0:
            print('\rPropagating batch %d' % i, end='', flush=True)
        x, label = batch
        with torch.no_grad():
            pred = model(x)

        if i*batch_size > pred_arr.shape[0]:
            pred_arr[i*batch_size:] = pred.cpu().numpy()
        else:
            pred_arr[i*batch_size:(i+1)*batch_size] = pred.cpu().numpy()
    print(' done')
    return pred_arr

def trim_state_dict_name(ckpt):
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

class Flatten(torch.nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def get_feature_extractor(pretrained_resnet):
    model = resnet50(shortcut_type='B')
    model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), Flatten()) # (N, 512)
    # ckpt from https://drive.google.com/file/d/1399AsrYpQDi1vq6ciKRQkfknLsQQyigM/view?usp=sharing
    ckpt = torch.load(pretrained_resnet)
    ckpt = trim_state_dict_name(ckpt["state_dict"])
    model.load_state_dict(ckpt) # No conv_seg module in ckpt
    model = nn.DataParallel(model).cuda()
    model.eval()
    print("Feature extractor weights loaded")
    return model

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def post_process(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def interpolate_3d(img):
    return F.interpolate(img, size=(256,256,256), mode='trilinear', align_corners=False)

def calculate_fid_real(pretrained_resnet, healthy, defective, fake, batch_size):
    """Calculates the FID of two paths"""
    if os.path.exists('act_real.npy'):
        act_real = np.load('act_real.npy')
    else:
        model = get_feature_extractor(pretrained_resnet)
        #dataset = COPD_dataset(img_size=args.img_size, stage="train", fold=args.fold, threshold=600)
        dataset_real = TwoClassDataset(healthy, defective, transform=interpolate_3d)
        num_samples_real = len(dataset_real)
        data_loader_real = torch.utils.data.DataLoader(dataset_real, batch_size=batch_size,drop_last=False,shuffle=False)
        act_real = get_activations_from_dataloader(model, data_loader_real, num_samples_real, batch_size)
        np.save('act_real.npy', act_real)
    m, s = post_process(act_real)


    model = get_feature_extractor(pretrained_resnet)
    dataset_fake = OneClassDataset(fake, transform=interpolate_3d)
    num_samples_fake = len(dataset_fake)
    data_loader_fake = torch.utils.data.DataLoader(dataset_fake, batch_size=batch_size, drop_last=False, shuffle=False)
    act_fake = get_activations_from_dataloader(model, data_loader_fake, num_samples_fake, batch_size)
    m1, s1 = post_process(act_fake)

    fid_value = calculate_frechet_distance(m1, s1, m, s)
    print('FID: ', fid_value)


if __name__ == '__main__':
    healthy = r'D:/Hugo/healthy'
    defective = r'D:/Hugo/defective'
    fake = r'D:\Hugo\VAE_generation'
    pretrained_resnet = r"D:/Hugo/resnet_50.pth"
    calculate_fid_real(pretrained_resnet, healthy, defective, fake,24)