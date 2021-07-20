import numpy as np
import torch.cuda.amp as tca
import torch
from models import decay
from torch import nn
import pdb
from tqdm import tqdm
from dutils import getnorm
import torchvision.transforms as transforms


class ClaDecNet(nn.Module):
    def __init__(self, cfg, inShape, nFea):
        super(ClaDecNet, self).__init__()
        self.channel_mult = int(64)
        self.expLinLay = len(inShape) == 2  # linear layer...

        if cfg["ds"][0] == "TinyImgNet":
            dim = 64 if self.expLinLay else 64 // int(inShape[-1])  # dimension of input
        else:
            dim = 32 if self.expLinLay else 32 // int(inShape[-1])  # dimension of input
        self.input_dim = np.prod(inShape[1:])
        self.inFea = inShape[-2] if not self.expLinLay else 1
        nLay = int(np.round(np.log2(dim) - 2))

        # Use batchnorm or bias? LeakyRelu or Relu   -->  Does not make much of a difference
        # bn,bias = lambda x:  nn.BatchNorm2d(x),False
        # rel =lambda x:  nn.LeakyReLU(0.01)
        bn, bias = lambda x: nn.Identity(), True
        rel = lambda x: nn.ReLU()

        if (
            self.inFea == 1
        ):  # There is no spatial dimension (or it is one) -> use a dense layer as the first layer
            self.useDense = True
            self.fc_output_dim = max(
                self.input_dim, self.channel_mult
            )  # number of input features
            self.fc = nn.Sequential(
                nn.Linear(self.input_dim, self.fc_output_dim), nn.ReLU(True)
            )  # , nn.BatchNorm1d(self.fc_output_dim)
        else:  # The spatial extend is larger one, use a conv layer, otherwise have too many parameters
            self.fc_output_dim = inShape[1]  # number of input features
            self.fc = nn.Sequential(
                nn.Conv2d(inShape[1], inShape[1], 3, stride=1, padding=1, bias=bias),
                nn.ReLU(True),
            )  # , nn.BatchNorm1d(self.fc_output_dim)
            self.useDense = False

        self.deconv = [
            nn.ConvTranspose2d(
                self.fc_output_dim,
                self.channel_mult * (2 ** nLay),
                4,
                stride=2,
                padding=1,
                bias=bias,
            ),
            bn(self.channel_mult * (2 ** nLay)),
            rel(None),
        ]
        for j in range(nLay, 0, -1):
            self.deconv += [
                nn.ConvTranspose2d(
                    self.channel_mult * (2 ** j),
                    self.channel_mult * (2 ** (j - 1)),
                    4,
                    stride=2,
                    padding=1,
                    bias=bias,
                ),
                bn(self.channel_mult * (2 ** (j - 1))),
                rel(None),
            ]
        self.deconv.append(
            nn.ConvTranspose2d(
                self.channel_mult * 1, nFea, 4, stride=2, padding=1, bias=True
            )
        )
        self.deconv = nn.Sequential(*self.deconv)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        if self.useDense:
            x = x.view(-1, self.input_dim)
            x = self.fc(x)
            x = x.view(-1, self.fc_output_dim, self.inFea, self.inFea)
        else:
            x = self.fc(x)
        x = self.deconv(x)
        x = self.sig(
            x
        )  # This sometimes gives better visualization, but you need to take care to standardize inputs as well
        return x


def getClaDec(cfg, netCl, norm, train_dataset):
    alpha = cfg["alpha"]
    closs, cclloss, crloss, teaccs, trep, clloss, clr = (
        0,
        0,
        0,
        [],
        cfg["opt"][1],
        nn.CrossEntropyLoss(),
        cfg["opt"][2],
    )
    print("Train CLaDec")
    scaler = tca.GradScaler()
    d = next(iter(train_dataset))
    netDec = ClaDecNet(cfg, d[2].squeeze(0).shape, cfg["imCh"]).cuda()
    netDec.train()
    netCl.train()
    optimizerCl = torch.optim.Adam(netDec.parameters(), lr=0.0003, weight_decay=1e-5)
    aeloss = nn.MSELoss()
    ulo = (
        lambda closs, totloss, i: 0.97 * closs + 0.03 * totloss.item()
        if epoch > 20
        else 0.8 * closs + 0.2 * totloss.item()
    )
    for epoch in tqdm(range(trep)):
        for i, data in enumerate(train_dataset):
            with tca.autocast():
                optimizerCl.zero_grad()
                if cfg["ds"][0] == "TinyImgNet":
                    dsx, dsy, dsact = (
                        data[0].squeeze(0).cuda(),
                        data[1].squeeze(0).cuda(),
                        data[2].squeeze(0).cuda(),
                    )
                else:
                    dsx, dsy, dsact = data[0].cuda(), data[1].cuda(), data[2].cuda()
                #dsx, dsy, dsact = data[0].cuda(), data[1].cuda(), data[2].cuda()

                output = netDec(dsact.float())
                recloss = aeloss(output, dsx)
                if cfg["ds"][0] == "TinyImgNet":
                    norm = getnorm("TinyImgNet")
                    transform = transforms.Compose(
                        [
                            transforms.Normalize(mean=norm[0], std=norm[1]),
                            transforms.RandomResizedCrop(64),
                            transforms.RandomHorizontalFlip(),
                        ]
                    )
                    claloss = clloss(netCl(transform(output)), dsy.long())
                else:
                    claloss = clloss(netCl(output), dsy.long())
                # claloss = clloss(netCl(output), dsy.long())
                totloss = (1 - alpha) * recloss + alpha * claloss
                scaler.scale(totloss).backward()
                scaler.step(optimizerCl)
                scaler.update()
                closs, cclloss, crloss = (
                    ulo(closs, totloss, epoch),
                    ulo(cclloss, claloss, epoch),
                    ulo(crloss, recloss, epoch),
                )

        decay(cfg["opt"], epoch, optimizerCl)
        if (epoch % 2 == 0 and epoch <= 10) or (epoch % 10 == 0 and epoch > 10):
            print(epoch, np.round(np.array([closs, crloss, cclloss]), 5))

    lcfg = {"ClaLo": closs}
    netDec.eval()
    return netDec, lcfg


def getActModel(cfg, classifier):
    ind = cfg["layInd"]
    if cfg["ds"][0] == "TinyImgNet":
        if ind < -1:
            ind = ind - 3
    else:
        if ind < -1:
            ind = ind - 2
    actModel = nn.Sequential(*list(classifier.children())[:ind])
    actModel.eval()
    return actModel


class RefAE(nn.Module):
    def __init__(self, cfg, inShape):
        super(RefAE, self).__init__()
        self.cladec = ClaDecNet(cfg, inShape, cfg["imCh"])
        self.cladec.train()
        from models import Classifier

        cla = Classifier(cfg)
        actModel = getActModel(cfg, cla)
        actModel.train()
        self.seq = nn.Sequential(actModel, self.cladec)

    def forward(self, x):
        return self.seq(x)


def getRefAE(cfg,acts ,train_dataset):
    closs, teaccs, trep, clloss, clr = (
        0,
        [],
        cfg["opt"][1],
        nn.CrossEntropyLoss(),
        cfg["opt"][2],
    )
    print("Train RefAE")
    scaler = tca.GradScaler()
    d = next(iter(acts))
    netDec = RefAE(cfg, d[2].squeeze(0).shape).cuda()
    netDec.train()
    optimizerCl = torch.optim.Adam(netDec.parameters(), lr=0.0003, weight_decay=1e-5)
    aeloss = nn.MSELoss()
    ulo = (
        lambda closs, totloss, i: 0.97 * closs + 0.03 * totloss.item()
        if epoch > 20
        else 0.8 * closs + 0.2 * totloss.item()
    )
    for epoch in tqdm(range(trep)):
        for i, data in enumerate(train_dataset):
            with tca.autocast():
                optimizerCl.zero_grad()
                # if cfg["ds"][0] == "TinyImgNet":
                #     dsx = data[0].squeeze(0).cuda()
                # else:
                #     dsx = data[0].cuda()
                dsx = data[0].cuda()
                output = netDec(dsx.float())
                recloss = aeloss(output, dsx)
                scaler.scale(recloss).backward()
                scaler.step(optimizerCl)
                scaler.update()
                closs = ulo(closs, recloss, epoch)

        decay(cfg["opt"], epoch, optimizerCl)
        if (epoch % 2 == 0 and epoch <= 10) or (epoch % 10 == 0 and epoch > 10):
            print(epoch, np.round(np.array([closs]), 5))

    lcfg = {"RefAELo": closs}
    netDec.eval()
    return netDec, lcfg
