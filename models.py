import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as tca
from tqdm import tqdm

class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, in_tensor):
        return in_tensor.view((in_tensor.size()[0], -1))

class B2lock2(nn.Module):
    def __init__(self, in_planes, planes,ker=3,stride=1,down=True):
        super(B2lock2, self).__init__()
        usemp=False
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=ker, stride=stride if not down or usemp else 2, padding=ker>1, bias=False)
        self.bnF = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d((2, 2), stride=2) if down and usemp else None


    def forward(self, out):
        out=self.conv1(out)
        out = self.bnF(out)
        out = self.relu(out)
        out = out if self.mp is None else self.mp(out)
        return out

class Classifier(nn.Module):
    def __init__(self, cfg):
        super(Classifier, self).__init__()
        tr = lambda x: x
        self.in_channels=cfg["imCh"]
        chans = [self.in_channels, 32,  64,  128, 128, 256, 256, 512, 512,512]

        i=-1
        def getConv(ker=3, down=True):
            nonlocal i
            i+=1
            return B2lock2( (tr(chans[i]) if i>0 else chans[i]),tr(chans[i+1]), ker=ker,down=down)

        self.conv0 = getConv()
        self.conv1 = getConv()
        self.conv2a = getConv( down=False)
        self.conv2 = getConv()
        self.conv3a = getConv( down=False)
        self.conv3 = getConv()
        self.conv4a = getConv( down=False)
        self.conv4 = getConv()
        
        self.convLays = [self.conv0,self.conv1, self.conv2a,self.conv2, self.conv3a,self.conv3, self.conv4a, self.conv4]
        if cfg["ds"][0] == "TinyImgNet":
            self.mp4 = nn.MaxPool2d((2,2), stride=2)
            self.convLays.append(self.mp4)

        i, ker = -1, 1
        self.flat = Flatten()
        self.dropout = nn.Dropout(0.5)
        #self.nfea = 2048 if cfg["ds"][0] == "TinyImgNet"else tr(512)
        self.nfea = tr(512)
        self.pred = nn.Linear(self.nfea, cfg["num_classes"])
        self.allExplainLays = self.convLays+[self.pred]


    def forward(self, x):
        for il,l in enumerate(self.convLays):
            x = l(x)
        x=self.flat(x)
        x = self.dropout(x)
        x=self.pred(x)
        return x

def decay(opt,epoch,optimizerCl):
    if opt[0] == "S" and (epoch + 1) % (opt[1] // 3+opt[1]//10+2 ) == 0:
        for p in optimizerCl.param_groups: p['lr'] *= 0.1
        #print("  D", np.round(optimizerCl.param_groups[0]['lr'],5))

def getAcc(net, dataset,  niter=10000,norm=None):
    correct,total = 0,0
    net.eval()
    with torch.no_grad():
        for cit,data in enumerate(dataset):
            with tca.autocast():
                dsx,dsy = data[0].cuda(),data[1].cuda()
                if len(dsx.size()) == 5:
                    dsx = dsx.squeeze(0)
                    dsy = dsy.squeeze(0)
                total += dsy.size(0)
                outputs = net(dsx.float())
                _, predicted = torch.max(outputs.data, 1)
                correct += torch.eq(predicted, dsy).sum().item()
                if cit>=niter: break
    return correct/total


def getclassifier(cfg,train_dataset,val_dataset,norm=None):
    netCl=Classifier(cfg).cuda()
    if cfg["ds"][0] == "TinyImgNet":
        optimizerCl = optim.Adam(netCl.parameters(), lr=cfg["opt"][2], weight_decay=cfg["opt"][3])
    else:
        optimizerCl = optim.SGD(netCl.parameters(), lr=cfg["opt"][2], momentum=0.9, weight_decay=cfg["opt"][3])
    #optimizerCl = optim.SGD(netCl.parameters(), lr=cfg["opt"][2], momentum=0.9, weight_decay=cfg["opt"][3])
    closs,teaccs,trep,loss,clr = 0,[],cfg["opt"][1],nn.CrossEntropyLoss(), cfg["opt"][2]
    print("Train Classifier to explain")
    scaler = tca.GradScaler()
    teAccs,trAccs=[],[]
    clAcc = lambda dataset: getAcc(netCl, dataset,  niter=1e10,norm=norm)
    for epoch in tqdm(range(trep)):
        netCl.train()
        for i, data in enumerate(tqdm(train_dataset, position=0, leave=True)):
          with tca.autocast():
            optimizerCl.zero_grad()
            dsx = data[0]
            dsx,dsy = dsx.cuda(),data[1].cuda()
            if len(dsx.size()) == 5:
                dsx = dsx.squeeze(0)
                dsy = dsy.squeeze(0)
            output = netCl(dsx.float())
            errD_real = loss(output, dsy.long())
            scaler.scale(errD_real).backward()
            scaler.step(optimizerCl)
            scaler.update()
            closs = 0.97 * closs + 0.03 * errD_real.item() if i > 20 else 0.8 * closs + 0.2 * errD_real.item()
        decay(cfg["opt"],epoch,optimizerCl)
        netCl.eval()
        teAccs.append(clAcc(val_dataset))
        if (epoch % 4 == 0 and epoch<=13) or (epoch % 20==0 and epoch>13):
            print(epoch, np.round(np.array([closs, teAccs[-1], clAcc(train_dataset)]), 5))
    lcfg = {"testAcc": clAcc(val_dataset), "trainAcc": clAcc(train_dataset)}
    netCl.eval()
    return netCl, lcfg