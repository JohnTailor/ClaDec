import torchvision,torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset,TensorDataset
import numpy as np,sklearn
import torch.nn.functional as F

gds = lambda dataset,cfg: torch.utils.data.DataLoader(TensorDataset(*[torch.from_numpy(x) for x in dataset]), batch_size=cfg["batchSize"])

def getnorm(dname):
     if dname == "Ci10": return (torch.from_numpy(np.array((0.4914, 0.4822, 0.4465),np.float32).reshape(1,3,1,1)).cuda(), torch.from_numpy(np.array((0.2023, 0.1994, 0.2010),np.float32).reshape(1,3,1,1)).cuda())
     elif dname == "Ci100": return (torch.from_numpy(np.array((0.5060725 , 0.48667726, 0.4421305),np.float32).reshape(1,3,1,1)).cuda() , torch.from_numpy(np.array((0.2675421,0.25593522,0.27593908),np.float32).reshape(1,3,1,1)).cuda())
     elif dname == "Fash": return (torch.from_numpy(np.array((0.281),np.float32).reshape(1,1,1,1)).cuda() , torch.from_numpy(np.array((0.352),np.float32).reshape(1,1,1,1)).cuda())

def getFullDS(cfg):
    dname=cfg["ds"][0]
    trans=transforms.Compose([transforms.ToTensor()])
    if  dname== "Ci10":
        cdat = torchvision.datasets.CIFAR10  # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) #transform = transforms.Compose([transforms.ToTensor(), norm])
        cfg["imCh"] = 3
    elif dname == "Ci100":
        cdat = torchvision.datasets.CIFAR100  # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        cfg["imCh"] = 3
    elif dname == "Fash":
        cdat = torchvision.datasets.FashionMNIST
        refu = lambda x: F.interpolate(x.unsqueeze(0), size=32).squeeze(0) #img = img - np.array([0.281])            img = img / np.array([0.352])
        trans = transforms.Compose([transforms.ToTensor(), refu])
        cfg["imCh"] = 1

    ntrain,down=cfg["ntrain"],True
    def loadStore(isTrain,ndat):
        nonlocal  cdat
        trainset = cdat(root="data/", train=isTrain, download=down,transform=trans)
        train_dataset = torch.utils.data.DataLoader(trainset, batch_size=ndat, num_workers=4)  # cfg["batchSize"]
        ds = next(iter(train_dataset))
        X, Y = ds[0].clone().numpy(), ds[1].clone().numpy()
        #normA = lambda bx: (bx - np.min(bx)) / (np.max(bx) - np.min(bx) + 1e-10)  # "Normalizing - necessary if no BN"
        #X,Y = normA(X), normA(Y)
        ds = [X, Y]
        print("Data stats", cdat, X.shape, np.mean(X, axis=(0, 2, 3)), np.std(X, axis=(0, 2, 3)),np.max(X),np.min(X), " Data should be normalized")
        ds = sklearn.utils.shuffle(*ds)
        return  ds[0].astype(np.float16), ds[1].astype(np.int16)

    trX,trY=loadStore(True,ntrain)
    teX, teY=loadStore(False, ntrain//2)
    def cds(trX,trY,shuffle=True):
        ds=TensorDataset(torch.from_numpy(trX), torch.from_numpy(trY))
        return torch.utils.data.DataLoader(ds, batch_size=cfg["batchSize"], shuffle=shuffle, num_workers=4)
    return cds(trX, trY), cds(teX, teY,False),None
