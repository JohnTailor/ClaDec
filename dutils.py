import torchvision, torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np, sklearn
import torch.nn.functional as F
import os
import requests
import zipfile
import io
from shutil import copyfile

from torchvision.transforms.transforms import CenterCrop, Resize


gds = lambda dataset, cfg: torch.utils.data.DataLoader(
    TensorDataset(*[torch.from_numpy(x) for x in dataset]), batch_size=cfg["batchSize"]
)


def getnorm(dname):
    if dname == "Ci10":
        return (
            torch.from_numpy(
                np.array((0.4914, 0.4822, 0.4465), np.float32).reshape(1, 3, 1, 1)
            ).cuda(),
            torch.from_numpy(
                np.array((0.2023, 0.1994, 0.2010), np.float32).reshape(1, 3, 1, 1)
            ).cuda(),
        )
    elif dname == "Ci100":
        return (
            torch.from_numpy(
                np.array((0.5060725, 0.48667726, 0.4421305), np.float32).reshape(
                    1, 3, 1, 1
                )
            ).cuda(),
            torch.from_numpy(
                np.array((0.2675421, 0.25593522, 0.27593908), np.float32).reshape(
                    1, 3, 1, 1
                )
            ).cuda(),
        )
    elif dname == "Fash":
        return (
            torch.from_numpy(np.array((0.281), np.float32).reshape(1, 1, 1, 1)).cuda(),
            torch.from_numpy(np.array((0.352), np.float32).reshape(1, 1, 1, 1)).cuda(),
        )
    elif dname == "MNIST":
        return (
            torch.from_numpy(np.array((0.1307), np.float32).reshape(1, 1, 1, 1)).cuda(),
            torch.from_numpy(np.array((0.3081), np.float32).reshape(1, 1, 1, 1)).cuda(),
        )
    elif dname == "TinyImgNet":
        return (
            torch.from_numpy(
                np.array((0.4802, 0.4481, 0.3975), np.float32).reshape(3, 1, 1)
            ),
            torch.from_numpy(
                np.array((0.2302, 0.2265, 0.2262), np.float32).reshape(3, 1, 1)
            ),
        )


def getFullDS(cfg, reconstr=False):
    dname = cfg["ds"][0]
    trans = transforms.Compose([transforms.ToTensor()])
    refu = lambda x: F.interpolate(x.unsqueeze(0), size=32).squeeze(0)
    if dname == "Ci10":
        cdat = (
            torchvision.datasets.CIFAR10
        )  # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) #transform = transforms.Compose([transforms.ToTensor(), norm])
        cfg["imCh"] = 3
    elif dname == "Ci100":
        cdat = (
            torchvision.datasets.CIFAR100
        )  # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        cfg["imCh"] = 3
    elif dname == "Fash":
        cdat = torchvision.datasets.FashionMNIST
        # img = img - np.array([0.281])            img = img / np.array([0.352])
        trans = transforms.Compose([transforms.ToTensor(), refu])
        cfg["imCh"] = 1
    elif dname == "MNIST":
        cdat = torchvision.datasets.MNIST
        # trans = transforms.Compose([transforms.ToTensor(), refu, transforms.Normalize(0.1307,0.3081)])
        trans = transforms.Compose([transforms.ToTensor(), refu])
        cfg["imCh"] = 1
    elif dname == "TinyImgNet":
        cdat = tinyImgNet
        norm = getnorm(dname)
        # gamma_corr = lambda img : transforms.functional.adjust_gamma(img, gamma=, float=)
        # resize_tr = random.choice([transforms.RandomResizedCrop(56), transforms.CenterCrop(56)])
        # aug_tr = random.choice([transforms.RandomHorizontalFlip(), transforms.RandomAffine(30, translate=10, scale = [0.8])])
        # data_augs = [transforms.RandomResizedCrop(32), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(), transforms.RandomRotation(45)]
        train_trans = transforms.Compose(
            [
                transforms.ToTensor(),
                #transforms.Normalize(mean=norm[0], std=norm[1]),
                transforms.RandomResizedCrop(64),
                # transforms.CenterCrop(56),
                # transforms.Resize(32),
                transforms.RandomHorizontalFlip(),
            ]
        )
        val_trans = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.CenterCrop(56),
                #transforms.Normalize(mean=norm[0], std=norm[1]),
                # transforms.Resize(32)
            ]
        )
        cfg["imCh"] = 3

    ntrain, down = cfg["ntrain"], True
    if dname == "TinyImgNet":
        down = False
        if reconstr == True:
            trainset = cdat(
                root="data/",
                train=True,
                download=down,
                transform=transforms.Compose([transforms.ToTensor()]),
            )
            valset = cdat(
                root="data/",
                train=False,
                download=down,
                transform=transforms.Compose([transforms.ToTensor()]),
            )

        else:
            trainset = cdat(
                root="data/", train=True, download=down, transform=train_trans
            )
            valset = cdat(root="data/", train=False, download=down, transform=val_trans)
        train_loader = DataLoader(
            trainset, batch_size=cfg["batchSize"], shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            valset, batch_size=cfg["batchSize"], shuffle=False, num_workers=4
        )
        return train_loader, val_loader, None, trainset, valset

    def loadStore(isTrain, ndat):
        nonlocal cdat
        trainset = cdat(root="data/", train=isTrain, download=down, transform=trans)
        train_dataset = torch.utils.data.DataLoader(
            trainset, batch_size=ndat, num_workers=4
        )  # cfg["batchSize"]
        ds = next(iter(train_dataset))
        X, Y = ds[0].clone().numpy(), ds[1].clone().numpy()
        # normA = lambda bx: (bx - np.min(bx)) / (np.max(bx) - np.min(bx) + 1e-10)  # "Normalizing - necessary if no BN"
        # X,Y = normA(X), normA(Y)
        ds = [X, Y]
        print(
            "Data stats",
            cdat,
            X.shape,
            np.mean(X, axis=(0, 2, 3)),
            np.std(X, axis=(0, 2, 3)),
            np.max(X),
            np.min(X),
            " Data should be normalized",
        )
        ds = sklearn.utils.shuffle(*ds)
        return ds[0].astype(np.float16), ds[1].astype(np.int16)

    trX, trY = loadStore(True, ntrain)
    teX, teY = loadStore(False, ntrain // 2)

    def cds(trX, trY, shuffle=True):
        ds = TensorDataset(torch.from_numpy(trX), torch.from_numpy(trY))
        return DataLoader(
            ds, batch_size=cfg["batchSize"], shuffle=shuffle, num_workers=4
        )

    return cds(trX, trY), cds(teX, teY, False), None


def tinyImgNet(root: str, train: bool, download: bool, transform) -> Dataset:
    filter_val = train and not download

    if download:
        os.makedirs(root, exist_ok=True)
        r = requests.get(
            "http://cs231n.stanford.edu/tiny-imagenet-200.zip", stream=True
        )
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(root)

    main_dir_path = os.path.join(root, "tiny-imagenet-200")
    train_path = os.path.join(main_dir_path, "train")
    val_path = os.path.join(main_dir_path, "val")
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        raise ValueError("Train and Val paths don't exist")

    val_formatted = os.path.join(main_dir_path, "val_formatted")
    if not filter_val and not os.path.exists(val_formatted):
        with open(os.path.join(val_path, "val_annotations.txt"), "r") as f:
            img_class_map = f.readlines()
        for img_class in img_class_map:
            img_name, class_name = img_class.split("\t")[0], img_class.split("\t")[1]
            img_path = os.path.join("images", img_name)
            class_dir = os.path.join(val_formatted, class_name)
            os.makedirs(os.path.join(class_dir, "images"), exist_ok=True)
            copyfile(
                os.path.join(val_path, img_path), os.path.join(class_dir, img_path)
            )
    if train:
        return ImageFolder(train_path, transform)
    else:
        return ImageFolder(val_formatted, transform)


from torchvision.datasets.folder import DatasetFolder
import torch
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import numpy as np

class NumpyFolder(DatasetFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = np.load,
            is_valid_file: Optional[Callable[[str], bool]] = None,):
        super(NumpyFolder, self).__init__(root, loader, (".npy") if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        (act,img) = self.loader(path, allow_pickle=True)
        img = torch.Tensor(img).type(torch.float16)
        act = torch.Tensor(act).type(torch.float16)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, act