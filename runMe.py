import torch
from torch.utils.data import Dataset,TensorDataset
import models as clModel,dutils
import cladec
import numpy as np,os
import torch.cuda.amp as tca
from matplotlib import pyplot as plt

def showPics(pics, mtit="", tits =None,nrows = 16,ncols=12,fname="expPics"):
    picPerFig = nrows * ncols
    for i in range(max(1,pics.shape[0] // picPerFig)):
        fig = plt.figure(figsize=(20,30))
        wm = plt.get_current_fig_manager()
        wm.resize(*wm.window.maxsize())  # wm.window.state('zoomed')
        fig.suptitle(mtit, fontsize=8)
        for j in range(picPerFig):
            if i * picPerFig + j == pics.shape[0]: break
            ax1 = plt.subplot(nrows, (picPerFig - 1) // nrows + 1, j + 1)
            if not tits is None:
               tit=ax1.set_title(str(tits[i*picPerFig + j]), fontsize=10)
               plt.setp(tit, color='black')
            cpic=pics[i * picPerFig + j].squeeze()
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.axis('off')
            cpic = 1-(cpic-np.min(cpic))/(np.max(cpic)-np.min(cpic)+1e-10)
            plt.imshow(cpic.astype(np.float32),cmap='Greys')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(fname+str(i)+ ".png")
        plt.close()
        if i * picPerFig + j >= pics.shape[0] - 1: break

def trainOne():
    dummy=False
    #dummy = True
    cfg={ 'ds': ('Fash', 10),  #Dataset either  ('Ci100', 100) or ('Ci10', 10)
          'batchSize': 128, 'opt': ('S', 1 if dummy else 64, 0.1, 0.0001), #optimizer settings
          'layInd':-2, #Layer to explain (from last layer back, ie. -1 is last)
          'alpha': 0.01, #tradeoff parameter reconstruction vs. classification loss
          'ntrain': 500 if dummy else 60000}
    print("Executing config",cfg)
    cfg["num_classes"]=cfg["ds"][1]
    #Get Data
    print("Get dataset")
    train_dataset, val_dataset,norm=dutils.getFullDS(cfg)

    # Train and save non-reflective Model
    classifier, lcfg = clModel.getclassifier(cfg,  train_dataset, val_dataset, norm=norm)
    print("Classifier Accuracy",lcfg)

    acts = []
    def get_activation():
        def hook(model, input, output):
            nonlocal acts
            acts.append(output.detach().cpu())
        return hook
    laytoExp = classifier.allExplainLays[cfg["layInd"]]
    laytoExp.register_forward_hook(get_activation())
    def getActs(ds):
        nonlocal acts
        acts=[]
        X,y=[],[]
        for i, data in enumerate(ds):
            with tca.autocast():
                dsx, dsy = data[0].cuda(), data[1].cuda()
                X.append(data[0])
                y.append(data[1])
                classifier(dsx)
        X=torch.cat(X,dim=0)
        y=torch.cat(y, dim=0)
        conacts=torch.cat(acts,dim=0)
        dsact=TensorDataset(X,y,conacts)
        return torch.utils.data.DataLoader(dsact, batch_size=cfg["batchSize"], shuffle=True, num_workers=4)
    trds=getActs(train_dataset)

    # Train ClaDec
    cladecNet,ccfg=  cladec.getClaDec(cfg,classifier,norm,trds)
    print("ClaDec Final loss", ccfg)

    #Explain
    teds = getActs(val_dataset)
    allimgs=[]
    foldname = "imgs_Lay_" + str(cfg["layInd"]) + "_alpha_" + str(cfg["alpha"]) + "/"
    os.makedirs(foldname, exist_ok=True)
    for i, data in enumerate(teds):
        with tca.autocast():
            dsx, dsy,dsact = data[0].cuda(), data[1].cuda(), data[2].cuda()
            output = cladecNet(dsact)
            for j in range(data[0].shape[0]):
                allimgs.append(data[0][j].numpy())
                allimgs.append(output[j].detach().cpu().numpy())
                if len(allimgs)==16*11: break
            if len(allimgs) == 16 * 11: break
    showPics(np.array(allimgs),fname=foldname+"Vis_Orig_AndClaDec")

if __name__ == '__main__':
    trainOne()
