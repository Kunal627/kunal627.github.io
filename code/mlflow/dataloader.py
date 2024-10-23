from imports import *

# Define a custom Dataset class
class ECGChapmanDataset(Dataset):
    def __init__(self, datadir, filename, num_classes, seqlen, transform=None, perm=None, leadidx=None):
        self.folder = "ecgdata"  #change this to get the actual ecg path
        self.seqlen = seqlen
        self.datadir = datadir
        self.num_classes = num_classes
        self.transform = transform
        self.perm = perm
        self.leadidx = leadidx
        self.samples = pd.read_csv(os.path.join(self.datadir, filename))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ecgfilepath = os.path.join(self.datadir, self.folder, self.samples.iloc[idx,0]) + ".csv"
        ecgdata = pd.read_csv(ecgfilepath, header=1).to_numpy()[:self.seqlen, :]
        label = torch.tensor(self.samples.iloc[idx,-1])
        one_hot_labels = torch.nn.functional.one_hot(label, num_classes=self.num_classes).float()

        if self.transform:
            ecgdata = self.transform(ecgdata)

        ecgdata = torch.tensor(ecgdata.transpose(1,0), dtype=torch.float32)

        if self.perm == "TRANS":
            ecgdata = ecgdata.permute(1,0)
        
        if self.perm =="12LEAD":
            ecgdata = ecgdata.unsqueeze(1)
            if self.leadidx is not None:
                ecgdata = ecgdata[self.leadidx , :, :]

        # get only features which are most important in explaining the class label
        # these features will be used in conjunction to the 12 lead ECG samples.
        # these features will be fed to FCNN
        features = torch.tensor(self.samples.iloc[idx,[3,5,6,8,10,11]].to_numpy(dtype=np.float32))

        return ecgdata, one_hot_labels, features
