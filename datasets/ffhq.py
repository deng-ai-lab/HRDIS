import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler, Subset
from .lmdb_dataset import LMDBDataset
from glob import glob
from torchvision.datasets import VisionDataset
from PIL import Image



def get_ffhq_dataset(root, split, transform='default', subset=-1, **kwargs):
    if transform == 'default':
        transform = transforms.ToTensor()
    # dset = LMDBDataset(root, 'ffhq', split, transform)
    # if isinstance(subset, int) and subset > 0:
    #     dset = Subset(dset, list(range(subset)))
    # else:
    #     assert isinstance(subset, list)
    #     dset = Subset(dset, subset)
    # return dset
    dset = FFHQDataset(root, transform)

    return dset


def get_ffhq_loader(dset, *, batch_size, num_workers, shuffle, drop_last, pin_memory, **kwargs):
    sampler = DistributedSampler(dset, shuffle=shuffle, drop_last=drop_last)
    loader = DataLoader(
        dset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, sampler=sampler, pin_memory=pin_memory, persistent_workers=True
    )
    return loader


class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        target = 0
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, target, {'index': index}
