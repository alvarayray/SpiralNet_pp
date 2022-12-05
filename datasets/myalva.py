import os
import os.path as osp
from glob import glob

import torch
from torch_geometric.data import InMemoryDataset, extract_zip
from utils.read import read_mesh, read_mesh_obj

from tqdm import tqdm



class MyAlva(InMemoryDataset):

    def __init__(self,
                 root,
                 train=True,
                 split='interpolation',
                 transform=None,
                 pre_transform=None):

        self.split = split

        if not osp.exists(osp.join(root, 'processed', self.split)):
            os.makedirs(osp.join(root, 'processed', self.split))

        super().__init__(root, transform, pre_transform)
        
        path = self.processed_paths[0] if train else self.processed_paths[1]

        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return None

    @property
    def processed_file_names(self):
        if self.split == 'interpolation':
            return [
                osp.join(self.split, 'training.pt'),
                osp.join(self.split, 'test.pt')
            ]
        else:
            raise RuntimeError(
                ('Expected the split of interpolation, but'
                 ' found {}').format(self.split))

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download COMA_data.zip from {} and '
            'move it to {}'.format(self.url, self.raw_dir))

    def process(self):
        print('Processing...')
        fps = glob(osp.join(self.raw_dir, '*.obj'))

        if len(fps) == 0:
            raise NotImpletementedError
            # extract_zip(self.raw_paths[0], self.raw_dir, log=False)
            # fps = glob(osp.join(self.raw_dir, '*/*/*.ply'))

        train_data_list, test_data_list = [], []
        for idx, fp in enumerate(tqdm(fps)):
            data = read_mesh_obj(fp)
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if self.split == 'interpolation':
                if (idx % 100) < 10:
                    test_data_list.append(data)
                else:
                    train_data_list.append(data)
            else:
                raise RuntimeError((
                    'Expected the split of interpolation or extrapolation, but'
                    ' found {}').format(self.split))

        torch.save(self.collate(train_data_list), self.processed_paths[0])
        #TODO what is self.collate doing?
        torch.save(self.collate(test_data_list), self.processed_paths[1])