from pathlib import Path
import urllib.request
import json
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class RIVAL10(Dataset):

    def __init__(self, root_data: str, imagenet_root: str, train=True, return_masks=False, transform=None):
        '''
        When return_masks is True, the object mask is returned, as well as the image and label.

        Use LocalRIVAL10 with masks_dict=True to obtain attribute masks. See local_rival10.py.
        '''
        self.transform = transform
        self.root_data = root_data
        self._IMAGENET_ROOT = imagenet_root
        self._PARTITIONED_URLS_DICT_PATH = Path(root_data) / 'RIVAL10' / 'meta' / 'train_test_split_by_url.json'
        self._LABEL_MAPPINGS = Path(root_data) / 'RIVAL10' / 'meta' / 'label_mappings.json'
        self._WNID_TO_CLASS = Path(root_data) / 'RIVAL10' / 'meta' / 'wnid_to_class.json'
        self.imagenet_root = imagenet_root
        self._MASKS_TEMPLATE_ROOT = Path(imagenet_root) / 'RIVAL10' / '{}' / 'entire_object_masks/'
        self._NO_EO_MASKS_PATH = Path(imagenet_root) / 'RIVAL10' / 'meta' / 'no_eo_masks.json'
        self._S3_IMAGENET_ROOT = 'https://feizi-lab-datasets.s3.us-east-2.amazonaws.com/imagenet/'

        self.train = train
        self.return_masks = return_masks
        self.mask_root =  Path(imagenet_root) / 'RIVAL10' / '{}'.format('train' if self.train else 'test') / 'entire_object_masks/'
        self.instances = self.collect_instances()
        self.resize = transforms.Resize((224,224))

    def collect_instances(self):
        with open(self._PARTITIONED_URLS_DICT_PATH, 'r') as f:
            train_test_split = json.load(f)
        if self.train:
            urls = train_test_split['train']
        else:
            urls = train_test_split['test']

        with open(self._LABEL_MAPPINGS, 'r') as f:
            label_mappings = json.load(f)
        with open(self._WNID_TO_CLASS, 'r') as f:
            wnid_to_class = json.load(f)


        wnids = [url.split('/')[-2] for url in urls]
        inet_class_names = [wnid_to_class[wnid] for wnid in wnids]
        dcr_idx = [label_mappings[class_name][1] for class_name in inet_class_names]


        # original urls correspond to S3 links. We want to access saved ImageNet copy in cml
        local_urls = [self._IMAGENET_ROOT + url[len(self._S3_IMAGENET_ROOT):] for url in urls]

        # if we are returning masks, let's filter out any urls that don't have a mask
        # print(local_urls[:10])
        # if self.return_masks: 
        #     # mask_exists = lambda url: os.path.exists(self.mask_root + url.split('/')[-2] + '_' + url.split('/')[-1])
        #     with open(_NO_EO_MASKS_PATH, 'r') as f:
        #         failed_urls = json.load(f)
        #     new_failure = False
        #     local_urls2, dcr_idx2 = [], []
        #     for url, ind in zip(local_urls, dcr_idx):
        #         if url not in failed_urls:
        #             local_urls2.append(url)
        #             dcr_idx2.append(ind)
        #     print('Missing {} entire object masks. Fix that.'.format(len(failed_urls)))

        #     local_urls = local_urls2
        #     dcr_idx = dcr_idx2

        instances = dict({i:(url, dcr_ind) for i,(url, dcr_ind) in enumerate(zip(local_urls, dcr_idx))})
        return instances

    def __len__(self):
        return len(self.instances)

    def _base_transform(self, imgs):
        transformed_imgs = []
        i, j, h, w = transforms.RandomResizedCrop.get_params(imgs[0], scale=(0.8,1.0),ratio=(0.75,1.25))
        coin_flip = (random.random() < 0.5)
        for ind, img in enumerate(imgs):
            if self.train:
                img = TF.crop(img, i, j, h, w)

                if coin_flip:
                    img = TF.hflip(img)

            if self.transform is not None:
                img = self.transform(img)
            else:
                img = TF.to_tensor(self.resize(img))

            if img.shape[0] == 1:
                img = torch.cat([img, img, img], axis=0)

            transformed_imgs.append(img)

        return transformed_imgs

    def __getitem__(self, i):
        url, label = self.instances[i]
        img = Image.open(url)
        if img.mode == 'L':
            img = np.array(img)
            img = np.stack([img, img, img], axis=-1) 
            img = Image.fromarray(img)
        imgs = [img]
        if self.return_masks:
            wnid, fname = url.split('/')[-2:]
            mask_path = self.mask_root + wnid + '_' + fname
            mask = Image.open(mask_path)
            imgs.append(mask)

        imgs = self._base_transform(imgs)
        if self.return_masks:
            return imgs[0], imgs[1], label
        else:
            return imgs[0], label
