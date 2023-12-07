from glob import glob
from sklearn.model_selection import train_test_split
from src.datamodules.colon_test2_datamodule import ColonTestDataset
from src.datamodules.colon_datamodule import ColonDataModule, ColonDataset
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

from imgaug import augmenters as iaa
import imgaug as ia
import torch
from cv2 import cv2

class UbcDataset(Dataset):
    def __init__(self, pair_list, transform=None):
        self.pair_list = pair_list
        self.transform = transform

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        pair = self.pair_list[idx]
        image = cv2.imread(pair[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = pair[1]

        if self.transform:
            image = self.transform(image=image)

        return torch.from_numpy(image.copy()).permute(2,0,1), label


class UbcDataset_pd(UbcDataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.labels = df["label"].values

class UbcDataModule(LightningDataModule):
    """
    prostate_miccai_2019_patches_690_80_step05
    class 0: 1811
    class 2: 7037
    class 3: 11431
    class 4: 292
    1284 BN, 5852 grade 3, 9682 grade 4, and 248 grade 5

    """

    def __init__(
        self,
        data_dir: str = "./",
        img_size: int = 256,
        num_workers: int = 4,
        batch_size: int = 16,
        pin_memory=False,
        drop_last=False,
        data_name="ubc",
        data_ratio=1.0,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        resize_value = 256 if self.hparams.img_size == 224 else 456

        sometimes = lambda aug: iaa.Sometimes(0.2, aug)
        self.train_transform = iaa.Sequential(
            [
                # apply the following augmenters to most images

                iaa.Resize({"height": self.hparams.img_size, "width": self.hparams.img_size},
                           interpolation='nearest'),

                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.5),  # vertically flip 50% of all images
                sometimes(iaa.Affine(
                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    shear=(-16, 16),  # shear by -16 to +16 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode='symmetric'
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(3, 11)),
                                   # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                               iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )
        self.test_transform = iaa.Resize({"height": self.hparams.img_size, "width": self.hparams.img_size},
                                         interpolation='nearest')

    @property
    def num_classes(self) -> int:
        return 4

    def setup(self, stage=None):
        if stage == "fit" or stage is None:

            train_set, valid_set, test_set = make_ubc_dataset()
            if self.hparams.data_ratio < 1.0:
                train_set = (
                    pd.DataFrame(train_set, columns=["path", "class"])
                    .groupby("class")
                    .apply(
                        lambda x: x.sample(
                            frac=self.hparams.data_ratio, random_state=42
                        )
                    )
                    .reset_index(drop=True)
                )
                train_set = list(train_set.to_records(index=False))

            self.train_dataset = UbcDataset(train_set, self.train_transform)
            self.valid_dataset = UbcDataset(valid_set, self.test_transform)

        else:
            _, _, test_set = make_ubc_dataset()
            self.test_dataset = UbcDataset(test_set, self.test_transform)


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False, # True
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
            shuffle=False,
        )

def make_ubc_dataset():
    """
    prostate_miccai_2019_patches_690_80_step05
    class 0: 1811
    class 2: 7037
    class 3: 11431
    class 4: 292
    1284 BN, 5852 grade 3, 9682 grade 4, and 248 grade 5
    """

    def _split(dataset):  # train val test 80/10/10
        train, rest = train_test_split(dataset, train_size=0.8, shuffle=False, random_state=42)
        valid, test = train_test_split(rest, test_size=0.5, shuffle=False, random_state=42)
        return train, valid, test

    data_root_dir = '/ubc data path/'
    files = glob(f"{data_root_dir}/*/*.jpg")

    data_class0 = [data for data in files if int(data.split("_")[-1].split(".")[0]) == 0]
    data_class2 = [data for data in files if int(data.split("_")[-1].split(".")[0]) == 2]
    data_class3 = [data for data in files if int(data.split("_")[-1].split(".")[0]) == 3]
    data_class4 = [data for data in files if int(data.split("_")[-1].split(".")[0]) == 4]

    train_data0, validation_data0, test_data0 = _split(data_class0)
    train_data2, validation_data2, test_data2 = _split(data_class2)
    train_data3, validation_data3, test_data3 = _split(data_class3)
    train_data4, validation_data4, test_data4 = _split(data_class4)

    label_dict = {0: 0, 2: 1, 3: 2, 4: 3}

    train_path = train_data0 + train_data2 + train_data3 + train_data4
    valid_path = (validation_data0 + validation_data2 + validation_data3 + validation_data4)
    test_path = test_data0 + test_data2 + test_data3 + test_data4

    train_label = [int(path.split(".")[0][-1]) for path in train_path]
    valid_label = [int(path.split(".")[0][-1]) for path in valid_path]
    test_label = [int(path.split(".")[0][-1]) for path in test_path]

    test_label = [label_dict[k] for k in test_label]
    train_label = [label_dict[k] for k in train_label]
    valid_label = [label_dict[k] for k in valid_label]
    train_set = list(zip(train_path, train_label))
    valid_set = list(zip(valid_path, valid_label))
    test_set = list(zip(test_path, test_label))

    # print_number_of_sample(train_set, valid_set, test_set)

    return train_set, valid_set, test_set

