from glob import glob
from src.datamodules.colon_datamodule import ColonDataset, ColonDataModule
from src.datamodules.colon_test2_datamodule import ColonTestDataset
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

from imgaug import augmenters as iaa
import imgaug as ia
import torch
from cv2 import cv2

class HarvardDataset(Dataset):
    def __init__(self, pair_list, transform=None, pre_transform=None):
        self.pair_list = pair_list
        self.transform = transform
        self.pre_transform = pre_transform

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        pair = self.pair_list[idx]
        image = cv2.imread(pair[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = pair[1]

        # 512 x 512변경 후 하기
        image = self.pre_transform(image=image)
        if self.transform:
            image = self.transform(image=image)

        return torch.from_numpy(image.copy()).permute(2,0,1), label

class HarvardDataset_pd(ColonDataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.labels = df["label"].values


class HarvardDataModule(ColonDataModule):
    """
    class 0: 2869
    class 1: 8828
    class 2: 7235
    class 3: 3090
    """

    def __init__(
        self,
        data_dir: str = "./",
        img_size: int = 256,
        num_workers: int = 4,
        batch_size: int = 16,
        pin_memory=False,
        drop_last=False,
        data_name="harvard",
        data_ratio=1.0,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        resize_value = 256 if self.hparams.img_size == 224 else 456

        sometimes = lambda aug: iaa.Sometimes(0.2, aug)

        self.pre_transform = iaa.Resize({"height": 512, "width": 512},
                                         interpolation='nearest')

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

            train_set, valid_set = prepare_prostate_harvard_data(stage="train")
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

            self.train_dataset = HarvardDataset(train_set, self.train_transform, self.pre_transform)
            self.valid_dataset = HarvardDataset(valid_set, self.test_transform, self.pre_transform)

        else:
            test_set = prepare_prostate_harvard_data(stage="test")
            self.test_dataset = HarvardDataset(test_set, self.test_transform, self.pre_transform)

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


def prepare_prostate_harvard_data(stage="train"):
    def load_data_info(pathname):
        file_list = glob(pathname)

        label_list = [
            int(file_path.split("_")[-1].split(".")[0]) for file_path in file_list
        ]

        return list(zip(file_list, label_list))
    # 원본 파일 163.152.183.213 jh/data 경로에 있음

    data_root_dir = "data path"
    data_root_dir_train = f"{data_root_dir}/patches_train_750_v0/"
    data_root_dir_valid = f"{data_root_dir}/patches_validation_750_v0/"
    data_root_dir_test = f"{data_root_dir}/patches_test_750_v0/"

    if stage == "train":
        train_set_111 = load_data_info(f"{data_root_dir_train}/ZT111*/*.jpg")
        train_set_199 = load_data_info(f"{data_root_dir_train}/ZT199*/*.jpg")
        train_set_204 = load_data_info(f"{data_root_dir_train}/ZT204*/*.jpg")
        valid_set = load_data_info(f"{data_root_dir_valid}/ZT76*/*.jpg")
        train_set = train_set_111 + train_set_199 + train_set_204

        return train_set, valid_set

    elif stage == "test":
        test_set = load_data_info(f"{data_root_dir_test}/patho_1/*/*.jpg")
        return test_set
