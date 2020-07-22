from fastai.vision import *
from fastai.collab import *
from fastai.metrics import error_rate
from fastai.tabular import *
from fastai.collab import *
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torchvision
import struct

class MnistDataset(torch.utils.data.Dataset):
    """Mnist dataset."""

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample, target = self.images[idx], self.labels[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    @classmethod
    def load(cls, images_path, labels_path, valid_percent=0.25, transform=None):
        images = None
        labels = None
        with open(images_path, 'rb') as f:
            fb_data = f.read()

            offset = 0
            fmt_header = '>iiii'  # 以大端法读取4个 unsinged int32
            magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, fb_data, offset)
            offset += struct.calcsize(fmt_header)
            fmt_image = '>' + str(num_rows * num_cols) + 'B'
            images = np.empty((num_images, 1, num_rows, num_cols), dtype=np.float)  # 补一个channel=1
            for i in range(num_images):
                im = struct.unpack_from(fmt_image, fb_data, offset)
                images[i] = np.array(im).reshape((num_rows, num_cols))
                offset += struct.calcsize(fmt_image)

        with open(labels_path, 'rb') as f:
            fb_data = f.read()

            offset = 0
            fmt_header = '>ii'  # 以大端法读取2个 unsinged int32
            magic_number, num_labels = struct.unpack_from(fmt_header, fb_data, offset)
            offset += struct.calcsize(fmt_header)
            fmt_image = '>' + str(num_labels) + 'B'
            labels = struct.unpack_from(fmt_image, fb_data, offset)
            labels = np.array(labels)

        num_of_train = round(images.shape[0] * (1 - valid_percent))
        train_ds = cls(images[:num_of_train], labels[:num_of_train], transform)
        valid_ds = cls(images[num_of_train:], labels[num_of_train:], transform)
        return {'train': train_ds, 'valid': valid_ds}


if __name__ == '__main__':
    p = Path('../data/mnist')
    # path = Path('../data/mnist-raw/')
    data_transforms = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    #
    # datasets = MnistDataset.load(path / 'train-images-idx3-ubyte', path / 'train-labels-idx1-ubyte', 0.25,
    #                              transform=data_transforms['train'])

    datasets = datasets.ImageFolder(p)
    item = next(iter(datasets))

    tfms = get_transforms(do_flip=False)
    data = (ImageList.from_folder(p)
            .split_by_folder(train='train', valid='valid')
            )

    data = data.label_from_folder()

    data = data.transform(tfms)

    data = data.databunch(bs=16, num_workers=0)

    data.normalize()

    data.show_batch(3, figsize=(6, 6), hide_axis=False)
    learner = cnn_learner(data, models.vgg16_bn, metrics=error_rate)
    learner.lr_find()
    learner.recorder.plot()
