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
        sample = np.divide(sample, 255.0)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    @classmethod
    def load(cls, images_path, labels_path, valid_percent=0.25, transform=None, need_stats=False):
        images = None
        labels = None
        with open(images_path, 'rb') as f:
            fb_data = f.read()

            offset = 0
            fmt_header = '>iiii'  # 以大端法读取4个 unsinged int32
            magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, fb_data, offset)
            offset += struct.calcsize(fmt_header)
            fmt_image = '>' + str(num_rows * num_cols) + 'B'
            images = np.empty((num_images, num_rows, num_cols), dtype=np.float)  # 不需要补channel transforms.ToTensor会补
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
        if num_of_train != images.shape[0]:
            valid_ds = cls(images[num_of_train:], labels[num_of_train:], transform)
        else:
            valid_ds = None
        result = {'train': train_ds, 'valid': valid_ds}
        if need_stats:
            mean = images.mean()
            std = images.std()
            result['stats'] = [mean, std]
        return result


class WrapInput:
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx][0]


class WrapTarget:
    def __init__(self, y):
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.y[idx][1]

class MyBlockModel(nn.Module):

    def __init__(self, num_classes=2, init_weights=True, batch_norm=False):
        super(MyBlockModel, self).__init__()
        #cfgs = [16, 'M', 32, 'M'] #下面层数更多的网络同样的训练轮次和动态学习率比这个错误率更低
        cfgs = [16, 16, 'M', 32, 32, 'M', 32, 32, 'M']
        layers = []
        in_channels = 1
        for v in cfgs:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256), #多加了这一层的作用感觉是训练轮次和损失的曲线更平滑了
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def Block(pretrained=False, batch_norm=True, **kwargs):
    model = MyBlockModel(batch_norm=batch_norm, **kwargs)
    return model


if __name__ == '__main__':
    p = Path('../data/mnist')
    path = Path('../data/mnist-raw/')
    data_transforms = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.131], [0.308])
        ]),
        'valid': torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.131], [0.308])
        ]),
    }

    datasets = MnistDataset.load(path / 'train-images-idx3-ubyte', path / 'train-labels-idx1-ubyte',
                                 valid_percent=0.25,
                                 transform=data_transforms['train'])

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    train_ds = datasets['train']
    train_x = ItemList(WrapInput(train_ds))
    train_y = CategoryList(WrapTarget(train_ds), classes)
    train_y.c2i = {v: i for i, v in enumerate(classes)}
    train_ll = LabelList(train_x, train_y)

    valid_ds = datasets['valid']
    valid_x = ItemList(WrapInput(valid_ds))
    valid_y = CategoryList(WrapTarget(valid_ds), classes)
    valid_y.c2i = {v: i for i, v in enumerate(classes)}
    valid_ll = LabelList(valid_x, valid_y)

    data = LabelLists(path, train_ll, valid_ll).databunch(bs=16, num_workers=0)

    learner = cnn_learner(data, Block, metrics=error_rate)

    learner.lr_find()

    datasets = datasets.ImageFolder(p)
    item = next(iter(datasets))

    tfms = get_transforms(do_flip=False)
    data = (ImageList.from_folder(p)
            .split_by_folder(train='train', valid='valid'))

    data = data.label_from_folder()

    data = data.transform(tfms)

    data = data.databunch(bs=16, num_workers=0)

    # data.normalize()

    # data.show_batch(3, figsize=(6, 6), hide_axis=False)
    learner = cnn_learner(data, models.vgg16_bn, metrics=error_rate)
    learner.lr_find()
    learner.recorder.plot()
