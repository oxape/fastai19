from fastai.vision import *
from fastai.collab import *
from fastai.metrics import error_rate
from fastai.tabular import *
from fastai.collab import *
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

if __name__ == '__main__':
    p = Path()
    print(p.resolve())
    p = Path('../data/mnist')

    datasets = datasets.ImageFolder(p)
    item = next(iter(datasets))

    tfms = get_transforms(do_flip=False)
    data = (ImageList.from_folder(p)
            .split_by_folder(train='train', valid='valid')
            )

    data = data.label_from_folder()

    data = (data.transform(tfms)
            .databunch(bs=16, num_workers=0))
    data.show_batch(3, figsize=(6, 6), hide_axis=False)
    learner = cnn_learner(data, models.vgg16_bn, metrics=error_rate)
    learner.lr_find()
    learner.recorder.plot()
