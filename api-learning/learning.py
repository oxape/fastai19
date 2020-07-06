from fastai.vision import *
from fastai.collab import *
from fastai.metrics import error_rate

if __name__ == '__main__':
    p = Path()
    print(p.resolve())
    p = Path('../data/mnist')
    tfms = get_transforms(do_flip=False)
    data = (ImageList.from_folder(p)
            .split_by_folder(train='train', valid='valid')
            .label_from_folder()
            .transform(tfms)
            .databunch(bs=16, num_workers=0))
    data.show_batch(3, figsize=(6, 6), hide_axis=False)
    learner = cnn_learner(data, models.vgg16_bn, metrics=error_rate)
    learner.lr_find()
    learner.recorder.plot()
