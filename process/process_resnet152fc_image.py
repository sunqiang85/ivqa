# convert the image into the last input feature to fc of resnet 152

from CocoDataset import CocoImages
import h5py
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.models as models
from tqdm import tqdm
import config
import utils


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet152(pretrained=True)
        self.model.fc = nn.Linear(2048, 2048)
        torch.nn.init.eye(self.model.fc.weight)

        # def save_output(module, input, output):
        #     self.buffer = output
        # self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        return self.model(x)
        # return self.buffer


def create_coco_loader(path):
    transform = utils.get_transform(config.image_size, config.central_fraction)
    dataset = CocoImages(path, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.preprocess_batch_size,
        num_workers=config.data_workers,
        shuffle=False,
        pin_memory=True,
    )
    return data_loader


def preprocess(input_path, out_path):
    cudnn.benchmark = True
    device = torch.device(config.device_id)
    net = Net()
    net.to(device)
    net.eval()

    loader = create_coco_loader(input_path)
    features_shape = (
        len(loader.dataset),
        config.output_features,
    )

    with h5py.File(out_path, libver='latest') as fd:
        features = fd.create_dataset('features', shape=features_shape, dtype='float16')
        coco_ids = fd.create_dataset('ids', shape=(len(loader.dataset),), dtype='int32')
        with torch.no_grad():
            i = j = 0
            for ids, imgs in tqdm(loader):
                imgs = imgs.to(device)
                out = net(imgs)
                j = i + imgs.size(0)
                features[i:j] = out.data.cpu().numpy().astype('float16')
                coco_ids[i:j] = ids.numpy().astype('int32')
                i = j


if __name__ == '__main__':
    input_path = config.img_dir + "/train2014"
    out_path = config.coco_dir + "/res152fc/train2014_image_feature.h5"
    preprocess(input_path, out_path)

    input_path = config.img_dir + "/val2014"
    out_path = config.coco_dir + "/res152fc/val2014_image_feature.h5"
    preprocess(input_path, out_path)

    input_path = config.img_dir + "/test2015"
    out_path = config.coco_dir + "/res152fc/test2015_image_feature.h5"
    preprocess(input_path, out_path)
