import configparser
import os

import torch
import torchvision
from PIL import Image
from torchvision import transforms

from dataset.dataset import image_od


def object_detection(year_of_dataset, image_id):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    print(in_features)
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.ini")
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")

    if year_of_dataset == "2015":
        image_dir_path = config.get("path", "twitter2017_images_path")
    elif year_of_dataset == "2017":
        image_dir_path = config.get("path", "twitter2017_images_path")
    image_path = os.path.join(image_dir_path, image_id)

    # process = image_process(image_path, transform)
    process = Image.open(image_path).convert('RGB')
    loader = transforms.Compose([transforms.ToTensor()])

    process = loader(process)
    # pdb.set_trace()
    _, h, w = process.size()
    model.eval()
    _, boxes = image_od(process, model, 0.95)
    boxes_ = boxes
    boxes_ = boxes_ / torch.tensor([w, h, w, h], dtype=torch.float)
    # print(_)
    # print(boxes_)


def sort(text_emb, vision_emb):
    emb = text_emb * vision_emb
    up = torch.sum(emb, -1)
    down = torch.norm(text_emb, dim=-1) * torch.norm(vision_emb, dim=-1)
    print(torch.div(up, down))
    print(torch.div(up, down)[torch.argsort(torch.div(up, down), descending=True)])
    return torch.argsort(torch.div(up, down), descending=True)


def ana_tensor():
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    print(a)
    print(b)
    print(sort(a, b))


def ana_tensor_index():
    a = torch.ones(3, 4, requires_grad=True)
    c = a[[2, 0, 1]]
    w = 2 * torch.ones(3, 4)
    w[2] = 1
    y = (c * w).sum()
    y.backward()
    print(a.grad)


if __name__ == "__main__":
    ana_tensor_index()
# fix_seed(1)
# object_detection("2017", "O_3345.jpg")
