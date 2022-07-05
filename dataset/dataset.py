import configparser
import os
import pdb
import pickle

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.item import MMInputExample, InputItem
from utils.evaluate import get_entities
from utils.labels import labels_list, aux_labels_list, names


def readfile(filename):
    """
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    """
    f = open(filename)
    data = []  # [(sentence, label), ...]
    imgs = []  # [ 1, 2, ...]
    auxlabels = []
    sentence = []
    label = []
    auxlabel = []
    imgid = ''
    for line in f:
        if line.startswith('IMGID:'):
            imgid = line.strip().split('IMGID:')[1] + '.jpg'
            continue
        if line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                imgs.append(imgid)
                auxlabels.append(auxlabel)
                sentence = []
                label = []
                imgid = ''
                auxlabel = []
            continue
        splits = line.split('\t')
        sentence.append(splits[0])
        cur_label = splits[-1][:-1]
        if cur_label == 'B-OTHER':
            cur_label = 'B-MISC'
        elif cur_label == 'I-OTHER':
            cur_label = 'I-MISC'
        label.append(cur_label)
        auxlabel.append(cur_label[0])

    if len(sentence) > 0:
        data.append((sentence, label))
        imgs.append(imgid)
        auxlabels.append(auxlabel)

    print("The number of samples: " + str(len(data)))
    print("The number of images: " + str(len(imgs)))
    return data, imgs, auxlabels


def create_examples(lines, imgs, auxlabels, set_type):
    examples = []
    for i, (sentence, label) in enumerate(lines):
        guid = "%s-%s" % (set_type, i)
        text_a = ' '.join(sentence)
        text_b = None
        img_id = imgs[i]
        label = label
        auxlabel = auxlabels[i]
        examples.append(
            MMInputExample(guid=guid, text_a=text_a, text_b=text_b, img_id=img_id, label=label, auxlabel=auxlabel))
    return examples


def convert_examples_to_features(examples, label_list, auxlabel_list, max_seq_length, tokenizer, path_img,
                                 od_model, od_max_length, od_device):
    """Loads a data file into a list of `InputBatch`s."""
    # 从1开始
    label_map = {label: i for i, label in enumerate(label_list, 1)}
    auxlabel_map = {label: i for i, label in enumerate(auxlabel_list, 1)}

    features = []
    count = 0

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    loader = transforms.Compose([transforms.ToTensor()])
    all_images_path = []

    for (ex_index, example) in enumerate(examples):

        textlist = example.text_a.split(' ')
        label_list = example.label
        auxlabel_list = example.auxlabel
        tokens = []
        labels = []
        auxlabels = []
        # 记录 wordpiece 的位置
        labels_record = [-1]
        # tokens    [today I      am a woman ] -> [to ##day I      am a wo     #man    ]
        # labels    [O     B-PER  O  O B-PER ] -> [O  O     B-PER  O  O B-PER  B-PER   ]
        # auxlabels [O     B      O  B O     ] -> [O  O     B      O  O B      B       ]
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = label_list[i]
            auxlabel_1 = auxlabel_list[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    labels_record.append(1)
                    auxlabels.append(auxlabel_1)
                else:
                    labels.append(label_1)
                    labels_record.append(0)
                    auxlabels.append(auxlabel_1)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            labels_record = labels_record[0:(max_seq_length - 2)]
            auxlabels = auxlabels[0:(max_seq_length - 2)]
        labels_record.append(-1)
        n_tokens = []
        segment_ids = []
        label_ids = []
        auxlabel_ids = []
        n_tokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])
        auxlabel_ids.append(auxlabel_map["[CLS]"])
        for i, token in enumerate(tokens):
            n_tokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
            auxlabel_ids.append(auxlabel_map[auxlabels[i]])
        n_tokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])
        auxlabel_ids.append(auxlabel_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(n_tokens)
        input_mask = [1] * len(input_ids)

        # padding
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            labels_record.append(0)
            auxlabel_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(auxlabel_ids) == max_seq_length

        image_name = example.img_id
        image_path = os.path.join(path_img, image_name)
        all_images_path.append(image_path)

        if not os.path.exists(image_path):
            print(image_path)
        try:
            image = image_process(image_path, loader)
            image_feat = image_process(image_path, transform)
        except :
            count += 1
            image_path_fail = os.path.join(path_img, '17_06_4705.jpg')
            image = image_process(image_path_fail, loader)
            image_feat = image_process(image_path_fail, transform)
        _, h, w = image.size()
        od_model.to(od_device)
        od_model.eval()
        image = image.to(od_device)
        od_tokens, _od_boxes = image_od(image, od_model, 0.95)
        with torch.no_grad():
            _od_boxes = _od_boxes / torch.tensor([w, h, w, h], dtype=torch.float, device=od_device)
        tokenized = []
        positions_list = []  # 1 表示对象第一个单词地方 0 表示后续的单词

        for od_token in od_tokens:
            positions_list.append(1)
            tokenize = tokenizer.tokenize(od_token)
            tokenized.extend(tokenize)
            for _ in range(1, len(tokenize)):
                positions_list.append(0)

        if len(tokenized) > od_max_length - 2:
            tokenized = tokenized[0:(od_max_length - 2)]
            positions_list = positions_list[0:(od_max_length - 2)]

        tokenized = ["[CLS]"] + tokenized + ["[SEP]"]
        od_boxes = torch.zeros(od_max_length, 4)
        index_of_positions = 0
        for idx, item in enumerate(positions_list, 1):
            if item:
                od_boxes[idx] = _od_boxes[index_of_positions]
                index_of_positions += 1

        od_input_ids = tokenizer.convert_tokens_to_ids(tokenized)
        od_segment_ids = [0] * od_max_length

        od_input_mask = [1] * len(od_input_ids)
        while len(od_input_ids) < od_max_length:
            od_input_ids.append(0)
            od_input_mask.append(0)

        assert len(od_input_ids) == od_max_length
        assert len(od_input_mask) == od_max_length
        features.append(
            InputItem(input_ids, input_mask, segment_ids, image_feat,
                      od_input_ids, od_input_mask, od_segment_ids, od_boxes,
                      label_ids, labels_record, auxlabel_ids))

    print('the number of problematic samples: ' + str(count))
    return features


def image_process(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image


def image_od(image, od_model, valid_score):
    with torch.no_grad():
        predicts = od_model([image])
    boxes = predicts[0]["boxes"]
    od_tokens = [names[str(item.item())] for item in predicts[0]["labels"]]
    scores = predicts[0]["scores"]
    valid_count = 0
    for score in scores:
        if score > valid_score:
            valid_count += 1
    od_tokens = od_tokens[:valid_count]
    boxes = boxes[:valid_count]
    return od_tokens, boxes


def convert_to_tensor(features, device='cpu'):
    input_ids_tensor = torch.tensor([f.input_ids for f in features], dtype=torch.long).squeeze()
    input_mask_tensor = torch.tensor([f.input_mask for f in features], dtype=torch.long).squeeze()
    segment_ids_tensor = torch.tensor([f.segment_ids for f in features], dtype=torch.long).squeeze()
    img_feat_tensor = torch.stack([f.img_feat for f in features]).squeeze()
    label_ids_tensor = torch.tensor([f.label_ids for f in features], dtype=torch.long).squeeze()
    labels_record_tensor = torch.tensor([f.labels_record for f in features], dtype=torch.long).squeeze()
    auxlabel_ids_tensor = torch.tensor([f.auxlabel_ids for f in features], dtype=torch.long).squeeze()

    od_input_ids_tensor = torch.tensor([f.od_input_ids for f in features], dtype=torch.long).squeeze()
    od_input_mask_tensor = torch.tensor([f.od_input_mask for f in features], dtype=torch.long).squeeze()
    l, d = features[0].od_boxes.size()
    boxes_tensor = torch.zeros(len(features), l, d, dtype=torch.float)
    for i, feature in enumerate(features):
        boxes_tensor[i] = feature.od_boxes
    boxes_tensor = boxes_tensor.squeeze()
    od_segment_ids_tensor = torch.tensor([f.od_segment_ids for f in features], dtype=torch.long).squeeze()
    return input_ids_tensor, input_mask_tensor, segment_ids_tensor, img_feat_tensor, od_input_ids_tensor, od_input_mask_tensor, od_segment_ids_tensor, boxes_tensor, label_ids_tensor, labels_record_tensor, auxlabel_ids_tensor

    # [input_ids_tensor], [input_mask_tensor], [segment_ids_tensor], [od_input_ids_tensor], [
    # od_input_mask_tensor], [od_segment_ids_tensor], [boxes_tensor], [label_ids_tensor], [auxlabel_ids_tensor]


def collate_fn(batch):
    return batch


def valid_evaluate_func(examples, max_seq_length):
    # ...
    all_entity_num = 0
    for (ex_index, example) in enumerate(examples):
        label_list = example.label
        if len(label_list) >= max_seq_length - 1:
            label_list = label_list[0:(max_seq_length - 2)]
        all_entity_num += len(get_entities(label_list))
    pdb.set_trace()


class MNERDataset(Dataset):
    def __init__(self, task_name, max_seq_length, dataset_type, data_dir, tokenizer, od_model,
                 od_max_length, od_device):
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.dataset_type = dataset_type
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.od_model = od_model
        self.od_max_length = od_max_length
        self.od_device = od_device

        self.root_dir = "/".join(os.path.realpath(__file__).split("/")[:-2])
        self.image_dir = self.root_dir + "/" + self.data_dir
        self.save_data_name = os.path.join(self.image_dir, self.dataset_type + ".dset")
        config_path = os.path.join(self.root_dir, "config.ini")
        config = configparser.ConfigParser()
        config.read(config_path, encoding="utf-8")
        if self.task_name == "twitter2017":
            self.path_image = config.get("path", "twitter2017_images_path")
        elif self.task_name == "twitter2015":
            self.path_image = config.get("path", "twitter2015_images_path")

        data, images, aux_labels = readfile(self.image_dir + "/" + self.dataset_type + ".txt")
        self.len = len(data)
        if len(data) > 0:
            print(data[0], images[0], aux_labels[0])
        examples = create_examples(data, images, aux_labels, self.dataset_type)
        # valid_evaluate_func(examples, max_seq_length)
        if os.path.exists(self.save_data_name):
            print("Loading processed data")
            with open(self.save_data_name, "rb") as f:
                self.features = pickle.load(f)
        else:
            self.features = convert_examples_to_features(examples, labels_list, aux_labels_list, self.max_seq_length,
                                                         self.tokenizer, self.path_image, self.od_model,
                                                         self.od_max_length,
                                                         self.od_device)
            print('Dumping data')
            with open(self.save_data_name, 'wb') as f:
                pickle.dump(self.features, f)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return convert_to_tensor([self.features[index]])
