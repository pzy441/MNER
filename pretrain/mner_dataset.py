from torch.utils.data import Dataset

from dataset.dataset import readfile


class InputExample(object):
    """A single training/test example for the language model."""
    def __init__(self, uid, sent, visual_feats=None,
                 obj_labels=None, attr_labels=None,
                 is_matched=None, label=None):
        self.uid = uid
        self.sent = sent
        self.visual_feats = visual_feats
        self.obj_labels = obj_labels
        self.attr_labels = attr_labels
        self.is_matched = is_matched  # whether the visual and obj matched
        self.label = label


class MMDataset(Dataset):
    def __init__(self, path):
        self.path = path

        readfile(path, )

    def

