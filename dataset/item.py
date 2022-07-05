class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class MMInputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, img_id, label=None, auxlabel=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.img_id = img_id
        self.label = label
        self.auxlabel = auxlabel


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class MMInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, added_input_mask, segment_ids, img_feat, label_id, auxlabel_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.added_input_mask = added_input_mask
        self.segment_ids = segment_ids
        self.img_feat = img_feat
        self.label_id = label_id
        self.auxlabel_id = auxlabel_id


class InputItem(object):
    def __init__(self, input_ids, input_mask, segment_ids, img_feat,
                 od_input_ids, od_input_mask, od_segment_ids, od_boxes,
                 label_ids, labels_record, auxlabel_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

        self.img_feat = img_feat
        self.od_input_ids = od_input_ids
        self.od_input_mask = od_input_mask
        self.od_segment_ids = od_segment_ids
        self.od_boxes = od_boxes

        self.label_ids = label_ids
        self.labels_record = labels_record
        self.auxlabel_ids = auxlabel_ids
