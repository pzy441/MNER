import numpy as np


def ner_evaluate(y_pred, y_true, id2label):
    token_acc_list = []
    total_correct = 0.
    total_pred = 0.
    total_true = 0.
    for pred, true in zip(y_pred, y_true):
        token_acc_list += [a == b for (a, b) in zip(pred, true)]
        pred_entity_list = set(get_entities(pred))
        true_entity_list = set(get_entities(true))
        total_correct += len(pred_entity_list & true_entity_list)
        total_pred += len(pred_entity_list)
        total_true += len(true_entity_list)

    p = total_correct / total_pred if total_correct > 0 else 0
    r = total_correct / total_true if total_correct > 0 else 0
    f1 = 2 * p * r / (p + r) if total_correct > 0 else 0
    acc = np.mean(token_acc_list)
    return acc, f1, p, r


def get_entities(tag_list):
    temp_list = ["O", "[SEP]", "[CLS]"]
    entity_list = []
    cur_entity_type, cur_entity_start = None, None
    for i, tag in enumerate(tag_list):
        if tag in temp_list and cur_entity_type is not None:
            entity = (cur_entity_type, cur_entity_start, i - 1)
            entity_list.append(entity)
            cur_entity_type, cur_entity_start = None, None
        elif tag not in temp_list:
            temp = tag.split("-")
            if len(temp) != 2:
                print(tag)
                raise RuntimeError("tag error")
            cur_tag_class, cur_tag_type = temp
            if cur_entity_type is None and cur_tag_class == "B":
                cur_entity_type, cur_entity_start = cur_tag_type, i
            elif cur_tag_class == "B":
                entity = (cur_entity_type, cur_entity_start, i - 1)
                entity_list.append(entity)
                cur_entity_type, cur_entity_start = cur_tag_type, i
            elif cur_tag_class == "I" and cur_tag_type != cur_entity_type:
                entity = (cur_entity_type, cur_entity_start, i - 1)
                entity_list.append(entity)
                cur_entity_type, cur_entity_start = None, None

    if cur_entity_type is not None:
        entity = (cur_entity_type, cur_entity_start, len(tag_list))
        entity_list.append(entity)
    return entity_list


if __name__ == "__main__":
    _tag_list = ["O", "O", "O", "B-PER", "I-PER", "I-PER", "I-LOC", "B-MISC", "I-PER", "O"]
    print(get_entities(_tag_list))
