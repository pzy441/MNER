import json
import os
import pdb

import torch
from tqdm import tqdm

from ner_evaluate import evaluate, evaluate_each_class
from utils.labels import labels_list


class MNERTrainer(object):

    def __init__(self, mner_model, img_encoder, output_dir, train_dataloader, valid_dataloader, test_dataloader, optim, device):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.optim = optim
        self.mner_model = mner_model
        self.img_encoder = img_encoder
        self.device = device
        self.output_dir = output_dir
        
        self.label_map = {i: label for i, label in enumerate(labels_list, 1)}
        self.label_map[0] = "PAD"

        self.max_dev_f1 = 0.0
        self.best_dev_epoch = 0
        self.test_f1s = []

        self.mner_model.to(self.device)

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader, "train")

    def valid(self, epoch):
        self.iteration(epoch, self.valid_dataloader, "valid")

    def test(self, epoch):
        self.iteration(epoch, self.test_dataloader, "test")

    def iteration(self, epoch, dataloader, dataset_type):
        if dataset_type == "train":
            self.mner_model.train()
        else:
            self.mner_model.eval()
            result_file = self.output_dir + dataset_type + ".txt"
        y_true = []
        y_pred = []
        y_true_idx = []
        y_pred_idx = []
        for step, batch in enumerate(tqdm(dataloader, desc=dataset_type)):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, img_feats, od_input_ids, od_input_mask, od_segment_ids, od_boxes, label_ids, labels_record, auxlabel_ids = batch
            with torch.no_grad():
                imgs_f, img_mean, img_att = self.img_encoder(img_feats)
            if dataset_type == "train":

                loss = self.mner_model(input_ids, input_mask, segment_ids, img_att,
                                       od_input_ids, od_input_mask, od_segment_ids, od_boxes,
                                       label_ids, auxlabel_ids)
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
            else:
                predicts = self.mner_model(input_ids, input_mask, segment_ids, img_att,
                                           od_input_ids, od_input_mask, od_segment_ids, od_boxes)
                label_ids = label_ids.to('cpu').numpy()
                input_mask = input_mask.to('cpu').numpy()
                labels_record = labels_record.to('cpu').numpy()
                for i, mask in enumerate(input_mask):
                    temp_1 = []
                    temp_2 = []
                    tmp1_idx = []
                    tmp2_idx = []
                    for j, m in enumerate(mask):
                        if j == 0:
                            continue
                        if m:
                            if labels_record[i][j] == 1:  # self.label_map[label_ids[i][j]] != "X" and self.label_map[label_ids[i][j]] != "[SEP]":
                                temp_1.append(self.label_map[label_ids[i][j]])
                                temp_2.append(self.label_map[predicts[i][j]])
                                tmp1_idx.append(label_ids[i][j])
                                tmp2_idx.append(predicts[i][j])
                        else:
                            break
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    y_true_idx.append(tmp1_idx)
                    y_pred_idx.append(tmp2_idx)
        if dataset_type != "train":
            reverse_label_map = {label: i for i, label in enumerate(labels_list, 1)}
            acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, reverse_label_map)
            print("Overall: ", p, r, f1)
            # _acc, _f1, _p, _r = ner_evaluate(y_pred, y_true, labels_list)
            # print("my evaluate", _p, _r, _f1)
            per_f1, per_p, per_r = evaluate_each_class(y_pred_idx, y_true_idx, reverse_label_map, 'PER')
            print("Person: ", per_p, per_r, per_f1)
            loc_f1, loc_p, loc_r = evaluate_each_class(y_pred_idx, y_true_idx, reverse_label_map, 'LOC')
            print("Location: ", loc_p, loc_r, loc_f1)
            org_f1, org_p, org_r = evaluate_each_class(y_pred_idx, y_true_idx, reverse_label_map, 'ORG')
            print("Organization: ", org_p, org_r, org_f1)
            misc_f1, misc_p, misc_r = evaluate_each_class(y_pred_idx, y_true_idx, reverse_label_map, 'MISC')
            print("Miscellaneous: ", misc_p, misc_r, misc_f1)
            with open(result_file, "a+") as writer:
                writer.write("epoch: " + str(epoch) + '\n')
                writer.write("Overall: " + str(p) + ' ' + str(r) + ' ' + str(f1) + '\n')
                writer.write("Person: " + str(per_p) + ' ' + str(per_r) + ' ' + str(per_f1) + '\n')
                writer.write("Location: " + str(loc_p) + ' ' + str(loc_r) + ' ' + str(loc_f1) + '\n')
                writer.write("Organization: " + str(org_p) + ' ' + str(org_r) + ' ' + str(org_f1) + '\n')
                writer.write("Miscellaneous: " + str(misc_p) + ' ' + str(misc_r) + ' ' + str(misc_f1) + '\n')

        if dataset_type == "valid" and f1 > self.max_dev_f1:
            # model_to_save = self.mner_model.module if hasattr(self.mner_model, 'module') else self.mner_model
            # torch.save(model_to_save.state_dict(), output_model_file)
            # with open(output_config_file, 'w') as f:
            #    f.write(model_to_save.config.to_json_string())
            # model_config = self.mner_model.get_config()
            # json.dump(model_config, open(os.path.join(self.output_dir, model_config_file), "w"))
            self.max_dev_f1 = f1
            self.best_dev_epoch = epoch
        if dataset_type == "test":
            self.test_f1s.append(f1)

    def print_result(self):
        final_file = self.output_dir + "result.txt"
        with open(final_file, "w") as f:
            f.write("best result: " + str(self.test_f1s[self.best_dev_epoch]))
            f.write("best epoch: " + str(self.best_dev_epoch))
        print("**************************************************")
        print("best result " + str(self.test_f1s[self.best_dev_epoch]))
        print("best epoch " + str(self.best_dev_epoch))
        print('\n')
