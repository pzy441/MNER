import argparse
import configparser
import logging
import os
import json
from datetime import datetime

import torch
import torchvision

from torch.utils.data import DataLoader, RandomSampler
from tqdm import trange

import resnet.resnet as resnet
from resnet.resnet_utils import myResnet

from dataset.dataset import MNERDataset
from model.mner_model import TempBert
from trainer.optimization import BertAdam
from trainer.tokenization import BertTokenizer
from trainer.trainer import MNERTrainer
from utils.file_name import config_name
from utils.labels import labels_list, aux_labels_list
from utils.seed import set_seed

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # 模型和数据
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    # 其他参数
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--od_max_length", default=32, type=int)
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=30, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=32,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.(>=1)")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--mm_model', default='TempBert',
                        help='model name')
    parser.add_argument('--layer_num1', type=int, default=1,
                        help='number of txt2img layer')
    parser.add_argument('--layer_num2', type=int, default=1,
                        help='number of img2txt layer')
    parser.add_argument('--fine_tune_cnn', action='store_true',
                        help='fine tune pre-trained CNN if True')
    parser.add_argument('--resnet_root', default='./resnet',
                        help='path the pre-trained cnn models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='crop size of image')
    parser.add_argument('--path_image', default='../pytorch-pretrained-BERT/twitter_subimages/',
                        help='path to images')
    # loss 的系数
    parser.add_argument('--lamb', default=0.45, type=float)
    parser.add_argument('--temp', type=float, default=0.14,
                        help="parameter for CL training")
    parser.add_argument('--temp_lamb', type=float, default=0.2,
                        help="parameter for CL training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    hyperparams = {"bert_model": args.bert_model, "max_seq_length": args.max_seq_length,
                   "od_max_length": args.od_max_length, "do_lower_case": args.do_lower_case,
                   "batch_size": args.train_batch_size, "lr": args.learning_rate, "epoch": args.num_train_epochs,
                   "warmup_proportion": args.warmup_proportion, "seed": args.seed, "layer_num1": args.layer_num1,
                   "layer_num2": args.layer_num2, "lamb": args.lamb, "temp": args.temp, "temp_lamb": args.temp_lamb}
    # output dir
    args.output_dir += str(datetime.now()).strip("'").split(".")[0].replace(" ", "_") + "/"

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    json.dump(hyperparams, open(args.output_dir + config_name, "w"))
    set_seed(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare model
    try:
        from torch.hub import _get_torch_home
        torch_cache_home = _get_torch_home()
    except ImportError:
        torch_cache_home = os.path.expanduser(
            os.getenv('TORCH_HOME', os.path.join(
                os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
    default_cache_path = os.path.join(torch_cache_home, 'pytorch_pretrained_bert')
    try:
        from pathlib import Path
        PYTORCH_PRETRAINED_BERT_CACHE = Path(
            os.getenv('PYTORCH_PRETRAINED_BERT_CACHE', default_cache_path))
    except (AttributeError, ImportError):
        PYTORCH_PRETRAINED_BERT_CACHE = os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                                  default_cache_path)
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), "temp")

    mner_model = TempBert.from_pretrained(args.bert_model,
                                          cache_dir=cache_dir,
                                          layer_num1=args.layer_num1,
                                          layer_num2=args.layer_num2,
                                          num_labels=len(labels_list) + 1,
                                          auxnum_labels=len(aux_labels_list) + 1,
                                          temp=args.temp,
                                          temp_lamb=args.temp_lamb,
                                          lamb=args.lamb)
    # faster-RCNN
    od_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # resnet
    net = getattr(resnet, 'resnet152')()
    net.load_state_dict(torch.load(os.path.join("./resnet", 'resnet152.pth')))
    img_encoder = myResnet(net, args.fine_tune_cnn, device)
    img_encoder.eval()
    img_encoder.to(device)

    output_model_file = os.path.join(args.output_dir, "config.json")
    output_config_file = os.path.join(args.output_dir, "pytorch_model.bin")
    output_encoder_file = os.path.join(args.output_dir, "pytorch_encoder.bin")

    num_train_optimization_steps = None
    if args.do_train:
        train_data = MNERDataset(args.task_name, args.max_seq_length, "train", args.data_dir,
                                 tokenizer, od_model, args.od_max_length, device)
        num_train_optimization_steps = int(len(train_data) / args.train_batch_size) * args.num_train_epochs
        param_optimizer = list(mner_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        # 验证集
        valid_data = MNERDataset(args.task_name, args.max_seq_length, "valid", args.data_dir,
                                 tokenizer, od_model, args.od_max_length, device)

        valid_sampler = RandomSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.eval_batch_size)

        # 测试集
        test_data = MNERDataset(args.task_name, args.max_seq_length, "test", args.data_dir,
                                tokenizer, od_model, args.od_max_length, device)

        test_sampler = RandomSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        trainer = MNERTrainer(mner_model, img_encoder, args.output_dir, train_dataloader, valid_dataloader, test_dataloader,
                              optimizer, device)
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            trainer.train(epoch)
            trainer.valid(epoch)
            trainer.test(epoch)
        trainer.print_result()


if __name__ == "__main__":
    main()
