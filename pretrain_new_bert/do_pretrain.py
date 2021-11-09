# coding: utf-8
# Name:     do_pretrain
# Author:   dell
# Data:     2021/11/8
"""
transformers-4.12.3
torch-1.5.0
torchvision=0.6.0
"""

import os
import torch
import random
import warnings
import numpy as np
from argparse import ArgumentParser

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import BertTokenizer, BertConfig, BertForMaskedLM
from transformers.trainer_utils import get_last_checkpoint
from transformers import TextDataset


# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)  # 为cpu分配随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为gpu分配随机种子
        torch.cuda.manual_seed_all(seed)  # 若使用多块gpu，使用该命令设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmard = False


def main():
    parser = ArgumentParser()

    parser.add_argument("--pretrain_data_path", type=str, default="./pretrain_data/preprocessed_data.txt")
    parser.add_argument("--pretrain_model_path", type=str, default="./ckpt/bert-base-chinese")
    parser.add_argument("--data_caches", type=str, default="./caches")
    parser.add_argument("--vocab_path", type=str, default="./pretrain_data/vocab.txt")
    parser.add_argument("--config_path", type=str, default="./pretrain_data/config.json")
    parser.add_argument("--checkpoint_save_path", type=str, default="./ckpt/checkpoint")
    parser.add_argument("--save_path", type=str, default="./ckpt/bert-base-patent")
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--max_seq_len", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=5)          # 限制checkpoints的数量，最多5个

    # python通过调用warnings模块中定义的warn()函数来发出警告，我们可以通过警告过滤器进行控制是否发出警告消息。
    warnings.filterwarnings("ignore")
    args = parser.parse_args()

    setup_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path))

    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, model_max_length=args.max_seq_len)
    bert_config = BertConfig.from_pretrained(args.config_path)
    model = BertForMaskedLM(config=bert_config)
    model = model.to(device)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(
        seed=args.seed,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        output_dir=args.checkpoint_save_path,
        learning_rate=args.learning_rate,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size
    )

    print("=========loading TextDateset=========")
    dataset = TextDataset(tokenizer=tokenizer, block_size=args.max_seq_len, file_path=args.pretrain_data_path)
    print("=========TextDateset loaded =========")

    trainer = Trainer(model, args=training_args, train_dataset=dataset, data_collator=data_collator)

    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("=========training=========")
        train_result = trainer.train()
    print(train_result)
    trainer.save_model(args.save_path)
    tokenizer.save_vocabulary(args.save_path)



if __name__ == "__main__":
    main()


