# coding: utf-8
# Name:     pre_process
# Author:   dell
# Data:     2021/11/9
from tqdm import tqdm

def get_text(data_path):
    f = open(data_path, "r", encoding="utf-8")
    lines = f.readlines()
    texts = []
    for i in tqdm(range(0, len(lines)-1, 3)):
        text = lines[i].replace("\ufeff", "").split()
        # slots = lines[i+1]
        texts.append(text)
        # cur = lines[i]
        # if cur != "\n":
        #     print(i)
        #     f.close()
        #     exit()
    f.close()

    return texts

def write2file(texts, tgt):
    g = open(tgt, "w", encoding="utf-8")
    for i in tqdm(range(len(texts))):
        text = "".join(texts[i][1:])
        g.write(text + "\n")

    g.close()

if __name__ == "__main__":
    data_path = "preds.txt"

    texts = get_text(data_path)
    tgt = "original_text.txt"
    write2file(texts, tgt)

