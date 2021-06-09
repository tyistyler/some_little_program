# coding: utf-8
# Name:     do_test
# Author:   dell
# Data:     2021/6/9
from collections import defaultdict
def _bioes_tag_to_spans(tags, ignore_labels=None):
    """
    给定一个tags的lis，比如['O', 'B-singer', 'I-singer', 'E-singer', 'O', 'O']。
    返回[('singer', (1, 4))] (左闭右开区间)

    :param tags: List[str],
    :param ignore_labels: List[str], 在该list中的label将被忽略
    :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bioes_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bioes_tag, label = tag[:1], tag[2:]
        if bioes_tag in ('b', 's'):
            spans.append((label, [idx, idx]))
        elif bioes_tag in ('i', 'e') and prev_bioes_tag in ('b', 'i') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bioes_tag == 'o':
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bioes_tag = bioes_tag
    return [(span[0], (span[1][0], span[1][1] + 1))
            for span in spans
            if span[0] not in ignore_labels
            ]

if __name__ == '__main__':

    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)

    real_tags = ['O', 'O', 'B-LOC.NAM', 'I-LOC.NAM', 'E-LOC.NAM', 'O', 'B-ORG.NAM', 'I-ORG.NAM', 'I-ORG.NAM',
                 'I-ORG.NAM', 'I-ORG.NAM', 'E-ORG.NAM']
    pred_tags = ['O', 'O', 'B-PER.NOM', 'E-PER.NOM', 'O', 'O', 'B-GPE.NAM', 'E-GPE.NAM', 'B-LOC.NAM', 'I-LOC.NAM',
                 'I-LOC.NAM', 'E-LOC.NAM']
    gold_spans = _bioes_tag_to_spans(tags=real_tags)
    pred_spans = _bioes_tag_to_spans(tags=pred_tags)

    print(gold_spans)
    print(pred_spans)
    for span in pred_spans:
        if span in gold_spans:
            true_positives[span[0]] += 1
            gold_spans.remove(span)
        else:
            false_positives[span[0]] += 1
    for span in gold_spans:
        false_negatives[span[0]] += 1

    print(true_positives)
    print(false_positives)
    print(false_negatives)
