import torch
import torch.nn.functional as F
from typing import List
from nltk import tokenize

def get_shape(arr) -> list:
    if type(arr) == tuple:
        return [len(arr)] + get_shape(arr[0])
    elif type(arr) == torch.Tensor:
        return [arr.size()]
    
def same_content(arr1, arr2) -> bool:
    same = True
    if type(arr1) == tuple:
        for i in range(len(arr1)):
            same = same and same_content(arr1[i], arr2[i])
    elif type(arr1) == torch.Tensor:
        same = torch.equal(arr1, arr2)
    return same

def sentence_split(text: str) -> List[str]:
    return tokenize.sent_tokenize(text)


# Function taken from https://github.com/oneal2000/MIND/blob/main/utils/gen.py#L93
# Original authors: Weihang Su et al. (2024)
def get_pe(logit, id_, start_at):
    probabilities = F.softmax(logit, dim=2)
    log_probabilities = torch.log(probabilities)
    entropy = -probabilities * log_probabilities
    entropy_sum = torch.sum(entropy, dim=-1)

    pl = []
    el = []
    for i, idx in enumerate(id_[1:]):
        if i < start_at - 1:
            continue
        pl.append(probabilities[0][i][idx].item())
        el.append(entropy_sum[0][i].item())
    return pl, el