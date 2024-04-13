import torch

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