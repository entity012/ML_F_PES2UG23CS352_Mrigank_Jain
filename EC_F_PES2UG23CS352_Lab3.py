# student_lab.py
import torch
import math

def get_entropy_of_dataset(tensor: torch.Tensor):
    """
    Calculate the entropy of the dataset based on target column (last column).
    """
    # Extract target column
    target = tensor[:, -1]
    classes, counts = torch.unique(target, return_counts=True)
    total = target.size(0)

    # Probabilities
    probs = counts.float() / total

    # Entropy: -Σ p * log2(p)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-9))  # add epsilon to avoid log(0)
    return entropy.item()


def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    """
    Compute average information (weighted entropy) of given attribute.
    """
    total = tensor.size(0)
    values, counts = torch.unique(tensor[:, attribute], return_counts=True)

    avg_info = 0.0
    for v, count in zip(values, counts):
        subset = tensor[tensor[:, attribute] == v]
        subset_entropy = get_entropy_of_dataset(subset)
        weight = count.item() / total
        avg_info += weight * subset_entropy

    return avg_info


def get_information_gain(tensor: torch.Tensor, attribute: int):
    """
    Information Gain = Entropy(S) - Avg_Info(attribute).
    """
    total_entropy = get_entropy_of_dataset(tensor)
    avg_info = get_avg_info_of_attribute(tensor, attribute)
    info_gain = total_entropy - avg_info
    return round(info_gain, 4)


def get_selected_attribute(tensor: torch.Tensor):
    """
    Return dictionary of {attribute_index: information_gain} and best attribute.
    """
    num_attributes = tensor.size(1) - 1  # exclude target
    gains = {}

    for attr in range(num_attributes):
        gains[attr] = get_information_gain(tensor, attr)

    # Pick attribute with maximum IG
    best_attr = max(gains, key=gains.get)
    return gains, best_attr
