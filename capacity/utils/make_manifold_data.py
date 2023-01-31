'''
Tool for creating manifold data from a pytorch style dataset
'''
import numpy as np
from collections import defaultdict
import torch
from torchvision.models.feature_extraction import create_feature_extractor

def make_manifold_data(dataset, sampled_classes, examples_per_class, max_class=None, seed=0):
    '''
    Samples manifold data for use in later analysis

    Args:
        dataset: PyTorch style dataset, or iterable that contains (input, label) pairs
        sampled_classes: Number of classes to sample from (must be less than or equal to
            the number of classes in dataset)
        examples_per_class: Number of examples per class to draw (there should be at least
            this many examples per class in the dataset)
        max_class (optional): Maximum class to sample from. Defaults to sampled_classes if unspecified
        seed (optional): Random seed used for drawing samples

    Returns:
        data: list containing tensors for each class of shape (examples_per_class, input dimensions)
    '''
    if max_class is None:
        max_class = sampled_classes
    assert sampled_classes <= max_class, 'Not enough classes in the dataset'
    assert examples_per_class * max_class <= len(dataset), 'Not enough examples per class in dataset'

    # Set the seed
    np.random.seed(seed)
    # Storage for samples
    sampled_data = defaultdict(list)
    # Sample the labels
    sampled_labels = np.random.choice(list(range(max_class)), size=sampled_classes, replace=False)
    # Shuffle the order to iterate through the dataset
    idx = [i for i in range(len(dataset))]
    np.random.shuffle(idx)
    # Iterate through the dataset until enough samples are drawn
    for i in idx:
        sample, label = dataset[i]
        if label in sampled_labels and len(sampled_data[label]) < examples_per_class:
            sampled_data[label].append(sample)
        # Check if enough samples have been drawn
        complete = True
        for s in sampled_labels:
            if len(sampled_data[s]) < examples_per_class:
                complete = False
        if complete:
            break
    # Check that enough samples have been found
    assert complete, 'Could not find enough examples for the sampled classes'
    # Combine the samples into batches
    data = []
    for s, d in sampled_data.items():
        data.append(torch.stack(d))
    return data

def extract_features(data, model, node_names):
    '''
    Extract intermediate features from some manifold data. 
    Args: 
        - manifold data. A list of length (num_classes) with arrays of shape (num_samples, input_dims) 
        - model: Pytorch model
        - node names: Desired node names to extract
    Returns:
        - features: Dictionary with each key corresponding to a node and each value set to a list of length (num_classes) with arrays of shape (num_samples, feature_dims) 
    '''
    extractor = create_feature_extractor(model, return_nodes=node_names)
    features = defaultdict(list) 
    for cls in data:
        feat = extractor(cls) 
        for key, val in feat.items(): 
            features[key].append(val)
    return features 
    
