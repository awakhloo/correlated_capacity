'''
Tool for creating manifold data from a pytorch style dataset
'''
import numpy as np
from collections import defaultdict
import torch
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import Subset
from tqdm import tqdm

def make_manifold_data(dataset, sampled_classes, examples_per_class, seed, classes=None, max_class=None):
    '''
    Samples manifold data for use in later analysis

    Args:
        dataset: PyTorch style dataset, or iterable that contains (input, label) pairs
        sampled_classes: Number of classes to sample from (must be less than or equal to
            the number of classes in dataset)
        examples_per_class: Number of examples per class to draw (there should be at least
            this many examples per class in the dataset)
        seed: numpy random seed
        classes (optional): Superset of class IDs to choose randomly from. 
        max_class (optional): Choose class IDs randomly from 0 up to this value.
        
    Returns:
        data: list containing tensors for each class of shape (examples_per_class, input dimensions)
    '''
    assert examples_per_class * sampled_classes <= len(dataset), 'Not enough examples per class in dataset'
    assert classes is not None or max_class is not None, "One of max class or classes must be passed" 
    # Set the seed
    np.random.seed(seed)
    # Storage for samples
    sampled_data = defaultdict(list)
    # Sample the labels
    if classes != None:
        assert len(classes) >= sampled_classes, "Not enough classes in dataset"
        sampled_labels = np.random.choice(classes, size=sampled_classes, replace=False)
    else:
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
    
def get_subset(dataset, p): 
    '''
    Restrict the imagenet dataset to P-many classes.  
    Args:
    - dataset: torchvision dataset
    - p: number of classes
    Returns:
    - subset: a pytorch dataset, restricted to the images in a specific class
    - class_idxs: the class labels 
    '''
    num_cls = len(dataset.classes)
    # choose the classes labels to look for
    class_idxs = np.random.choice(np.arange(num_cls), size=p, replace=False)
    print('class ids = ', class_idxs)
    # grab the position of all images in the dataset which are in these classes
    idxs = [i for i in range(len(dataset)) if dataset.imgs[i][1] in class_idxs]
    return Subset(dataset, idxs), class_idxs 

@torch.no_grad()
def score_imgs(dataset, model):
    '''
    Iterate through a dataset and score the performance of the model on each image.
    Args:
    - dataset: pytorch dataset. We assume a batchsize of 1.
    - model: torch model 
    Returns:
    - scores: a tensor of shape (len(dataset), 2) containing the class id and the score.
    '''
    scores = torch.zeros(size=(len(dataset), 2))
    for i, (img, label) in tqdm(enumerate(dataset)): 
        out = model(img.unsqueeze(0)).squeeze(0).log_softmax(0)
        score = out[label]
        # print(i,out.argmax())
        scores[i, 0] = label
        scores[i, 1] = score 
    return scores

def get_top_k(dataset, model, p, k):
    '''
    Get the top k images from a subset of p classes.
    Args:
    - dataset: pytorch style iterable dataset
    - model: torch model
    - p: number of classes to take
    - k: number of samples per class
    Returns: 
    - dat: dataset containing the top k datapoints
    '''
    # subset the data to p classes and score each image
    subset, class_idxs = get_subset(dataset, p)
    print(len(subset), flush=True)
    scores = score_imgs(subset, model)
    # sort the scores and data along the relevant column
    print('scores pre sort = ', scores[:15])
    score_order = scores[:, 1].argsort(descending=True)
    scores = scores[score_order]
    subset = Subset(subset, score_order)
    assert len(subset) == len(scores)
    print('scores post sort = ', scores[:15])
    # determine the position of the top k images in each class
    positions = torch.zeros(p*k)
    for i in range(p):
        # get the position of the top-scoring k occurences of a given label
        positions[i*k:(i+1)*k] = torch.argwhere(scores[:, 0]==class_idxs[i]).squeeze()[:k]
    # take a second subset of the data 
    positions = positions.to(torch.int)
    return Subset(subset, positions), class_idxs
