import torchvision
import torch
import numpy as np

def poisonLabels(labels, percentPoisoned):
    numLabels = len(labels) #get num of labels
    numPoisoned = int(len(labels) * percentPoisoned) #calculate number of labels to poison
    
    poisonIndexes= np.random.choice(numLabels, numPoisoned, replace=False) #get random indexes to poison
    poisonedLabels = np.random.randint(0,10, size=numPoisoned) #get random labels to poison with
    for i in range(numPoisoned): #poison the labels
        labels[poisonIndexes[i]] = poisonedLabels[i]
        
    return labels, poisonIndexes, poisonedLabels #


def get_data_loader(transform_train, transform_test, poisonPercentage): #Modified get_data_loader
    trainset = torchvision.datasets.CIFAR10(root='./data',
            train=True, download=True, transform=transform_train
        )

    testset = torchvision.datasets.CIFAR10(root='./data',
            train=False, download=True, transform=transform_test
        )
    
    original_labels = np.copy(trainset.targets) #get original labels
    if poisonPercentage > 0: #if we are poisoning the labels
        trainset.targets, poisonIndexes, poisonedLabels = poisonLabels(trainset.targets, poisonPercentage) #poison the labels
    for i in poisonIndexes: #print the original and poisoned labels
        print(f"Original labels: {original_labels[i]}")
        print(f"Poisoned labels: {trainset.targets[i]}")
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False
    )
    return train_loader, test_loader




train_loader, test_loader = get_data_loader(None, None, 0.2)
