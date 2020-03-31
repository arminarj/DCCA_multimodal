import gzip
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import torch


def make_tensor(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    data_x = torch.tensor(data_x)
    data_y = np.asarray(data_y, dtype='int32')
    return data_x, data_y


def svm_classify(outputs, labels, C=0.01):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    train_output = outputs[0][0]
    valid_output = outputs[1][0] 
    test_output  = outputs[2][0]
    
    train_label = labels[0].cpu().numpy() 
    valid_label = labels[1].cpu().numpy()  
    test_label  = labels[2].cpu().numpy()
    print(f'train_label shape : {train_label.shape}')
    print(f'valid_label shape : {valid_label.shape}')
    print(f'test_label shape : {test_label.shape}')

    train_data = train_output.cpu().numpy() 
    valid_data = valid_output.cpu().numpy() 
    test_data = test_output.cpu().numpy()  
    print(f'train_data shape : {train_data.shape}')
    print(f'valid_data shape : {valid_data.shape}')
    print(f'test_data shape : {test_data.shape}')

    print('training SVM...')
    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(train_data, train_label.ravel())

    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)
    p = clf.predict(valid_data)
    valid_acc = accuracy_score(valid_label, p)

    return clf, [test_acc, valid_acc]


def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret

def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model, name=''):
    name = save_load_name(args, name)
    torch.save(model, f'pre_trained_models/{name}.pt')


def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load(f'pre_trained_models/{name}.pt')
    return model