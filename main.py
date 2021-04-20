import math
import torch
import pickle
import itertools
import numpy as np

### Dependent variables
num_train_examples_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
random_labels_list = [True, False]
binary_digits_list = [True, False]
depth_list = [2, 7]
width_list = [5000, 10000]
seed_list = [0, 1, 2]

### Data hyperparameters
batch_size = 50

### Training hyperparameters
init_lr = 0.01
epochs = 20
decay = 0.9

### Estimator hyperparameters
num_samples = 10000
cuda = True
delta = 0.01


### Criterions
lmcl_loss = LMCL_loss(num_classes=2, feat_dim=-1, s=7.00, m=0.2)
softmax = SoftmaxLoss(feat_dim=-1, num_classes=2)
criterion_list = [lmcl_loss, softmax]

param_product = itertools.product( num_train_examples_list, random_labels_list, binary_digits_list, criterion_list, depth_list, width_list, seed_list )

for params in param_product:
    print('\n', params)
    num_train_examples, random_labels, binary_digits, criterion_class, depth, width, seed = params

    ### Set random seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    ### Setup criterion
    criterion.feat_dim = width

    ### Get data
    full_batch_train_loader, train_loader, test_loader = get_data(  num_train_examples=num_train_examples,
                                                                    batch_size=batch_size, 
                                                                    random_labels=random_labels, 
                                                                    binary_digits=binary_digits )

    ### Train network
    train_acc, test_acc, model = train_network( train_loader = train_loader,
                                                test_loader = test_loader,
                                                depth=depth,
                                                width=width,
                                                criterion=criterion,
                                                epochs=epochs, 
                                                init_lr=init_lr, 
                                                decay=decay.
                                                loss_lr=0.1)

    print(f"Train acc: {train_acc[-1]}")
    print(f"Test acc: {test_acc}")

    results = (train_acc, test_acc)
    fname = 'logs/' + str(params) + '.pickle'
    pickle.dump( results, open( fname, "wb" ) )