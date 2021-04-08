import argparse
import torch

def get_args(model_opt, dataset):
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=600,
                        help='Number of epochs to train.')
##################
 #   parser.add_argument('--lr', type=float, default=0.1,
  #                      help='Initial learning rate.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate.')
    #parser.add_argument('--weight_decay', type=float, default=5e-6,
    #                    help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--weight_decay', type=float, default=5e-6,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--degree', type=int, default=4,
                        help='degree of the approximation.')
  #  parser.add_argument('--hidden', type=int, default=16,
  #                      help='Number of hidden units.')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
###################
    args = parser.parse_args(args=[])

# depend on model
    if model_opt == 'GCN':
        args.droupout = 0
        args.lr = 0.01
        args.weight_decay = 5e-4
        if dataset == "SmallGap":
            args.lr = 0.01
            args.weight_decay = 5e-4
           # args.epochs = 1000
            args.epochs = 500
        elif dataset == "SameFeats":
            args.lr = 0.05
            args.weight_decay = 5e-5
        
        
    elif model_opt == 'SGC':
        args.lr = 0.2
        if dataset == "cora":
            args.weight_decay = 5e-6
        # elif dataset == "citeseer":
        #     args.lr = 0.1
        #     args.weight_decay = 5e-4
        elif dataset == "citeseer":
            args.lr = 0.2
            args.weight_decay = 5e-6
        elif dataset == "pubmed":
            args.weight_decay = 5e-6
        elif dataset == "SmallGap":
            args.lr = 0.4
#             args.weight_decay = 5e-4
            args.epochs = 1000
        elif dataset == "SameFeats":
            args.weight_decay = 2e-05
            
    elif model_opt == 'GFNN':
        args.lr = 0.05
        if dataset == "cora":
            args.weight_decay = 5e-05
        elif dataset == "citeseer":
            args.weight_decay = 5e-04
        elif dataset == "pubmed":
            args.lr = 0.08
            args.weight_decay = 5e-04
        elif dataset == "SmallGap":
            args.lr = 0.02
            args.weight_decay = 2e-05
            args.epochs = 1000
        elif dataset == "SameFeats":
            args.weight_decay = 5e-05
            
            
    elif model_opt == 'GFN':
        args.lr = 0.008
        args.hidden = 64
        if dataset == "cora":
            args.weight_decay = 5e-6
        elif dataset =="citeseer":
            args.weight_decay = 2e-4
        elif dataset =="pubmed":
            args.weight_decay = 2e-4
            args.dropout = 0.5
        elif dataset =="SmallGap":
            args.weight_decay = 2e-4
            args.epochs = 1000
        elif dataset =="SameFeats":
            args.weight_decay = 5e-4
            args.lr = 0.05
        
        
    elif model_opt == 'GIN':
        args.lr = 0.005
        args.weight_decay = 5e-3
        args.dropout = 0.3
        if dataset == 'pubmed':
            args.lr = 0.08
            args.weight_decay = 8e-4
        if dataset == "SmallGap":
            args.lr = 0.02
            args.weight_decay = 5e-6
            args.epochs = 1000
        elif dataset == "SameFeats":
            args.lr = 0.05
            args.weight_decay = 5e-5

            
        
    elif model_opt == 'AFGNN' or 'FD1' or 'FD2':
        args.lr = 0.1
        args.weight_decay = 5e-4
        if dataset == 'cora':
            args.weight_decay = 8e-4
        if dataset == 'citeseer':
            args.weight_decay = 1e-3
        if dataset == 'pubmed':
            args.lr = 0.12
            args.weight_decay = 1e-3
        if dataset == "SmallGap":
            args.lr = 0.12
            args.weight_decay = 2e-5
            args.epochs = 1000
        elif dataset == "SameFeats":
            args.lr = 0.2
            args.weight_decay = 8e-4
       

    elif model_opt == 'AFGNN_fixed_order':
        args.lr = 0.12
        args.weight_decay = 5e-4
        if dataset == 'cora':
            args.weight_decay = 8e-4
        if dataset == 'citeseer':
            args.weight_decay = 1e-3
        if dataset == 'pubmed':
            args.lr = 0.12
            args.weight_decay = 1e-3
        if dataset == "SmallGap":
            args.lr = 0.12
            args.weight_decay = 2e-5
            args.epochs = 1000
        elif dataset == "SameFeats":
            args.lr = 0.2
            args.weight_decay = 8e-4 
        


####################
 
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
