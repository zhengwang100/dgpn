import torch
import torch.nn as nn
import numpy as np
from dgpn import DGPN
from util_functions import get_data_split, get_acc, setup_seed, use_cuda
from util_functions import load_data_set, symmetric_normalize_adj
from lazy_random_walk_utils import get_lrw_pre_calculated_feature_list

device = use_cuda()
#setup_seed(42)

def train(args):
    [c_train, c_val] = args.train_val_class
    idx, labellist, G, features, csd_matrix = load_data_set(args.dataset)
    G = symmetric_normalize_adj(G).todense()

    my_feature_list = get_lrw_pre_calculated_feature_list(features, torch.FloatTensor(G), k=args.k, beta=args.beta)
    idx_train, idx_test, idx_val = get_data_split(c_train=c_train, c_val=c_val, idx=idx, labellist=labellist)
    y_true = np.array([int(temp[0]) for temp in labellist]) #[n, 1]
    y_true = torch.from_numpy(y_true).type(torch.LongTensor).to(device)

    model = DGPN(n_in=my_feature_list[0].shape[1], n_h=args.n_hidden, dropout=args.dropout).to(device)
    csd_matrix = csd_matrix.to(device)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    for epoch in range(args.n_epochs+1):
        model.train()
        optimiser.zero_grad()
        
        preds, local_pred_list = model(my_feature_list, csd_matrix)
        loss_global = criterion(preds[idx_train], y_true[idx_train])
        
        local_loss_list = []
        for local_pred in local_pred_list:
            local_loss_list.append(criterion(local_pred[idx_train], y_true[idx_train]))
        loss_local = args.alpha*torch.mean(torch.stack(local_loss_list), dim=0)
        
        loss = loss_global + loss_local
        
        if epoch % 100 == 0:
            train_acc = get_acc(preds[idx_train], y_true[idx_train], c_train=c_train, c_val=c_val, model='train') 
            test_acc = get_acc(preds[idx_test], y_true[idx_test], c_train=c_train, c_val=c_val, model='test') 
            print(epoch, 'Loss:',  loss.item(), 'Train_acc:', train_acc, 'Test_acc:', test_acc)
            
            model.eval()
            preds, _ = model(my_feature_list, csd_matrix)
            test_acc = get_acc(preds[idx_test], y_true[idx_test], c_train=c_train, c_val=c_val, model='test') 
            print('Evaluation!', 'Test_acc:', test_acc, "+++")
                       
        loss.backward()
        optimiser.step()
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MODEL')
    parser.add_argument("--dataset", type=str, default='cora', choices=['cora', 'citeseer', 'C-M10-M'], help="dataset")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--train-val-class", type=int, nargs='*', default=[3, 0], help="the first #train_class and #validation classes")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=1000, help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=128, help="number of hidden layers")
    parser.add_argument("--wd", type=float, default=0, help="Weight for L2 loss")
    parser.add_argument("--k", type=int, default=3, help="k-hop neighbors")
    parser.add_argument("--beta", type=float, default=0.7, help="probability of staying at the current node in a lazy random walk")
    parser.add_argument("--alpha", type=float, default=1.0, help="hyper-parameter for local loss")
    args = parser.parse_args()
    print(args)
    train(args)