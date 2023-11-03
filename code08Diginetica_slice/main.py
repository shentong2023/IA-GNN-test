import argparse
import torch
import os
from torch.multiprocessing import Process
import torch.distributed as dist
import pickle
from utils import Data, collate_fn
from torch.utils.data.distributed import DistributedSampler
from model import IAGNN, train_test
import tempfile
import torch.optim as optim

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name:diginetica/tmall/retailrocket')
parser.add_argument('--emb_size', type=int, default=100, help='embedding size')
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=10, help='the number of step after which learning rate decay')
parser.add_argument('--n_heads', type=int, default=3, help='number of heads in multi-head attention')
parser.add_argument('--n_intentions', type=int, default=3, help='number of intentions')
parser.add_argument('--temp', type=float, default=0.1, help='temperature')  # 0到正无穷区间 好像通常取0.1-1.0
# 参考Modeling Multi-Purpose Sessions for Next-Item Recommendations via Mixture-Channel Purpose Routing Networks
# 和 Self-supervised Graph Learning for Recommendation
parser.add_argument('--evaluate_k', type=list, default=[10, 20], help='k for evaluation')
parser.add_argument('--epsilon', type=float, default=0.85, help='threshold')
parser.add_argument('--epoch', type=int, default=30, help='number of epoch')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop')


opt = parser.parse_args()

print(opt)

def main():

    train_data = pickle.load(open('/home/huashanshan/IA-GNN/datasets/diginetica/preprocessed_slice/train.txt', 'rb'))
    # train_data = pickle.load(open('/home/huashanshan/DHCN/datasets/diginetica/train.txt', 'rb'))
    test_data = pickle.load(open('/home/huashanshan/IA-GNN/datasets/diginetica/preprocessed_slice/test.txt', 'rb'))
    # test_data = pickle.load(open('/home/huashanshan/DHCN/datasets/diginetica/test.txt', 'rb'))

    if opt.dataset == 'diginetica':
        #n_node = 43097  # 实际node个(不加补充的node)
        n_node = 40793  # 实际node个(不加补充的node)
    elif opt.dataset == 'tmall':
        n_node = 43062  # 已加1 含pad
    elif opt.dataset == 'retailrocket':
        n_node = 48984  # 已加1 含pad
    train_data = Data(train_data)
    test_data = Data(test_data)


    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=opt.batch_size,
                                               collate_fn=collate_fn,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=opt.batch_size,
                                              collate_fn=collate_fn,
                                              shuffle=False)

    model = IAGNN(opt, n_node).cuda()

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]  # model里有其他module,子module里的参数算
    optimizer = optim.Adam(pg, lr=opt.lr, weight_decay=opt.l2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)


    top_k = opt.evaluate_k
    best_result = {}
    for k in top_k:
        best_result['hit@%d' % k] = 0
        best_result['mrr@%d' % k] = 0
        best_result['epoch%d' % k] = [0, 0]
    bad_count = 0

    for epoch in range(opt.epoch):
        print('--------------------------------------')
        print('epoch', epoch)
        print('lr:', optimizer.state_dict()['param_groups'][0]['lr'])
        metric = train_test(model, train_loader, test_loader, optimizer, scheduler, opt)
        flag = 1
        for k in top_k:
            if metric['hit@%d' % k].item() > best_result['hit@%d' % k]:
                best_result['hit@%d' % k] = metric['hit@%d' % k].item()
                flag = 0
                if epoch != 0:
                    bad_count = 0
                best_result['epoch%d' % k][0] = epoch
            if metric['mrr@%d' % k].item() > best_result['mrr@%d' % k]:
                best_result['mrr@%d' % k] = metric['mrr@%d' % k].item()
                flag = 0
                if epoch != 0:
                    bad_count = 0
                best_result['epoch%d' % k][1] = epoch
        bad_count += flag
        print('this_epoch----')
        for k in top_k:
            print('recall@%d:%.4f\t mrr@%d:%.4f'
                  % (k, metric['hit@%d' % k].item(), k, metric['mrr@%d' % k].item()))
        print('best_result----')
        for k in top_k:
            print('recall@%d:%.4f\t mrr@%d:%.4f\t epoch:%d,%d'
                  % (k, best_result['hit@%d' % k], k, best_result['mrr@%d' % k], best_result['epoch%d' % k][0], best_result['epoch%d' % k][1]))
        if bad_count >= opt.patience:
            break

    print('Done')


if __name__ == '__main__':
    main()