import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
import torch.distributed as dist
import datetime


class GraphAttentionLayer(Module):
    def __init__(self, opt):
        super(GraphAttentionLayer, self).__init__()

        # self.cos = nn.CosineSimilarity(dim=2)
        # self.w = nn.Parameter(torch.randn(opt.emb_size, opt.emb_size))
        # # self.linear_1 = nn.Linear(4 * opt.emb_size, 2 * opt.emb_size)
        # self.linear_2 = nn.Linear(2 * opt.emb_size, opt.emb_size)
        # self.linear_3 = nn.Linear(opt.emb_size, 1)
        # self.linear_4 = nn.Linear(opt.emb_size, opt.emb_size, bias=False)
        # # self.BN1 = nn.BatchNorm1d(2 * opt.emb_size)
        # self.BN2 = nn.BatchNorm1d(opt.emb_size)

    def forward(self, item_emb, mask):  # (batch, max_len, emb_size)
        # print('item_emb',item_emb)
        batch = item_emb.size(0)
        max_len = item_emb.size(1)
        # w_i = self.w * item_emb
        final = [item_emb]
        # final = []
        for i in range(3):
            # item_emb = self.w * item_emb
            # A = self.cos(item_emb.repeat(1, 1, max_len).reshape(batch, max_len * max_len, -1),
            #              item_emb.repeat(1, max_len, 1)).reshape(batch, max_len, -1)  # pad节点的运算对吗？
            A = nn.functional.normalize(item_emb, dim=2)  # l2 norm
            A = torch.matmul(A, A.permute(0, 2, 1))
            # (batch, max_len * max_len, emb_size), (batch, max_len * max_len, emb_size) => (batch, max_len * max_len)
            # => (batch, max_len, max_len)
            # print('A:', A.size(), A)
            # A = F.gumbel_softmax(A)
            # A_min = torch.min(A, dim=-1, keepdim=True)  # (batch, max_len, 1)
            # A_max = torch.max(A, dim=-1, keepdim=True)
            # A = (A - A_min[0]) / (A_max[0] - A_min[0] + 2 ** -32)
            # A_flatten = A.reshape(batch, max_len * max_len)
            # sorted_A_flatten = torch.sort(A_flatten, dim=-1, descending=True)[0]
            # num_nonzero = torch.sum(torch.where(sorted_A_flatten > 0, torch.FloatTensor([1]).cuda(), torch.FloatTensor([0]).cuda()), dim=1)  # (batch)
            # epsilon = torch.gather(sorted_A_flatten, dim=1, index=torch.ceil(num_nonzero*0.5).unsqueeze(1).long())  # (batch, 1)
            # H = torch.where(A > epsilon.unsqueeze(2), torch.FloatTensor([1]).cuda(), torch.FloatTensor([0]).cuda())
            H = torch.where(A > 0, torch.FloatTensor([1]).cuda(), torch.FloatTensor([0]).cuda())
            # print('A_:', A_.size(), A_)
            B = torch.sum(H, dim=2)  # (batch, max_len)
            B = torch.where(B > 0, 1.0/B, B)
            item_emb = torch.matmul(B.unsqueeze(2) * H, item_emb)  # (batch, max_len, emb_size)
            # B_ = torch.sqrt(B).unsqueeze(1)  # (batch, 1, max_len)
            # H = H / (B_ + 2**-32)/ (B_.permute(0, 2, 1) + 2**-32)
            # # item_emb = torch.matmul(H, torch.matmul(H, item_emb))
            # item_emb_q = torch.matmul(H, torch.matmul(H, item_emb))  # (batch, max_len, emb_size)

            # att_inputs = torch.matmul(item_emb, self.w)  # (batch, max_len, emb_size)
            # # last_hidden_nn_layer = torch.cat(
            # #     [att_inputs, item_emb_q, att_inputs - item_emb_q, att_inputs * item_emb_q], -1
            # # )  # (batch, max_len, 4*emb_size)
            # last_hidden_nn_layer = torch.cat(
            #     [att_inputs, item_emb_q], -1
            # )  # (batch, max_len, 4*emb_size)
            # # curr_hidden_nn_layer = self.linear_1(last_hidden_nn_layer)  # (batch, max_len, 2*emb_size)
            # # curr_hidden_nn_layer = self.BN1(curr_hidden_nn_layer.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, max_len, 2*emb_size)
            # # curr_hidden_nn_layer = F.relu(curr_hidden_nn_layer)
            # curr_hidden_nn_layer = self.linear_2(last_hidden_nn_layer)
            # curr_hidden_nn_layer = self.BN2(curr_hidden_nn_layer.permute(0, 2, 1)).permute(0, 2, 1)
            # curr_hidden_nn_layer = F.relu(curr_hidden_nn_layer)
            # curr_hidden_nn_layer = self.linear_3(curr_hidden_nn_layer)  # (batch, max_len, 1)
            #
            #
            # att_weights = curr_hidden_nn_layer.squeeze(2)  # (batch, max_len)
            # ####mask
            # mask_paddings = torch.ones_like(att_weights) * (-(2 ** 32) + 1)
            # att_weights = F.softmax(torch.where(mask==1, att_weights, mask_paddings), dim=-1)  # 每行做softmax
            # # output = item_emb * att_weights.unsqueeze(-1)  # (batch, max_len, emb_size)
            # E = H * att_weights.unsqueeze(1)  # (batch, max_len, max_len)
            # E = F.leaky_relu(E)
            # ####mask
            # mask_paddings = torch.ones_like(E) * (-(2 ** 32) + 1)  # 负无穷（方便softmax）
            # E = F.softmax(torch.where(H==1, E, mask_paddings), dim=-1)
            # item_emb_c = torch.matmul(E, item_emb)   # (batch, max_len, emb_size)
            # item_emb = F.leaky_relu(self.linear_4(item_emb_q) + item_emb)
            # item_emb = item_emb_q + item_emb



            final.append(item_emb)
        # print('v:', v.size(), v)
        v = torch.sum(torch.stack(final), 0)
        return v


# class GraphAttentionNetwork(Module):
#     def __init__(self, opt):
#         super(GraphAttentionNetwork, self).__init__()
#         self.attentions = [GraphAttentionLayer(opt) for _ in range (opt.n_heads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
#         self.linear = nn.Linear(opt.n_heads * opt.emb_size, opt.emb_size, bias=True)  # bias为True?
#
#     def forward(self, item_emb):  # (batch, max_len, emb_size)
#         v_ = torch.cat([att(item_emb) for att in self.attentions], dim=2)  # (batch, max_len, n_heads * emb_size)
#         print('v_:', v_.size(), v_)
#         v = self.linear(v_)  # (batch, max_len, emb_size)
#         print('v:', v.size(), v)
#         return v


class IAGNN(Module):
    def __init__(self, opt, n_node):
        super(IAGNN, self).__init__()
        self.embedding = nn.Embedding(n_node, opt.emb_size)  # 参数初始化问题（模块里含模块）需要把第一个node参数全置为0吗？
        # self.GAT = GraphAttentionNetwork(opt)
        self.GAT = GraphAttentionLayer(opt)
        self.n_i = opt.n_intentions
        self.w_p = nn.Parameter(torch.randn(opt.n_intentions, opt.emb_size))
        self.w_c = nn.Parameter(torch.randn(opt.emb_size, opt.emb_size))
        self.q_c = nn.Parameter(torch.randn(opt.emb_size))
        self.temp = opt.temp
        self.w_u_1 = nn.Parameter(torch.randn(opt.emb_size, opt.emb_size))
        self.w_u_2 = nn.Parameter(torch.randn(opt.emb_size, opt.emb_size))
        # self.w_alpha = nn.Parameter(torch.randn(opt.emb_size, opt.emb_size))
        # self.b_alpha = nn.Parameter(torch.randn(opt.emb_size))
        self.q_alpha = nn.Parameter(torch.randn(opt.emb_size))
        self.epsilon = opt.epsilon
        self.linear1 = nn.Linear(opt.emb_size, opt.emb_size)
        # self.linear2 = nn.Linear(opt.emb_size, opt.emb_size, bias=False)
        self.linear2 = nn.Linear(opt.emb_size, opt.emb_size)
        self.w1 = nn.Parameter(torch.randn(opt.emb_size))
        self.linear3 = nn.Linear(opt.emb_size, opt.emb_size)
        # self.pos_embedding = nn.Embedding(200, opt.emb_size)
        # self.W = nn.Parameter(torch.randn(opt.emb_size, opt.emb_size))
        self.linear1_ = nn.Linear(3 * opt.emb_size, 2 * opt.emb_size)
        self.linear2_ = nn.Linear(2 * opt.emb_size, opt.emb_size)
        self.linear3_ = nn.Linear(opt.emb_size, 1)
        # self.BN1 = nn.BatchNorm2d(opt.n_intentions)


        self.init_parameters(opt)

    def init_parameters(self, opt):
        std = 1 / math.sqrt(opt.emb_size)
        # for weight in self.parameters():
        for name, weight in self.named_parameters():
            weight.data.uniform_(-std, std)
            #     weight.data[1:].uniform_(-std, std)
            #     print('name',name,'weight',weight)
            # else:


    def forward(self, data, opt):  # (batch, max_len) pad的元素会对结果有影响吗？需不需要mask？
        # with torch.no_grad():
        #     self.embedding.weight[0].fill_(0)
        mask = torch.where(data==0, torch.FloatTensor([0]).cuda(), torch.FloatTensor([1]).cuda())  # (batch, max_len)
        zeros = torch.zeros(1, opt.emb_size).cuda()
        item_embedding = torch.cat([zeros, self.embedding.weight], dim=0)  # (n_node+1, emb_size)
        item_emb = torch.FloatTensor(len(data), len(data[0]), opt.emb_size).cuda()
        for i in range(len(data)):
            item_emb[i] = item_embedding[data[i]]
        # item_emb = self.embedding(data)  # (batch, max_len, emb_size)
        batch = item_emb.size(0)
        max_len = item_emb.size(1)
        v = self.GAT(item_emb, mask)  # (batch, max_len, emb_size)
        # v = item_emb

        # v_i_w = torch.matmul(v.repeat(1, 1, self.n_i).reshape(batch, max_len * self.n_i, -1) * self.w_p.repeat(max_len, 1),
        #              self.w_c)  # (batch, max_len * n_intentions, emb_size) * (max_len * n_intentions, emb_size)
        # # => (batch, max_len * n_intentions, emb_size) => matmul(emb_size, emb_size) => (batch, max_len * n_intentions, emb_size)
        # C = torch.matmul(F.leaky_relu(v_i_w), self.q_c).reshape(batch, max_len, -1) # (batch, max_len * n_intentions) => (batch, max_len, n_intentions)
        # C = torch.matmul(v, self.w_p.permute(1, 0))
        C = torch.sigmoid(self.linear1(self.w_p.unsqueeze(0).unsqueeze(2).repeat(batch, 1, max_len, 1)) + self.linear2(v.unsqueeze(1).repeat(1, opt.n_intentions, 1, 1)))  # (batch, n_intentions, max_len, emb_size)
        # ffn_0 = torch.cat([self.w_p.unsqueeze(0).unsqueeze(2).repeat(batch, 1, max_len, 1),
        #                    v.unsqueeze(1).repeat(1, opt.n_intentions, 1, 1),
        #                    self.w_p.unsqueeze(0).unsqueeze(2).repeat(batch, 1, max_len, 1) * v.unsqueeze(1).repeat(1, opt.n_intentions, 1, 1)],
        #                   dim=-1)  # (batch, n_intentions, max_len, 3*emb_size)
        # ffn_1 = F.leaky_relu(self.linear1_(ffn_0))  # (batch, n_intentions, max_len, emb_size)
        # # ffn_1 = self.BN1(ffn_1)  # (batch, n_intentions, max_len, emb_size)
        # ffn_2 = F.leaky_relu(self.linear2_(ffn_1))  # (batch, n_intentions, max_len, emb_size/2)
        # C = self.linear3_(ffn_2).squeeze(-1)  # (batch, n_intentions, max_len)
        C = torch.matmul(C, self.w1)  # (batch, n_intentions, max_len)
        C_mask = mask.unsqueeze(1).repeat(1, opt.n_intentions, 1)  # (batch, n_intentions, max_len)
        mask_pad = torch.zeros_like(C_mask).cuda()
        C = torch.where(C_mask == 1, C, mask_pad)
        C_sum = torch.sum(C, dim=-1)  # (batch, n_intentions)
        C = C / C_sum.unsqueeze(-1)
        # print('C:', C.size(), C)
        # E = F.softmax(C / self.temp, dim=2)  # (batch, max_len, n_intentions)
        # E = F.softmax(C / self.temp, dim=1)  # (batch, n_intentions, max_len)
        # E = F.softmax(C, dim=1)
        # print('E:', E.size(), E)
        # e_v = F.leaky_relu(torch.matmul(E.permute(0, 2, 1), torch.matmul(v, self.W)))  # (batch, n_intentions, emb_size) 没考虑除以|s|平均
        # e_v = torch.matmul(E.permute(0, 2, 1), v)
        # m = torch.matmul(E, v)
        m = torch.matmul(C, v)
        # m = torch.matmul(C, item_emb)
        # BN2 = nn.BatchNorm1d(e_v.size(1)).cuda()
        # u = torch.sigmoid(torch.matmul(e_v, self.w_u_1) + torch.matmul(self.w_p, self.w_u_2).repeat(batch, 1, 1))  # (batch, n_intentions, emb_size)
        # print('u:', u.size(), u)
        # m = u * e_v + (1 - u) * (self.w_p.repeat(batch, 1, 1))  # (batch, n_intentions, emb_size)
        # m = e_v
        # print('m:', m.size(), m)

        # BN3 = nn.BatchNorm1d(e_v.size(1)).cuda()
        # alpha = torch.matmul(F.leaky_relu(torch.matmul(m, self.w_alpha) + self.b_alpha.repeat(batch, self.n_i, 1)),
        #                      self.q_alpha.unsqueeze(1))
        alpha = torch.sigmoid(self.linear3(m))
        alpha = torch.matmul(alpha, self.q_alpha.unsqueeze(1))
        # print('alpha:', alpha.size(), alpha)
        # (batch, n_intentions, emb_size) (emb_size, 1) => (batch, n_intentions, 1)
        # beta = alpha
        beta = F.softmax(alpha, dim=1)
        # print('beta:', beta.size(), beta)

        eps = torch.sort(beta.reshape(beta.size(0)*beta.size(1)), dim=0, descending=True)[0][int(self.epsilon*beta.size(0)*beta.size(1))]
        gamma = torch.where(beta >= eps, beta, torch.FloatTensor([0]).cuda())  # (batch, n_intentions, 1)
        # gamma = alpha
        # gamma = torch.div(gamma, torch.sum(gamma, dim=1).unsqueeze(1))
        # print('gamma:', gamma.size(), gamma)
        # gamma = torch.where(beta >= self.epsilon, beta, torch.zeros_like(beta))
        s = torch.matmul(gamma.permute(0, 2, 1), m).squeeze(1)  # (batch, 1, n_intentions) (batch, n_intentions, emb_size)
        # print('s:', s.size(), s)
        # => (batch, 1, emb_size) => (batch, emb_size)

        y = torch.matmul(s, self.embedding.weight.permute(1, 0)) # (batch, emb_size) (n_node, emb_size) => (batch, n_node)
        # print('y:', y.size(), y)
        # print('-----------------------------------------------------------------------------------------')
        return y


def train_test(model, train_loader, test_loader, optimizer, scheduler, opt):

    # 训练打印loss
    model.train()
    loss_func = nn.CrossEntropyLoss()
    train_loss = torch.zeros(1).cuda()

    print('start training...', datetime.datetime.now())

    for data in train_loader:
        model.zero_grad()  # optimizer.zero_grad()?
        y = model(torch.tensor(data[0]).cuda(), opt)  # (batch, max_len)->(batch, n_node)
        loss = loss_func(y, (torch.tensor(data[1])-1).cuda())  # 不需要-1
        loss.backward()
        # for name, parms in model.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
        optimizer.step()
        train_loss += loss.detach()

    print('\t train_loss : %.3f' % train_loss.item())

    # 测试返回metric(hit mrr)
    metric = {}
    for k in opt.evaluate_k:
        metric['hit@%d' % k] = []
        metric['mrr@%d' % k] = []

    model.eval()
    for data in test_loader:
        y = model(torch.tensor(data[0]).cuda(), opt)
        _, indices = torch.sort(y, dim=1, descending=True)  # indices不需要-1
        for target, index in zip(data[1], indices[:, :20].cpu()):
            for k in opt.evaluate_k:
                metric['hit@%d' % k].append(np.isin(target-1, index[:k]))
                if len(np.where(target-1 == index[:k])[0]) == 0:
                    metric['mrr@%d' % k].append(0)
                else:
                    metric['mrr@%d' % k].append(1. / (np.where(target-1 == index[:k])[0][0] + 1))

    for k in opt.evaluate_k:
        metric['hit@%d' % k] = torch.FloatTensor([np.mean(metric['hit@%d' % k])]).cuda()
        metric['mrr@%d' % k] = torch.FloatTensor([np.mean(metric['mrr@%d' % k])]).cuda()

    scheduler.step()

    return metric