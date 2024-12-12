   
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Inception

class GatedFusion(nn.Module):
    def __init__(self, embed_dim):
        super(GatedFusion, self).__init__()
        self.embed_dim = embed_dim

        self.norm = nn.LayerNorm(embed_dim)
        self.w = nn.Linear(embed_dim, embed_dim)
        self.trans = nn.Parameter(torch.zeros(embed_dim, embed_dim), requires_grad=True)
        nn.init.xavier_normal_(self.trans, gain=1.414)
        self.w_r = nn.Linear(embed_dim, embed_dim)
        self.u_r = nn.Linear(embed_dim, embed_dim)
        self.w_h = nn.Linear(embed_dim, embed_dim)
        self.w_u = nn.Linear(embed_dim, embed_dim)

    def forward(self, node_embeddings, dyn_embeddings):
        #node_embeddings shaped [N, D], x shaped [B, N, D]
        #output shaped [B, N, D] 
        batch_size = dyn_embeddings.shape[0]
        
        node_embeddings = self.norm(node_embeddings)
        node_embeddings_res = self.w(node_embeddings) + node_embeddings
        node_embeddings_res = node_embeddings_res.repeat(batch_size,  1, 1)

        et_res = dyn_embeddings + torch.einsum('bnd,dd->bnd', dyn_embeddings, self.trans)

        z = torch.sigmoid(node_embeddings_res + et_res)
        r = torch.sigmoid(self.w_r(dyn_embeddings) + self.u_r(node_embeddings).repeat(batch_size, 1, 1))
        h = torch.tanh(self.w_h(dyn_embeddings) + r * self.w_u(node_embeddings).repeat(batch_size, 1, 1))
        res = torch.add(z * node_embeddings, torch.mul(torch.ones(z.size()).to(dyn_embeddings.device) - z, h))

        return res



class DynAGCN(nn.Module):
    def __init__(self, dim_in, dim_out, embed_dim, H, W, cheb_k=3):
        super(DynAGCN, self).__init__()
        self.embed_dim = embed_dim
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.mem_num = 128
        self.H = H
        self.W = W
        self.num_node = H * W
        self.cheb_k = cheb_k

        # Dynamic Graph Embedding
        self.fc = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_in//2),
            nn.Sigmoid(),
            nn.Linear(self.dim_in//2, self.dim_in//2),
            nn.Sigmoid(),
            nn.Linear(self.dim_in//2, self.dim_in)
        )
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)
        self.Mem = nn.Parameter(torch.randn(self.mem_num, self.embed_dim), requires_grad=True)
        self.Wq = nn.Parameter(torch.randn(self.dim_in, self.embed_dim), requires_grad=True)
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        nn.init.xavier_normal_(self.node_embeddings)
        nn.init.xavier_normal_(self.Mem)
        nn.init.xavier_normal_(self.Wq)
        nn.init.xavier_normal_(self.weights_pool)
        nn.init.constant_(self.bias_pool, 0)
        self.gated_fusion = GatedFusion(embed_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        stc_E = self.node_embeddings
        x = x.reshape(batch_size, self.num_node, -1)
        x_e = self.fc(x)
        query = torch.einsum('bnd,de->bne', x_e, self.Wq)
        att_score = F.softmax(torch.matmul(query, self.Mem.transpose(0, 1)), dim=-1)
        dyn_E = torch.matmul(att_score, self.Mem)
        comb_E = self.gated_fusion(stc_E, dyn_E)
        support = F.softmax(F.relu(torch.matmul(comb_E, comb_E.transpose(-1,-2))), dim=-1)
        support_set = [torch.eye(self.num_node).to(x.device).unsqueeze(0).repeat(batch_size, 1, 1), support]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * support, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)

        x_g = torch.einsum("kbnm,bmc->bknc", supports, x)
        weights = torch.einsum('nd,dkio->nkio', stc_E, self.weights_pool)
        bias = torch.matmul(stc_E, self.bias_pool)
        x_g = x_g.permute(0, 2, 1, 3)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias

        x_gconv = x_gconv.reshape(batch_size, -1, self.H, self.W)
        return x_gconv


class Mid_XnetG(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker = [3,5,7,11], groups=8):
        super(Mid_XnetG, self).__init__()

        self.N_T = N_T
        self.enc = Inception(channel_in, channel_hid//2, channel_hid, incep_ker=[3], groups=groups)
        self.gcn = DynAGCN(dim_in=channel_hid, dim_out=channel_hid, embed_dim=10, H=16, W=16)
        self.dec = Inception(channel_hid, channel_hid//2, channel_in, incep_ker=[3], groups=groups)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        z = self.enc(x)
        z = self.gcn(z)
        z = self.dec(z)

        y = z.reshape(B, T, C, H, W)
        return y