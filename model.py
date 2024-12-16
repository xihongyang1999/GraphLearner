from layers import *
from inits_gat import glorot
from torch.nn import Linear, LeakyReLU

class Encoder_net(nn.Module):
    def __init__(self, dims):
        super(Encoder_net, self).__init__()
        self.layers1 = nn.Linear(dims[0], dims[1])

    def forward(self, x, is_train=True, sigma=0.01):
        out1 = self.layers1(x)
        out1 = F.normalize(out1, dim=1, p=2)
        return out1


class Atten_Model(torch.nn.Module):
    def __init__(self, fea, adj, nhidden, edge_indices_no_diag, nclass):
        super(Atten_Model, self).__init__()

        self.edge_indices_no_diag = edge_indices_no_diag
        self.in_features = fea.shape[1]
        self.out_features = nhidden
        self.num_classes = nclass
        self.W = Linear(self.in_features, self.out_features, bias=False)
        self.a = Parameter(torch.Tensor(2 * self.out_features, 1))
        self.W1 = Linear(self.in_features, self.num_classes, bias=False)
        self.num1 = fea.shape[0]
        self.features = fea
        self.leakyrelu = LeakyReLU(0.1)
        self.isadj = adj[0]
        self.adj = adj[1]
        self.tmp = []
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W.weight)
        glorot(self.a)
        glorot(self.W1.weight)

    def forward(self, h):
        Wh = self.W(h)
        self.A_ds_no_diag = self.CalAttenA(Wh)
        return self.A_ds_no_diag

    def CalAttenA(self, Wh):
        indices = self.edge_indices_no_diag.clone()
        indices = torch.nonzero(indices).t()
        fea1 = Wh[indices[0, :], :]
        fea2 = Wh[indices[1, :], :]
        fea12 = torch.cat((fea1, fea2), 1)
        atten_coef = torch.exp(self.leakyrelu(torch.mm(fea12, self.a))).flatten()
        A_atten = torch.zeros([self.num1, self.num1]).cuda()
        A_atten[indices[0, :], indices[1, :]] = atten_coef
        s1 = A_atten.sum(1)
        pos1 = torch.where(s1 == 0)[0]
        A_atten[pos1, pos1] = 1
        A_atten = A_atten.t() / A_atten.sum(1)
        return A_atten.t()



class MLP_model(nn.Module):
    def __init__(self, dims):
        super(MLP_model, self).__init__()
        self.layers1 = nn.Linear(dims[0], 1000)
        self.layers2 = nn.Linear(1000, dims[0])

    def forward(self, x):
        out1 = self.layers1(x)
        out2 = self.layers2(out1)
        out2 = F.normalize(out2, dim=1, p=2)
        return out2


