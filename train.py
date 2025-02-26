import argparse

from utils import *
from tqdm import tqdm
from torch import optim
from model import *
from layers import *
from sklearn.decomposition import PCA


# parameter settings
parser = argparse.ArgumentParser()
parser.add_argument('--gnnlayers', type=int, default=4, help="Number of gnn layers")
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=128, help='hidden_num')
parser.add_argument('--dims', type=int, default=500, help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--hop_num', type=float, default=2, help='Number of hops')
parser.add_argument('--alpha', type=float, default=0.5, help='Banlance parameter for loss function')
parser.add_argument('--threshold', type=float, default=0.95, help='Threshold for high confidence samples')
parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
parser.add_argument('--cluster_num', type=int, default=7, help='type of dataset.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--device', type=str, default='cuda', help='the training device')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("Using {} dataset".format(args.dataset))


# data load and process
adj, features, true_labels, idx_train, idx_val, idx_test = load_data(args.dataset)
pca = PCA(n_components=args.dims)
features = pca.fit_transform(features)
features = torch.FloatTensor(features)
adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
adj.eliminate_zeros()
adj_tensor = torch.tensor(adj.todense(), dtype=torch.float32)
adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
sm_fea_s = sp.csr_matrix(features).toarray()
for a in adj_norm_s:
    sm_fea_s = a.dot(sm_fea_s)
sm_fea_s = torch.FloatTensor(sm_fea_s)

# init augmentor network
MLP_model = MLP_model([features.shape[1]])
MLP_model = MLP_model.cuda()
optimizer_mlp = optim.Adam(MLP_model.parameters(), lr=args.lr)
Atten_Model = Atten_Model(fea=features.cuda(),
                        adj=adj,
                        nhidden=args.hidden,
                        edge_indices_no_diag=adj_tensor.cuda(),
                        nclass=args.cluster_num).cuda()
optimizer_att = optim.Adam(Atten_Model.parameters(), lr=1e-5, weight_decay=1e-5)

acc_list = []
nmi_list = []
ari_list = []
f1_list = []


for seed in range(10):
    setup_seed(seed)

    # init
    best_acc, best_nmi, best_ari, best_f1, predict_labels, centers = clustering(sm_fea_s, true_labels, args.cluster_num)

    # Encoder network to generate embeddings
    model = Encoder_net([features.shape[1]] + [args.dims])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # GPU
    if args.cuda:
        model.cuda()
        inx = sm_fea_s.cuda()

    best_acc = 0
    ident = torch.eye(sm_fea_s.shape[0]).cuda()
    target = torch.eye(sm_fea_s.shape[0]).cuda()
    feat = torch.tensor(features).cuda()

    for epoch in tqdm(range(args.epochs)):

        model.train()
        x_learn = MLP_model(feat)
        inx_2 = x_learn
        atten_adj = Atten_Model(feat) + ident

        H = ident - atten_adj
        for i in range(args.gnnlayers):
            inx_2 = H @ inx_2

        #adver loss
        x_loss = -F.mse_loss(x_learn, feat)
        a_loss = -F.mse_loss(atten_adj, adj_tensor.cuda())
        aug_loss = x_loss + a_loss

        #extract embeddings
        F1 = model(inx)
        F2 = model(inx_2)

        #Contrastive loss
        infoNCE = loss_cal(F1, F2)
        loss = infoNCE + args.alpha * aug_loss
        loss.backward()

        optimizer.step()
        optimizer_att.step()
        optimizer_mlp.step()

        #Epoch > 200, second stage --> Refine
        if epoch > 200:
            if epoch % 20 == 0:
                # select high-confidence samples
                distribute = F.softmax(torch.sum(torch.pow(((F1+F2)/2).unsqueeze(1) - centers, 2), 2), dim=1)
                distribute = torch.min(distribute, dim=1).values
                value, index = torch.topk(distribute, int(len(distribute) * (args.threshold)))
                distribute = torch.where(distribute <= value[-1], torch.ones_like(distribute),
                                         torch.zeros_like(distribute))
                pseudo_label_index = torch.nonzero(distribute).reshape(-1, )
                matrix_index = np.ix_(pseudo_label_index.cpu(), pseudo_label_index.cpu())
                predict_labels = torch.tensor(predict_labels).cuda()
                pseudo_matrix = (predict_labels == predict_labels.unsqueeze(1)).float().cuda()
                S = F1 @ F2.T
                S = normalize(S)

                # refine
                atten_adj[matrix_index] = atten_adj[matrix_index] * pseudo_matrix[matrix_index]
                atten_adj = atten_adj * S.detach()

                inx_2 = x_learn
                H_2 = ident - atten_adj
                for a in range(args.gnnlayers):
                    inx_2 = H @ inx_2
                inx_new_2 = inx_2
                inx_new_2 = inx_new_2.float().cuda()

                if epoch % 10 == 0:
                    model.eval()
                    F_1 = model(inx)
                    F_new_2 = model(inx_new_2)
                    hidden_emb = (F_1 + F_new_2) / 2
                    acc, nmi, ari, f1, predict_labels, centers = clustering(hidden_emb, true_labels, args.cluster_num)
                    if acc >= best_acc:
                        best_acc = acc
                        best_nmi = nmi
                        best_ari = ari
                        best_f1 = f1
        else:
            if epoch % 10 == 0:
                model.eval()
                F1 = model(inx)
                F2 = model(inx_2)
                hidden_emb = (F1 + F2) / 2
                acc, nmi, ari, f1, predict_labels,centers = clustering(hidden_emb, true_labels, args.cluster_num)
                if acc >= best_acc:
                    best_acc = acc
                    best_nmi = nmi
                    best_ari = ari
                    best_f1 = f1
    acc_list.append(best_acc)
    nmi_list.append(best_nmi)
    ari_list.append(best_ari)
    f1_list.append(best_f1)

    tqdm.write('best_acc: {}, best_nmi: {}, best_ari: {}, best_f1: {}'.format(best_acc, best_nmi, best_ari, best_f1))

acc_list = np.array(acc_list)
nmi_list = np.array(nmi_list)
ari_list = np.array(ari_list)
f1_list = np.array(f1_list)
print(acc_list.mean(), acc_list.std())
print(nmi_list.mean(), nmi_list.std())
print(ari_list.mean(), ari_list.std())
print(f1_list.mean(), f1_list.std())
