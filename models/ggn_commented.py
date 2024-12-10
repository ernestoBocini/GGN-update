import os

import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_mean, scatter, scatter_add, scatter_max
from torch_geometric.nn.conv import MessagePassing

import eeg_util
from eeg_util import DLog
from models.baseline_models import GAT
from models.graph_conv_layer import *
from models.encoder_decoder import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#############################################
##### GGN MODEL CLASS #######################
#############################################

class GGN(nn.Module):
    def __init__(self, adj, args, out_mid_features=False):
        super(GGN, self).__init__()

        # Initialize arguments, adjacency matrix, and other parameters
        self.args = args
        self.adj_eps = 0.1
        self.adj = adj
        self.adj_x = adj
        self.N = adj.shape[0]  # Number of electrodes (nodes)
        print('N:', self.N)

        en_hid_dim = args.encoder_hid_dim  # Hidden dimension for encoder
        en_out_dim = 16  # Output dimension for encoder
        self.out_mid_features = out_mid_features

        #############################################
        ##### TEMPORAL ENCODER ######################
        #############################################
        # Maps EEG time-series data to temporal features (Step 1 in the figure)
        if args.encoder == 'rnn':
            self.encoder = RNNEncoder(args, args.feature_len, en_hid_dim, en_out_dim, args.bidirect)
            decoder_in_dim = en_hid_dim
            if args.bidirect:
                decoder_in_dim *= 2
        elif args.encoder == 'lstm':
            self.encoder = LSTMEncoder(args, args.feature_len, en_hid_dim, en_out_dim, args.bidirect)
            decoder_in_dim = en_hid_dim
            if args.bidirect:
                decoder_in_dim *= 2
        elif args.encoder == 'cnn2d':
            # CNN option to extract temporal features (Step 1)
            self.encoder = CNN2d(in_dim=args.feature_len,
                                 hid_dim=en_hid_dim,
                                 out_dim=args.decoder_out_dim,
                                 width=34, height=self.N, stride=2, layers=3, dropout=args.dropout)
            decoder_in_dim = args.decoder_out_dim
        else:
            # Default to multi-layer encoders for temporal features
            self.encoder = MultiEncoders(args, args.feature_len, en_hid_dim, en_out_dim)
            decoder_in_dim = en_out_dim * 2

        #############################################
        ##### CONNECTIVITY GRAPH GENERATOR ##########
        #############################################
        # Generates dynamic latent connectivity graphs (Step 2 & 3 in the figure)
        if args.gnn_adj_type == 'rand':
            self.adj = None
            self.adj_tensor = None

        if args.lgg:
            self.LGG = LatentGraphGenerator(args, adj, args.lgg_tau, decoder_in_dim, args.lgg_hid_dim,
                                            args.lgg_k)

        #############################################
        ##### SPATIAL DECODER #######################
        #############################################
        # Decodes node and graph-level spatial features (Step 4 in the figure)
        if args.decoder == 'gnn':
            if args.cut_encoder_dim > 0:
                decoder_in_dim *= args.cut_encoder_dim
            self.decoder = GNNDecoder(self.N, args, decoder_in_dim, args.decoder_out_dim)
        elif args.decoder == 'gat_cnn':
            # Combines GAT and CNN-based spatial decoders (Step 4)
            self.adj_x = torch.ones((self.N, self.N)).float().cuda()
            g_pooling = GateGraphPooling(args, self.N)
            gnn = GAT(decoder_in_dim, args.gnn_hid_dim, args.decoder_out_dim, dropout=args.dropout, pooling=g_pooling)
            cnn = CNN2d(decoder_in_dim, args.decoder_hid_dim, args.decoder_out_dim,
                        width=34, height=self.N, stride=2, layers=3, dropout=args.dropout)
            self.decoder = SpatialDecoder(args, gnn, cnn)
        elif args.decoder == 'lgg_cnn':
            # Combines LGG and CNN for decoding spatial features
            gnn = GNNDecoder(self.N, args, decoder_in_dim, args.gnn_out_dim)
            cnn = CNN2d(decoder_in_dim, args.decoder_hid_dim, args.decoder_out_dim,
                        width=34, height=self.N, stride=2, layers=3, dropout=args.dropout)
            self.decoder = SpatialDecoder(args, gnn, cnn)
        else:
            self.decoder = None

        #############################################
        ##### CLASSIFIER ############################
        #############################################
        # Fully connected layers for classification (Step 5 in the figure)
        self.predictor = ClassPredictor(
            decoder_in_dim, hidden_channels=args.predictor_hid_dim,
            class_num=args.predict_class_num, num_layers=args.predictor_num, dropout=args.dropout)

        self.warmup = args.lgg_warmup
        self.epoch = 0
        self.reset_parameters()

    #############################################
    ##### ADJACENCY MATRIX PROCESSING ###########
    #############################################
    # Helper function to preprocess adjacency matrix
    def adj_to_coo_longTensor(self, adj):
        adj[adj > self.adj_eps] = 1
        adj[adj <= self.adj_eps] = 0
        idx = torch.nonzero(adj).T.long()  # (row, col)
        return idx

    #############################################
    ##### FORWARD PASS ##########################
    #############################################
    # Defines the forward pass through the entire architecture
    def forward(self, x, *options):
        B, C, N, T = x.shape  # Input shape: Batch, Channels, Nodes, Time Steps

        # Step 1: Temporal encoding of input data
        x = self.encode(x)

        # Step 2: Reshape temporal features for processing
        x = x.permute(0, 3, 1, 2)  # Permute to [Batch, Time, Nodes, Channels]

        # Step 3: Generate latent graphs using the LGG (if enabled)
        if self.args.lgg:
            if self.args.lgg_time:
                adj_x_times = []
                for t in range(T):
                    x_t = x[:, t, ...]
                    adj_x = self.LGG(x_t, self.adj)
                    adj_x_times.append(adj_x)
                self.adj_x = adj_x_times
            else:
                x_t = x[:, -1, ...]  # Use the last time step
                self.adj_x = self.LGG(x_t, self.adj)

        # Step 4: Decode spatial features
        x = self.decode(x, B, N, self.adj_x)

        # Step 5: Classify seizure types using the predictor
        x = self.predictor(x)
        return x

    #############################################
    ##### PARAMETER RESET FUNCTION ##############
    #############################################
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


#############################################
##### CLASSIFIER MODULE #####################
#############################################
# Fully connected classifier for seizure type prediction
class ClassPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, class_num, num_layers, dropout=0.5):
        super(ClassPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, class_num))
        self.dropout = dropout

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


#############################################
##### LATENT GRAPH GENERATOR ################
#############################################
# Generates dynamic connectivity graphs
class LatentGraphGenerator(nn.Module):
    def __init__(self, args, A_0, tau, in_dim, hid_dim, K=10):
        super(LatentGraphGenerator, self).__init__()
        self.N = A_0.shape[0]
        self.gumbel_tau = tau

        # Para-Learner: Three independent GNNs to learn Gaussian parameters
        self.mu_nn = MultilayerGNN(self.N, 2, None, in_dim, hid_dim, K, args.dropout)
        self.sig_nn = MultilayerGNN(self.N, 2, None, in_dim, hid_dim, K, args.dropout)
        self.pi_nn = MultilayerGNN(self.N, 2, None, in_dim, hid_dim, K, args.dropout)

    def update_A(self, mu, sig, pi):
        logits = torch.log(torch.softmax(pi, dim=-1))
        pi_onehot = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=False, dim=-1)
        mu_k = torch.sum(mu * pi_onehot, dim=-1)
        sig_k = torch.sum(sig * pi_onehot, dim=-1)
        n = torch.randn((mu_k.shape[0], self.N)).cuda()
        S = mu_k + n * sig_k
        P = torch.sigmoid(S @ S.T)
        return P.mean(dim=0)

    def forward(self, x, adj_t=None):
        mu = self.mu_nn(x, adj_t)
        sig = self.sig_nn(x, adj_t)
        pi = self.pi_nn(x, adj_t)
        return self.update_A(mu, sig, pi)
