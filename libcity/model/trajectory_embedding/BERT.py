import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
import copy
import random
from logging import getLogger


def drop_path_func(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_func(x, self.drop_prob, self.training)


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)  # (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)  # (max_len, d_model/2)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x, position_ids=None):
        """

        Args:
            x: (B, T, d_model)
            position_ids: (B, T) or None

        Returns:
            (1, T, d_model) / (B, T, d_model)

        """
        if position_ids is None:
            return self.pe[:, :x.size(1)].detach()
        else:
            batch_size, seq_len = position_ids.shape
            pe = self.pe[:, :seq_len, :]  # (1, T, d_model)
            pe = pe.expand((position_ids.shape[0], -1, -1))  # (B, T, d_model)
            pe = pe.reshape(-1, self.d_model)  # (B * T, d_model)
            position_ids = position_ids.reshape(-1, 1).squeeze(1)  # (B * T,)
            output_pe = pe[position_ids].reshape(batch_size, seq_len, self.d_model).detach()
            return output_pe


class GATLayer(torch.nn.Module):
    """
    Base class for all implementations as there is much code that would otherwise be copy/pasted.

    """

    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, load_trans_prob=True):

        super().__init__()

        # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection
        self.load_trans_prob = load_trans_prob

        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #
        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        if self.load_trans_prob:
            self.linear_proj_tran_prob = nn.Linear(1, num_of_heads * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        if self.load_trans_prob:
            self.scoring_trans_prob = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #
        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)

        self.init_params()

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        if self.load_trans_prob:
            nn.init.xavier_uniform_(self.linear_proj_tran_prob.weight)
            nn.init.xavier_uniform_(self.scoring_trans_prob)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, in_nodes_features, out_nodes_features):

        # if the tensor is not contiguously stored in memory we'll get an error after we try to do certain ops like view
        # only imp1 will enter this one
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class GATLayerImp3(GATLayer):
    """
    Implementation #3 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric

    But, it's hopefully much more readable! (and of similar performance)

    It's suitable for both transductive and inductive settings. In the inductive setting we just merge the graphs
    into a single graph with multiple components and this layer is agnostic to that fact! <3

    """

    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 0      # node dimension/axis
    head_dim = 1       # attention head dimension/axis

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, load_trans_prob=True):

        # Delegate initialization to the base class
        super().__init__(num_in_features, num_out_features, num_of_heads, concat, activation, dropout_prob,
                         add_skip_connection, bias, load_trans_prob)

    def forward(self, data):
        #
        # Step 1: Linear Projection + regularization
        #
        in_nodes_features, edge_index, edge_prob = data  # unpack data edge_prob=(E, 1)
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        if self.load_trans_prob:
            # shape = (E, 1) * (1, NH*FOUT) -> (E, NH, FOUT) where NH - number of heads, FOUT - num of output features
            trans_prob_proj = self.linear_proj_tran_prob(edge_prob).view(
                -1, self.num_of_heads, self.num_out_features)  # (E, NH, FOUT)
            trans_prob_proj = self.dropout(trans_prob_proj)
        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, FOUT) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(
            scores_source, scores_target, nodes_features_proj, edge_index)
        if self.load_trans_prob:
            # shape = (E, NH, FOUT) * (1, NH, FOUT) -> (E, NH, FOUT) -> (E, NH)
            scores_trans_prob = (trans_prob_proj * self.scoring_trans_prob).sum(dim=-1)  # (E, NH)
            scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted + scores_trans_prob)
        else:
            scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        # shape = (E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        # Add stochasticity to neighborhood aggregation
        attentions_per_edge = self.dropout(attentions_per_edge)

        #
        # Step 3: Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

        #
        # Step 4: Residual/skip connections, concat and bias
        #
        out_nodes_features = self.skip_concat_bias(in_nodes_features, out_nodes_features)
        return (out_nodes_features, edge_index, edge_prob)

    #
    # Helper functions (without comments there is very little code so don't be scared!)
    #

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and it's (exp) score) and similarly for 2-3 and 3-3
         i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

        Note:
        Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        and it's a fairly common "trick" used in pretty much every deep learning libcity.
        Check out this link for more details:

        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

        """
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()  # (E, NH)
        exp_scores_per_edge = scores_per_edge.exp()  # softmax, (E, NH)

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).

        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # E -> (E, NH)
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)


class GAT(nn.Module):

    def __init__(self, d_model, in_feature, num_heads_per_layer, num_features_per_layer,
                 add_skip_connection=True, bias=True, dropout=0.6, load_trans_prob=True, avg_last=True):
        super().__init__()
        self.d_model = d_model
        assert len(num_heads_per_layer) == len(num_features_per_layer), f'Enter valid arch params.'

        num_features_per_layer = [in_feature] + num_features_per_layer
        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below
        if avg_last:
            assert num_features_per_layer[-1] == d_model
        else:
            assert num_features_per_layer[-1] * num_heads_per_layer[-1] == d_model
        num_of_layers = len(num_heads_per_layer) - 1

        gat_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            if i == num_of_layers - 1:
                if avg_last:
                    concat_input = False
                else:
                    concat_input = True
            else:
                concat_input = True
            layer = GATLayerImp3(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
                num_out_features=num_features_per_layer[i+1],
                num_of_heads=num_heads_per_layer[i+1],
                concat=concat_input,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                load_trans_prob=load_trans_prob
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    def forward(self, node_features, edge_index_input, edge_prob_input, x):
        """

        Args:
            node_features: (vocab_size, fea_dim)
            edge_index_input: (2, E)
            edge_prob_input: (E, 1)
            x: (B, T)

        Returns:
            (B, T, d_model)

        """
        data = (node_features, edge_index_input, edge_prob_input)
        (node_fea_emb, edge_index, edge_prob) = self.gat_net(data)  # (vocab_size, num_channels[-1]), (2, E)
        batch_size, seq_len = x.shape
        node_fea_emb = node_fea_emb.expand((batch_size, -1, -1))  # (B, vocab_size, d_model)
        node_fea_emb = node_fea_emb.reshape(-1, self.d_model)  # (B * vocab_size, d_model)
        x = x.reshape(-1, 1).squeeze(1)  # (B * T,)
        out_node_fea_emb = node_fea_emb[x].reshape(batch_size, seq_len, self.d_model)  # (B, T, d_model)
        return out_node_fea_emb  # (B, T, d_model)


class BERTEmbedding(nn.Module):

    def __init__(self, d_model, dropout=0.1, add_time_in_day=False, add_day_in_week=False,
                 add_pe=True, node_fea_dim=10, add_gat=True,
                 gat_heads_per_layer=None, gat_features_per_layer=None, gat_dropout=0.6,
                 load_trans_prob=True, avg_last=True):
        """

        Args:
            vocab_size: total vocab size
            d_model: embedding size of token embedding
            dropout: dropout rate
        """
        super().__init__()
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week
        self.add_pe = add_pe
        self.add_gat = add_gat

        if self.add_gat:
            self.token_embedding = GAT(d_model=d_model, in_feature=node_fea_dim,
                                   num_heads_per_layer=gat_heads_per_layer,
                                   num_features_per_layer=gat_features_per_layer,
                                   add_skip_connection=True, bias=True, dropout=gat_dropout,
                                   load_trans_prob=load_trans_prob, avg_last=avg_last)
        if self.add_pe:
            self.position_embedding = PositionalEmbedding(d_model=d_model)
        if self.add_time_in_day:
            self.daytime_embedding = nn.Embedding(1441, d_model, padding_idx=0)
        if self.add_day_in_week:
            self.weekday_embedding = nn.Embedding(8, d_model, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, sequence, position_ids=None, graph_dict=None):
        """

        Args:
            sequence: (B, T, F) [loc, ts, mins, weeks, usr]
            position_ids: (B, T) or None
            graph_dict(dict): including:
                in_lap_mx: (vocab_size, lap_dim)
                out_lap_mx: (vocab_size, lap_dim)
                indegree: (vocab_size, )
                outdegree: (vocab_size, )

        Returns:
            (B, T, d_model)

        """
        if self.add_gat:
            x = self.token_embedding(node_features=graph_dict['node_features'],
                                         edge_index_input=graph_dict['edge_index'],
                                         edge_prob_input=graph_dict['loc_trans_prob'],
                                         x=sequence[:, :, 0])  # (B, T, d_model)
        if self.add_pe:
            x += self.position_embedding(x, position_ids)  # (B, T, d_model)
        if self.add_time_in_day:
            x += self.daytime_embedding(sequence[:, :, 2])  # (B, T, d_model)
        if self.add_day_in_week:
            x += self.weekday_embedding(sequence[:, :, 3])  # (B, T, d_model)
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):

    def __init__(self, num_heads, d_model, dim_out, attn_drop=0., proj_drop=0.,
                 add_cls=True, device=torch.device('cpu'), add_temporal_bias=True,
                 temporal_bias_dim=64, use_mins_interval=False):
        super().__init__()
        assert d_model % num_heads == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.device = device
        self.add_cls = add_cls
        self.scale = self.d_k ** -0.5  # 1/sqrt(dk)
        self.add_temporal_bias = add_temporal_bias
        self.temporal_bias_dim = temporal_bias_dim
        self.use_mins_interval = use_mins_interval

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.dropout = nn.Dropout(p=attn_drop)

        self.proj = nn.Linear(d_model, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.add_temporal_bias:
            if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
                self.temporal_mat_bias_1 = nn.Linear(1, self.temporal_bias_dim, bias=True)
                self.temporal_mat_bias_2 = nn.Linear(self.temporal_bias_dim, 1, bias=True)
            elif self.temporal_bias_dim == -1:
                self.temporal_mat_bias = nn.Parameter(torch.Tensor(1, 1))
                nn.init.xavier_uniform_(self.temporal_mat_bias)

    def forward(self, x, padding_masks, future_mask=True, output_attentions=False, batch_temporal_mat=None):
        """

        Args:
            x: (B, T, d_model)
            padding_masks: (B, 1, T, T) padding_mask
            future_mask: True/False
            batch_temporal_mat: (B, T, T)

        Returns:

        """
        batch_size, seq_len, d_model = x.shape

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # l(x) --> (B, T, d_model)
        # l(x).view() --> (B, T, head, d_k)
        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (x, x, x))]
        # q, k, v --> (B, head, T, d_k)

        # 2) Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # (B, head, T, T)

        if self.add_temporal_bias:
            if self.use_mins_interval:
                batch_temporal_mat = 1.0 / torch.log(
                    torch.exp(torch.tensor(1.0).to(self.device)) +
                    (batch_temporal_mat / torch.tensor(60.0).to(self.device)))
            else:
                batch_temporal_mat = 1.0 / torch.log(
                    torch.exp(torch.tensor(1.0).to(self.device)) + batch_temporal_mat)
            if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
                batch_temporal_mat = self.temporal_mat_bias_2(F.leaky_relu(
                    self.temporal_mat_bias_1(batch_temporal_mat.unsqueeze(-1)),
                    negative_slope=0.2)).squeeze(-1)  # (B, T, T)
            if self.temporal_bias_dim == -1:
                batch_temporal_mat = batch_temporal_mat * self.temporal_mat_bias.expand((1, seq_len, seq_len))
            batch_temporal_mat = batch_temporal_mat.unsqueeze(1)  # (B, 1, T, T)
            scores += batch_temporal_mat  # (B, 1, T, T)

        if padding_masks is not None:
            scores.masked_fill_(padding_masks == 0, float('-inf'))

        if future_mask:
            mask_postion = torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1).bool().to(self.device)
            if self.add_cls:
                mask_postion[:, 0, :] = 0
            scores.masked_fill_(mask_postion, float('-inf'))

        p_attn = F.softmax(scores, dim=-1)  # (B, head, T, T)
        p_attn = self.dropout(p_attn)
        out = torch.matmul(p_attn, value)  # (B, head, T, d_k)

        # 3) "Concat" using a view and apply a final linear.
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)  # (B, T, d_model)
        out = self.proj(out)  # (B, T, N, D)
        out = self.proj_drop(out)
        if output_attentions:
            return out, p_attn  # (B, T, dim_out), (B, head, T, T)
        else:
            return out, None  # (B, T, dim_out)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        """Position-wise Feed-Forward Networks
        Args:
            in_features:
            hidden_features:
            out_features:
            act_layer:
            drop:
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, d_model, attn_heads, feed_forward_hidden, drop_path,
                 attn_drop, dropout, type_ln='pre', add_cls=True,
                 device=torch.device('cpu'), add_temporal_bias=True,
                 temporal_bias_dim=64, use_mins_interval=False):
        """

        Args:
            d_model: hidden size of transformer
            attn_heads: head sizes of multi-head attention
            feed_forward_hidden: feed_forward_hidden, usually 4*d_model
            drop_path: encoder dropout rate
            attn_drop: attn dropout rate
            dropout: dropout rate
            type_ln:
        """

        super().__init__()
        self.attention = MultiHeadedAttention(num_heads=attn_heads, d_model=d_model, dim_out=d_model,
                                              attn_drop=attn_drop, proj_drop=dropout, add_cls=add_cls,
                                              device=device, add_temporal_bias=add_temporal_bias,
                                              temporal_bias_dim=temporal_bias_dim,
                                              use_mins_interval=use_mins_interval)
        self.mlp = Mlp(in_features=d_model, hidden_features=feed_forward_hidden,
                       out_features=d_model, act_layer=nn.GELU, drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.type_ln = type_ln

    def forward(self, x, padding_masks, future_mask=True, output_attentions=False, batch_temporal_mat=None):
        """

        Args:
            x: (B, T, d_model)
            padding_masks: (B, 1, T, T)
            future_mask: True/False
            batch_temporal_mat: (B, T, T)

        Returns:
            (B, T, d_model)

        """
        if self.type_ln == 'pre':
            attn_out, attn_score = self.attention(self.norm1(x), padding_masks=padding_masks,
                                                  future_mask=future_mask, output_attentions=output_attentions,
                                                  batch_temporal_mat=batch_temporal_mat)
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            attn_out, attn_score = self.attention(x, padding_masks=padding_masks,
                                                  future_mask=future_mask, output_attentions=output_attentions,
                                                  batch_temporal_mat=batch_temporal_mat)
            x = self.norm1(x + self.drop_path(attn_out))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            raise ValueError('Error type_ln {}'.format(self.type_ln))
        return x, attn_score


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, config, data_feature):
        """
        Args:
        """
        super().__init__()

        self.config = config

        self.vocab_size = data_feature.get('vocab_size')
        self.usr_num = data_feature.get('usr_num')
        self.node_fea_dim = data_feature.get('node_fea_dim')

        self.d_model = self.config.get('d_model', 768)
        self.n_layers = self.config.get('n_layers', 12)
        self.attn_heads = self.config.get('attn_heads', 12)
        self.mlp_ratio = self.config.get('mlp_ratio', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.drop_path = self.config.get('drop_path', 0.3)
        self.lape_dim = self.config.get('lape_dim', 256)
        self.attn_drop = self.config.get('attn_drop', 0.1)
        self.type_ln = self.config.get('type_ln', 'pre')
        self.future_mask = self.config.get('future_mask', False)
        self.add_cls = self.config.get('add_cls', False)
        self.device = self.config.get('device', torch.device('cpu'))
        self.cutoff_row_rate = self.config.get('cutoff_row_rate', 0.2)
        self.cutoff_column_rate = self.config.get('cutoff_column_rate', 0.2)
        self.cutoff_random_rate = self.config.get('cutoff_random_rate', 0.2)
        self.sample_rate = self.config.get('sample_rate', 0.2)
        self.device = self.config.get('device', torch.device('cpu'))
        self.add_time_in_day = self.config.get('add_time_in_day', True)
        self.add_day_in_week = self.config.get('add_day_in_week', True)
        self.add_pe = self.config.get('add_pe', True)
        self.add_gat = self.config.get('add_gat', True)
        self.gat_heads_per_layer = self.config.get('gat_heads_per_layer', [8, 1])
        self.gat_features_per_layer = self.config.get('gat_features_per_layer', [16, self.d_model])
        self.gat_dropout = self.config.get('gat_dropout', 0.6)
        self.gat_avg_last = self.config.get('gat_avg_last', True)
        self.load_trans_prob = self.config.get('load_trans_prob', False)
        self.add_temporal_bias = self.config.get('add_temporal_bias', True)
        self.temporal_bias_dim = self.config.get('temporal_bias_dim', 64)
        self.use_mins_interval = self.config.get('use_mins_interval', False)

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = self.d_model * self.mlp_ratio

        # embedding for BERT, sum of ... embeddings
        self.embedding = BERTEmbedding(d_model=self.d_model, dropout=self.dropout,
                                       add_time_in_day=self.add_time_in_day, add_day_in_week=self.add_day_in_week,
                                       add_pe=self.add_pe, node_fea_dim=self.node_fea_dim, add_gat=self.add_gat,
                                       gat_heads_per_layer=self.gat_heads_per_layer,
                                       gat_features_per_layer=self.gat_features_per_layer, gat_dropout=self.gat_dropout,
                                       load_trans_prob=self.load_trans_prob, avg_last=self.gat_avg_last)

        # multi-layers transformer blocks, deep network
        enc_dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.n_layers)]  # stochastic depth decay rule
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model=self.d_model, attn_heads=self.attn_heads,
                              feed_forward_hidden=self.feed_forward_hidden, drop_path=enc_dpr[i],
                              attn_drop=self.attn_drop, dropout=self.dropout,
                              type_ln=self.type_ln, add_cls=self.add_cls,
                              device=self.device, add_temporal_bias=self.add_temporal_bias,
                              temporal_bias_dim=self.temporal_bias_dim,
                              use_mins_interval=self.use_mins_interval) for i in range(self.n_layers)])

    def _shuffle_position_ids(self, x, padding_masks, position_ids):
        batch_size, seq_len, feat_dim = x.shape
        if position_ids is None:
            position_ids = torch.arange(512).expand((batch_size, -1))[:, :seq_len].to(device=self.device)

        # shuffle position_ids
        shuffled_pid = []
        for bsz_id in range(batch_size):
            sample_pid = position_ids[bsz_id]  # (512, )
            sample_mask = padding_masks[bsz_id]  # (seq_length, )
            num_tokens = sample_mask.sum().int().item()
            indexes = list(range(num_tokens))
            random.shuffle(indexes)
            rest_indexes = list(range(num_tokens, seq_len))
            total_indexes = indexes + rest_indexes
            shuffled_pid.append(torch.index_select(sample_pid, 0, torch.tensor(total_indexes).to(
                device=self.device)).unsqueeze(0))
        shuffled_pid = torch.cat(shuffled_pid, 0).to(device=self.device)   # (batch_size, seq_length)
        return shuffled_pid

    def _sample_span(self, x, padding_masks, sample_rate=0.2):
        batch_size, seq_len, feat_dim = x.shape

        if sample_rate > 0:
            true_seq_len = padding_masks.sum(1).cpu().numpy()
            mask = []
            for true_len in true_seq_len:
                sample_len = max(int(true_len * (1 - sample_rate)), 1)
                start_id = np.random.randint(0, high=true_len - sample_len + 1)
                tmp = [1] * seq_len
                for idx in range(start_id, start_id + sample_len):
                    tmp[idx] = 0
                mask.append(tmp)
            mask = torch.ByteTensor(mask).bool().to(self.device)
            x = x.masked_fill(mask.unsqueeze(-1), value=0.)
            padding_masks = padding_masks.masked_fill(mask, value=0)

        return x, padding_masks

    def _cutoff_embeddings(self, embedding_output, padding_masks, direction, cutoff_rate=0.2):
        batch_size, seq_len, d_model = embedding_output.shape
        cutoff_embeddings = []
        for bsz_id in range(batch_size):
            sample_embedding = embedding_output[bsz_id]
            sample_mask = padding_masks[bsz_id]
            if direction == "row":
                num_dimensions = sample_mask.sum().int().item()  # number of tokens
                dim_index = 0
            elif direction == "column":
                num_dimensions = d_model
                dim_index = 1
            elif direction == "random":
                num_dimensions = sample_mask.sum().int().item() * d_model
                dim_index = 0
            else:
                raise ValueError(f"direction should be either row or column, but got {direction}")
            num_cutoff_indexes = int(num_dimensions * cutoff_rate)
            if num_cutoff_indexes < 0 or num_cutoff_indexes > num_dimensions:
                raise ValueError(f"number of cutoff dimensions should be in (0, {num_dimensions}), but got {num_cutoff_indexes}")
            indexes = list(range(num_dimensions))
            random.shuffle(indexes)
            cutoff_indexes = indexes[:num_cutoff_indexes]
            if direction == "random":
                sample_embedding = sample_embedding.reshape(-1)  # (seq_length * d_model, )
            cutoff_embedding = torch.index_fill(sample_embedding, dim_index, torch.tensor(
                cutoff_indexes, dtype=torch.long).to(device=self.device), 0.0)
            if direction == "random":
                cutoff_embedding = cutoff_embedding.reshape(seq_len, d_model)
            cutoff_embeddings.append(cutoff_embedding.unsqueeze(0))
        cutoff_embeddings = torch.cat(cutoff_embeddings, 0).to(device=self.device)  # (batch_size, seq_length, d_model)
        assert cutoff_embeddings.shape == embedding_output.shape
        return cutoff_embeddings

    def _shuffle_embeddings(self, embedding_output, padding_masks):
        batch_size, seq_len, d_model = embedding_output.shape
        shuffled_embeddings = []
        for bsz_id in range(batch_size):
            sample_embedding = embedding_output[bsz_id]  # (seq_length, d_model)
            sample_mask = padding_masks[bsz_id]  # (seq_length, )
            num_tokens = sample_mask.sum().int().item()
            indexes = list(range(num_tokens))
            random.shuffle(indexes)
            rest_indexes = list(range(num_tokens, seq_len))
            total_indexes = indexes + rest_indexes
            shuffled_embeddings.append(torch.index_select(sample_embedding, 0, torch.tensor(total_indexes).to(
                device=self.device)).unsqueeze(0))
        shuffled_embeddings = torch.cat(shuffled_embeddings, 0).to(device=self.device)  # (batch_size, seq_length, d_model)
        return shuffled_embeddings

    def forward(self, x, padding_masks, batch_temporal_mat=None, argument_methods=None, graph_dict=None,
                output_hidden_states=False, output_attentions=False):
        """
        Args:
            x: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """
        position_ids = None

        if argument_methods is not None:
            if 'shuffle_position' in argument_methods:
                position_ids = self._shuffle_position_ids(
                    x=x, padding_masks=padding_masks, position_ids=None)
            if 'span' in argument_methods:
                x, attention_mask = self._sample_span(
                    x=x, padding_masks=padding_masks, sample_rate=self.sample_rate)

        # embedding the indexed sequence to sequence of vectors
        embedding_output = self.embedding(sequence=x, position_ids=position_ids,
                                          graph_dict=graph_dict)  # (B, T, d_model)

        if argument_methods is not None:
            if 'cutoff_row' in argument_methods:
                embedding_output = self._cutoff_embeddings(
                    embedding_output=embedding_output, padding_masks=padding_masks,
                    direction='row', cutoff_rate=self.cutoff_row_rate)
            if 'cutoff_column' in argument_methods:
                embedding_output = self._cutoff_embeddings(
                    embedding_output=embedding_output, padding_masks=padding_masks,
                    direction='column', cutoff_rate=self.cutoff_column_rate)
            if 'cutoff_random' in argument_methods:
                embedding_output = self._cutoff_embeddings(
                    embedding_output=embedding_output, padding_masks=padding_masks,
                    direction='random', cutoff_rate=self.cutoff_random_rate)
            if 'shuffle_embedding' in argument_methods:
                embedding_output = self._shuffle_embeddings(
                    embedding_output=embedding_output, padding_masks=padding_masks)

        padding_masks_input = padding_masks.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)  # (B, 1, T, T)
        # running over multiple transformer blocks
        all_hidden_states = [embedding_output] if output_hidden_states else None
        all_self_attentions = [] if output_attentions else None
        for transformer in self.transformer_blocks:
            embedding_output, attn_score = transformer.forward(
                x=embedding_output, padding_masks=padding_masks_input,
                future_mask=self.future_mask, output_attentions=output_attentions,
                batch_temporal_mat=batch_temporal_mat)  # (B, T, d_model)
            if output_hidden_states:
                all_hidden_states.append(embedding_output)
            if output_attentions:
                all_self_attentions.append(attn_score)
        return embedding_output, all_hidden_states, all_self_attentions  # (B, T, d_model), list of (B, T, d_model), list of (B, head, T, T)


class BERTDownstream(nn.Module):

    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config

        self.vocab_size = data_feature.get('vocab_size')
        self.usr_num = data_feature.get('usr_num')
        self.pooling = self.config.get('pooling', 'mean')
        self.d_model = self.config.get('d_model', 768)
        self.add_cls = self.config.get('add_cls', True)
        self.baseline_bert = self.config.get('baseline_bert', False)
        self.baseline_tf = self.config.get('baseline_tf', False)

        self._logger = getLogger()
        self._logger.info("Building BERTDownstream model")

        self.bert = BERT(config, data_feature)

    def forward(self, x, padding_masks, batch_temporal_mat=None,
                graph_dict=None, output_hidden_states=False, output_attentions=False):
        """
        Args:
            x: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, feat_dim)
        """
        if self.pooling in ['avg_first_last', 'avg_top2']:
            output_hidden_states = True
        token_emb, hidden_states, _ = self.bert(x=x, padding_masks=padding_masks,
                                                batch_temporal_mat=batch_temporal_mat, graph_dict=graph_dict,
                                                output_hidden_states=output_hidden_states,
                                                output_attentions=output_attentions)  # (batch_size, seq_length, d_model)
        if self.pooling == 'cls' or self.pooling == 'cls_before_pooler':
            if self.add_cls:
                return token_emb[:, 0, :]  # (batch_size, feat_dim)
            else:
                raise ValueError('No use cls!')
        elif self.pooling == 'mean':
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                token_emb.size()).float()  # (batch_size, seq_length, d_model)
            sum_embeddings = torch.sum(token_emb * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask  # (batch_size, feat_dim)
        elif self.pooling == 'max':
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                token_emb.size()).float()  # (batch_size, seq_length, d_model)
            token_emb[input_mask_expanded == 0] = float('-inf')  # Set padding tokens to large negative value
            max_over_time = torch.max(token_emb, 1)[0]
            return max_over_time  # (batch_size, feat_dim)
        elif self.pooling == "avg_first_last":
            first_hidden = hidden_states[0]  # (batch_size, seq_length, d_model)
            last_hidden = hidden_states[-1]  # (batch_size, seq_length, d_model)
            avg_emb = (first_hidden + last_hidden) / 2.0
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                avg_emb.size()).float()  # (batch_size, seq_length, d_model)
            sum_embeddings = torch.sum(avg_emb * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask  # (batch_size, feat_dim)
        elif self.pooling == "avg_top2":
            second_last_hidden = hidden_states[-2]  # (batch_size, seq_length, d_model)
            last_hidden = hidden_states[-1]  # (batch_size, seq_length, d_model)
            avg_emb = (second_last_hidden + last_hidden) / 2.0
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                avg_emb.size()).float()  # (batch_size, seq_length, d_model)
            sum_embeddings = torch.sum(avg_emb * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask  # (batch_size, feat_dim)
        else:
            raise ValueError('Error pooling type {}'.format(self.pooling))


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, d_model):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

    def forward(self, features):
        x = self.dense(features)
        x = self.activation(x)
        return x


class BERTPooler(nn.Module):

    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config

        self.pooling = self.config.get('pooling', 'mean')
        self.add_cls = self.config.get('add_cls', True)
        self.d_model = self.config.get('d_model', 768)
        self.linear = MLPLayer(d_model=self.d_model)

        self._logger = getLogger()
        self._logger.info("Building BERTPooler model")

    def forward(self, bert_output, padding_masks, hidden_states=None):
        """
        Args:
            bert_output: (batch_size, seq_length, d_model) torch tensor of bert output
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
            hidden_states: list of hidden, (batch_size, seq_length, d_model)
        Returns:
            output: (batch_size, feat_dim)
        """
        token_emb = bert_output  # (batch_size, seq_length, d_model)
        if self.pooling == 'cls':
            if self.add_cls:
                return self.linear(token_emb[:, 0, :])  # (batch_size, feat_dim)
            else:
                raise ValueError('No use cls!')
        elif self.pooling == 'cls_before_pooler':
            if self.add_cls:
                return token_emb[:, 0, :]  # (batch_size, feat_dim)
            else:
                raise ValueError('No use cls!')
        elif self.pooling == 'mean':
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                token_emb.size()).float()  # (batch_size, seq_length, d_model)
            sum_embeddings = torch.sum(token_emb * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask  # (batch_size, feat_dim)
        elif self.pooling == 'max':
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                token_emb.size()).float()  # (batch_size, seq_length, d_model)
            token_emb[input_mask_expanded == 0] = float('-inf')  # Set padding tokens to large negative value
            max_over_time = torch.max(token_emb, 1)[0]
            return max_over_time  # (batch_size, feat_dim)
        elif self.pooling == "avg_first_last":
            first_hidden = hidden_states[0]  # (batch_size, seq_length, d_model)
            last_hidden = hidden_states[-1]  # (batch_size, seq_length, d_model)
            avg_emb = (first_hidden + last_hidden) / 2.0
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                avg_emb.size()).float()  # (batch_size, seq_length, d_model)
            sum_embeddings = torch.sum(avg_emb * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask  # (batch_size, feat_dim)
        elif self.pooling == "avg_top2":
            second_last_hidden = hidden_states[-2]  # (batch_size, seq_length, d_model)
            last_hidden = hidden_states[-1]  # (batch_size, seq_length, d_model)
            avg_emb = (second_last_hidden + last_hidden) / 2.0
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                avg_emb.size()).float()  # (batch_size, seq_length, d_model)
            sum_embeddings = torch.sum(avg_emb * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask  # (batch_size, feat_dim)
        else:
            raise ValueError('Error pooling type {}'.format(self.pooling))


class LinearNextLoc(nn.Module):
    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config

        self.vocab_size = data_feature.get('vocab_size')
        self.usr_num = data_feature.get('usr_num')
        self.pooling = self.config.get('pooling', 'mean')
        self.d_model = self.config.get('d_model', 768)

        self._logger = getLogger()
        self._logger.info("Building Downstream LinearNextLoc model")

        self.model = BERTDownstream(config, data_feature)
        self.linear = nn.Linear(self.d_model, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, padding_masks, batch_temporal_mat=None,
                graph_dict=None, output_hidden_states=False, output_attentions=False):
        traj_emb = self.model(x=x, padding_masks=padding_masks,
                              batch_temporal_mat=batch_temporal_mat, graph_dict=graph_dict,
                              output_hidden_states=output_hidden_states,
                              output_attentions=output_attentions)  # (B, d_model)
        nloc_pred = self.softmax(self.linear(traj_emb))  # (B, n_class)
        return nloc_pred  # (B, n_class)


class LinearETA(nn.Module):
    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config

        self.vocab_size = data_feature.get('vocab_size')
        self.usr_num = data_feature.get('usr_num')
        self.pooling = self.config.get('pooling', 'mean')
        self.d_model = self.config.get('d_model', 768)

        self._logger = getLogger()
        self._logger.info("Building Downstream LinearETA model")

        self.model = BERTDownstream(config, data_feature)
        self.linear = nn.Linear(self.d_model, 1)

    def forward(self, x, padding_masks, batch_temporal_mat=None,
                graph_dict=None, output_hidden_states=False, output_attentions=False):
        traj_emb = self.model(x=x, padding_masks=padding_masks,
                              batch_temporal_mat=batch_temporal_mat, graph_dict=graph_dict,
                              output_hidden_states=output_hidden_states,
                              output_attentions=output_attentions)  # (B, d_model)
        eta_pred = self.linear(traj_emb)  # (B, 1)
        return eta_pred  # (B, 1)


class LinearClassify(nn.Module):
    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config

        self.usr_num = data_feature.get('usr_num')
        self.d_model = self.config.get('d_model', 768)
        self.dataset = self.config.get('dataset', '')
        self.classify_label = self.config.get('classify_label', 'vflag')

        self._logger = getLogger()
        self._logger.info("Building Downstream LinearClassify model")

        self.model = BERTDownstream(config, data_feature)
        if self.classify_label == 'vflag':
            self.linear = nn.Linear(self.d_model, 2)
            if self.dataset == 'geolife':
                self.linear = nn.Linear(self.d_model, 4)
        elif self.classify_label == 'usrid':
            self.linear = nn.Linear(self.d_model, self.usr_num)
        else:
            raise ValueError('Error classify_label = {}'.format(self.classify_label))
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, padding_masks, batch_temporal_mat=None,
                graph_dict=None, output_hidden_states=False, output_attentions=False):
        traj_emb = self.model(x=x, padding_masks=padding_masks,
                              batch_temporal_mat=batch_temporal_mat, graph_dict=graph_dict,
                              output_hidden_states=output_hidden_states,
                              output_attentions=output_attentions)  # (B, d_model)
        nloc_pred = self.softmax(self.linear(traj_emb))  # (B, n_class)
        return nloc_pred  # (B, n_class)


class LinearSim(nn.Module):
    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config

        self.usr_num = data_feature.get('usr_num')
        self.d_model = self.config.get('d_model', 768)

        self._logger = getLogger()
        self._logger.info("Building Downstream LinearSim model")

        self.model = BERTDownstream(config, data_feature)

    def forward(self, x, padding_masks, batch_temporal_mat=None,
                graph_dict=None, output_hidden_states=False, output_attentions=False):
        traj_emb = self.model(x=x, padding_masks=padding_masks,
                              batch_temporal_mat=batch_temporal_mat, graph_dict=graph_dict,
                              output_hidden_states=output_hidden_states,
                              output_attentions=output_attentions)  # (B, d_model)
        return traj_emb  # (B, d_model)


class BERTLM(nn.Module):

    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config

        self.vocab_size = data_feature.get('vocab_size')
        self.usr_num = data_feature.get('usr_num')
        self.d_model = self.config.get('d_model', 768)

        self._logger = getLogger()
        self._logger.info("Building BERTLM model")

        self.bert = BERT(config, data_feature)

        self.mask_l = MaskedLanguageModel(self.d_model, self.vocab_size)

    def forward(self, x, padding_masks, batch_temporal_mat=None,
                graph_dict=None, output_hidden_states=False, output_attentions=False):
        """
        Args:
            x: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, vocab_size)
        """

        x, _, _ = self.bert(x=x, padding_masks=padding_masks,
                            batch_temporal_mat=batch_temporal_mat, graph_dict=graph_dict,
                            output_hidden_states=output_hidden_states,
                            output_attentions=output_attentions)  # (B, T, d_model)
        a = self.mask_l(x)
        return a


class BERTContrastiveLM(nn.Module):

    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config

        self.vocab_size = data_feature.get('vocab_size')
        self.usr_num = data_feature.get('usr_num')
        self.d_model = self.config.get('d_model', 768)
        self.pooling = self.config.get('pooling', 'mean')

        self._logger = getLogger()
        self._logger.info("Building BERTContrastiveLM model")

        self.bert = BERT(config, data_feature)

        self.mask_l = MaskedLanguageModel(self.d_model, self.vocab_size)
        self.pooler = BERTPooler(config, data_feature)

    def forward(self, contra_view1, contra_view2, argument_methods1,
                argument_methods2, masked_input, padding_masks,
                batch_temporal_mat, padding_masks1=None, padding_masks2=None,
                batch_temporal_mat1=None, batch_temporal_mat2=None,
                graph_dict=None, output_hidden_states=False, output_attentions=False):
        """
        Args:
            contra_view1: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            contra_view2: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            masked_input: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
            graph_dict(dict):
        Returns:
            output: (batch_size, seq_length, vocab_size)
        """
        if self.pooling in ['avg_first_last', 'avg_top2']:
            output_hidden_states = True
        if padding_masks1 is None:
            padding_masks1 = padding_masks
        if padding_masks2 is None:
            padding_masks2 = padding_masks
        if batch_temporal_mat1 is None:
            batch_temporal_mat1 = batch_temporal_mat
        if batch_temporal_mat2 is None:
            batch_temporal_mat2 = batch_temporal_mat

        out_view1, hidden_states1, _ = self.bert(x=contra_view1, padding_masks=padding_masks1,
                                                 batch_temporal_mat=batch_temporal_mat1,
                                                 argument_methods=argument_methods1, graph_dict=graph_dict,
                                                 output_hidden_states=output_hidden_states,
                                                 output_attentions=output_attentions)  # (B, T, d_model)
        pool_out_view1 = self.pooler(bert_output=out_view1, padding_masks=padding_masks1,
                                     hidden_states=hidden_states1)  # (B, d_model)

        out_view2, hidden_states2, _ = self.bert(x=contra_view2, padding_masks=padding_masks2,
                                                 batch_temporal_mat=batch_temporal_mat2,
                                                 argument_methods=argument_methods2, graph_dict=graph_dict,
                                                 output_hidden_states=output_hidden_states,
                                                 output_attentions=output_attentions)  # (B, T, d_model)
        pool_out_view2 = self.pooler(bert_output=out_view2, padding_masks=padding_masks2,
                                     hidden_states=hidden_states2)  # (B, d_model)

        bert_output, _, _ = self.bert(x=masked_input, padding_masks=padding_masks,
                                      batch_temporal_mat=batch_temporal_mat, graph_dict=graph_dict,
                                      output_hidden_states=output_hidden_states,
                                      output_attentions=output_attentions)  # (B, T, d_model)
        a = self.mask_l(bert_output)
        return pool_out_view1, pool_out_view2, a


class BERTContrastive(nn.Module):

    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config
        self.pooling = self.config.get('pooling', 'mean')

        self._logger = getLogger()
        self._logger.info("Building BERTContrastive model")

        self.bert = BERT(config, data_feature)
        self.pooler = BERTPooler(config, data_feature)

    def forward(self, x, padding_masks, argument_methods, batch_temporal_mat=None,
                graph_dict=None, output_hidden_states=False, output_attentions=False):
        """
        Args:
            x: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, d_model)
        """
        x, hidden_states, _ = self.bert(x=x, padding_masks=padding_masks,
                                        argument_methods=argument_methods,
                                        batch_temporal_mat=batch_temporal_mat, graph_dict=graph_dict,
                                        output_hidden_states=output_hidden_states,
                                        output_attentions=output_attentions)  # (B, T, d_model)
        x = self.pooler(bert_output=x, padding_masks=padding_masks, hidden_states=hidden_states)  # (B, d_model)
        return x


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """

        Args:
            hidden: output size of BERT model
            vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
