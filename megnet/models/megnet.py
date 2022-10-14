import torch
from torch.nn import Module, ModuleList, Identity, Softplus, Dropout
from dgl.nn import Set2Set, AvgPooling

from .helper import MLP, EdgeSet2Set, EdgeAvgPooling
from ..layers import MEGNetBlock
from ..types import List, Optional, DGLGraph, Tensor


class MEGNet(Module):
    def __init__(
        self,
        in_dim: int,
        num_blocks: int,
        hiddens: List[int],
        conv_hiddens: List[int],
        s2s_num_layers: int,
        s2s_num_iters: int,
        output_hiddens: List[int],
        is_classification: bool = True,
        node_embed: Optional[Module] = None,
        edge_embed: Optional[Module] = None,
        attr_embed: Optional[Module] = None,
        dropout: Optional[float] = None,
        readout: str = 'set2set',
    ) -> None:
        super().__init__()

        self.edge_embed = edge_embed if edge_embed else Identity()
        self.node_embed = node_embed if node_embed else Identity()
        self.attr_embed = attr_embed if attr_embed else Identity()

        dims = [in_dim] + hiddens
        self.edge_encoder = MLP(dims, Softplus(), activate_last=True)
        self.node_encoder = MLP(dims, Softplus(), activate_last=True)
        self.attr_encoder = MLP(dims, Softplus(), activate_last=True)

        blocks_in_dim = hiddens[-1]
        block_out_dim = conv_hiddens[-1]
        block_args = dict(conv_hiddens=conv_hiddens, dropout=dropout, skip=True)
        blocks = []
        # first block
        blocks.append(MEGNetBlock(dims=[blocks_in_dim], **block_args))  # type: ignore
        # other blocks
        for _ in range(num_blocks - 1):
            blocks.append(MEGNetBlock(dims=[block_out_dim] + hiddens, **block_args))  # type: ignore
        self.blocks = ModuleList(blocks)

        if readout.lower() == 'set2set':
            s2s_kwargs = dict(n_iters=s2s_num_iters, n_layers=s2s_num_layers)
            self.edge_readout = EdgeSet2Set(block_out_dim, **s2s_kwargs)
            self.node_readout = Set2Set(block_out_dim, **s2s_kwargs)
        elif readout.lower() == 'avgpooling':
            self.node_readout = AvgPooling()
            self.edge_readout = EdgeAvgPooling() 
        else:
            raise ValueError(f"Invalid readout value ({readout})")

        self.output_proj = MLP(
            # S2S cats q_star to output producing double the dim
            dims=[2 * 2 * block_out_dim + block_out_dim] + output_hiddens + [1],
            activation=Softplus(),
            activate_last=False,
        )

        self.dropout = Dropout(dropout) if dropout else None
        # TODO(marcel): should this be an 1D dropout

        self.is_classification = is_classification

    def forward(
        self,
        graph: DGLGraph,
        edge_feat: Tensor,
        node_feat: Tensor,
        graph_attr: Tensor,
    ) -> None:

        edge_feat = self.edge_encoder(self.edge_embed(edge_feat))
        node_feat = self.node_encoder(self.node_embed(node_feat))
        graph_attr = self.attr_encoder(self.attr_embed(graph_attr))

        for block in self.blocks:
            output = block(graph, edge_feat, node_feat, graph_attr)
            edge_feat, node_feat, graph_attr = output

        node_vec = self.node_readout(graph, node_feat)
        edge_vec = self.edge_readout(graph, edge_feat)

        vec = torch.hstack([node_vec, edge_vec, graph_attr])

        if self.dropout:
            vec = self.dropout(vec)

        output = self.output_proj(vec)
        if self.is_classification:
            output = torch.sigmoid(output)

        return output
