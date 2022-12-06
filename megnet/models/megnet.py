"""
Implementation of MEGNet model.
"""
from __future__ import annotations

import dgl
import torch
import torch.nn as nn
from dgl.nn import Set2Set
from torch.nn import Dropout, Identity, Module, ModuleList

from .helper import MLP, EdgeSet2Set
from .layers import MEGNetBlock


class MEGNet(Module):
    """
    DGL implementation of MEGNet.
    """

    def __init__(
        self,
        in_dim: int,
        num_blocks: int,
        hiddens: list[int],
        conv_hiddens: list[int],
        s2s_num_layers: int,
        s2s_num_iters: int,
        output_hiddens: list[int],
        act: str = "swish",
        is_classification: bool = True,
        node_embed: Module | None = None,
        edge_embed: Module | None = None,
        attr_embed: Module | None = None,
        dropout: float | None = None,
        graph_transformations: list | None = None,
    ) -> None:
        """
        TODO: Add docs.
        :param in_dim:
        :param num_blocks:
        :param hiddens:
        :param conv_hiddens:
        :param s2s_num_layers:
        :param s2s_num_iters:
        :param output_hiddens:
        :param act:
        :param is_classification:
        :param node_embed:
        :param edge_embed:
        :param attr_embed:
        :param dropout:
        :param graph_transform: Perform a graph transformation, e.g., incorporate three-body interactions, prior to
            performing the GCL updates.
        """
        super().__init__()

        self.edge_embed = edge_embed if edge_embed else Identity()
        self.node_embed = node_embed if node_embed else Identity()
        self.attr_embed = attr_embed if attr_embed else Identity()

        dims = [in_dim] + hiddens

        if act == "swish":
            activation = nn.SiLU()
        elif act == "sigmoid":
            activation = nn.Sigmoid()
        elif act == "tanh":
            activation = nn.Tanh()
        else:
            raise Exception("Undefined activation type, please try using swish, sigmoid, tanh")

        self.edge_encoder = MLP(dims, activation, activate_last=True)
        self.node_encoder = MLP(dims, activation, activate_last=True)
        self.attr_encoder = MLP(dims, activation, activate_last=True)

        blocks_in_dim = hiddens[-1]
        block_out_dim = conv_hiddens[-1]
        block_args = dict(conv_hiddens=conv_hiddens, dropout=dropout, act=activation, skip=True)
        blocks = []

        # first block
        blocks.append(MEGNetBlock(dims=[blocks_in_dim], **block_args))  # type: ignore
        # other blocks
        for _ in range(num_blocks - 1):
            blocks.append(MEGNetBlock(dims=[block_out_dim] + hiddens, **block_args))  # type: ignore
        self.blocks = ModuleList(blocks)

        s2s_kwargs = dict(n_iters=s2s_num_iters, n_layers=s2s_num_layers)
        self.edge_s2s = EdgeSet2Set(block_out_dim, **s2s_kwargs)
        self.node_s2s = Set2Set(block_out_dim, **s2s_kwargs)

        self.output_proj = MLP(
            # S2S cats q_star to output producing double the dim
            dims=[2 * 2 * block_out_dim + block_out_dim] + output_hiddens + [1],
            activation=activation,
            activate_last=False,
        )

        self.dropout = Dropout(dropout) if dropout else None
        # TODO(marcel): should this be an 1D dropout

        self.is_classification = is_classification
        self.graph_transformations = graph_transformations or [Identity()] * num_blocks

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: torch.Tensor,
        node_feat: torch.Tensor,
        graph_attr: torch.Tensor,
    ):
        """
        TODO: Add docs.
        :param graph: Input graph
        :param edge_feat: Edge features
        :param node_feat: Node features
        :param graph_attr: Graph attributes / state features.
        :return: Prediction
        """

        graph_transformations = self.graph_transformations
        edge_feat = self.edge_encoder(self.edge_embed(edge_feat))
        node_feat = self.node_encoder(self.node_embed(node_feat))
        graph_attr = self.attr_encoder(self.attr_embed(graph_attr))

        for i, block in enumerate(self.blocks):
            graph = graph_transformations[i](graph)
            output = block(graph, edge_feat, node_feat, graph_attr)
            edge_feat, node_feat, graph_attr = output

        node_vec = self.node_s2s(graph, node_feat)
        edge_vec = self.edge_s2s(graph, edge_feat)

        vec = torch.hstack([node_vec, edge_vec, graph_attr])

        if self.dropout:
            vec = self.dropout(vec)  # pylint: disable=E1102

        output = self.output_proj(vec)
        if self.is_classification:
            output = torch.sigmoid(output)

        return output
