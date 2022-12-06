"""
Graph convolution layer (GCL) implementations.
"""
from __future__ import annotations

import dgl
import dgl.function as fn
import torch
from torch.nn import Dropout, Identity, Module

from .helper import MLP


class MEGNetGraphConv(Module):
    """
    A MEGNet graph convolution layer in DGL.
    """

    def __init__(
        self,
        edge_func: Module,
        node_func: Module,
        attr_func: Module,
    ) -> None:
        """
        :param edge_func: Edge update function.
        :param node_func: Node update function.
        :param attr_func: Global state update function.
        """

        super().__init__()
        self.edge_func = edge_func
        self.node_func = node_func
        self.attr_func = attr_func

    @staticmethod
    def from_dims(edge_dims: list[int], node_dims: list[int], attr_dims: list[int], activation) -> MEGNetGraphConv:
        """
        Initialize a GCL from dimensions based on MLP.

        :param edge_dims: Edge dimensions.
        :param node_dims: Node dimensions.
        :param attr_dims: Attribute dimensions.
        :return: MEGNetGraphConv.
        """
        # TODO(marcel): Softplus doesnt exactly match paper's SoftPlus2
        # TODO(marcel): Should we activate last?
        edge_update = MLP(edge_dims, activation, activate_last=True)
        node_update = MLP(node_dims, activation, activate_last=True)
        attr_update = MLP(attr_dims, activation, activate_last=True)
        return MEGNetGraphConv(edge_update, node_update, attr_update)

    def _edge_udf(self, edges: dgl.udf.EdgeBatch) -> torch.Tensor:
        vi = edges.src["v"]
        vj = edges.dst["v"]
        u = edges.src["u"]
        eij = edges.data.pop("e")
        inputs = torch.hstack([vi, vj, eij, u])
        mij = {"mij": self.edge_func(inputs)}
        return mij

    def edge_update_(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """
        Perform edge update.

        :param graph: Input graph
        :return: Output tensor for edges.
        """
        graph.apply_edges(self._edge_udf)
        graph.edata["e"] = graph.edata.pop("mij")
        return graph.edata["e"]

    def node_update_(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """
        Perform node update.

        :param graph: Input graph
        :return: Output tensor for nodes.
        """
        graph.update_all(fn.copy_e("e", "e"), fn.mean("e", "ve"))
        ve = graph.ndata.pop("ve")
        v = graph.ndata.pop("v")
        u = graph.ndata.pop("u")
        inputs = torch.hstack([v, ve, u])
        graph.ndata["v"] = self.node_func(inputs)
        return graph.ndata["v"]

    def attr_update_(self, graph: dgl.DGLGraph, attrs: torch.Tensor) -> torch.Tensor:
        """
        Perform attribute (global state) update.

        :param graph: Input graph
        :param attrs: Input attributes
        :return: Output tensor for attributes
        """
        u = attrs
        ue = dgl.readout_edges(graph, feat="e", op="mean")
        uv = dgl.readout_nodes(graph, feat="v", op="mean")
        inputs = torch.hstack([u, ue, uv])
        graph_attr = self.attr_func(inputs)
        return graph_attr

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: torch.Tensor,
        node_feat: torch.Tensor,
        graph_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform sequence of edge->node->attribute updates.

        :param graph: Input graph
        :param edge_feat: Edge features
        :param node_feat: Node features
        :param graph_attr: Graph attributes (global state)
        :return: (edge features, node features, graph attributes)
        """
        with graph.local_scope():
            graph.edata["e"] = edge_feat
            graph.ndata["v"] = node_feat
            graph.ndata["u"] = dgl.broadcast_nodes(graph, graph_attr)

            edge_feat = self.edge_update_(graph)
            node_feat = self.node_update_(graph)
            graph_attr = self.attr_update_(graph, graph_attr)

        return edge_feat, node_feat, graph_attr


class MEGNetBlock(Module):
    """
    A MEGNet block comprising a sequence of update operations.
    """

    def __init__(
        self,
        dims: list[int],
        conv_hiddens: list[int],
        act,
        dropout: float | None = None,
        skip: bool = True,
    ) -> None:
        """
        TODO: Add docs.
        :param dims:
        :param conv_hiddens:
        :param act:
        :param dropout:
        :param skip:
        """
        super().__init__()

        self.has_dense = len(dims) > 1
        self.activation = act
        conv_dim = dims[-1]
        out_dim = conv_hiddens[-1]

        mlp_kwargs = {
            "dims": dims,
            "activation": self.activation,
            "activate_last": True,
            "bias_last": True,
        }
        self.edge_func = MLP(**mlp_kwargs) if self.has_dense else Identity()
        self.node_func = MLP(**mlp_kwargs) if self.has_dense else Identity()
        self.attr_func = MLP(**mlp_kwargs) if self.has_dense else Identity()

        # compute input sizes
        edge_in = 2 * conv_dim + conv_dim + conv_dim  # 2*NDIM+EDIM+GDIM
        node_in = out_dim + conv_dim + conv_dim  # EDIM+NDIM+GDIM
        attr_in = out_dim + out_dim + conv_dim  # EDIM+NDIM+GDIM
        self.conv = MEGNetGraphConv.from_dims(
            edge_dims=[edge_in] + conv_hiddens,
            node_dims=[node_in] + conv_hiddens,
            attr_dims=[attr_in] + conv_hiddens,
            activation=self.activation,
        )

        self.dropout = Dropout(dropout) if dropout else None
        # TODO(marcel): should this be an 1D dropout
        self.skip = skip

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: torch.Tensor,
        node_feat: torch.Tensor,
        graph_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        TODO: Add docs.
        :param graph:
        :param edge_feat:
        :param node_feat:
        :param graph_attr:
        :return:
        """

        inputs = (edge_feat, node_feat, graph_attr)
        edge_feat = self.edge_func(edge_feat)
        node_feat = self.node_func(node_feat)
        graph_attr = self.attr_func(graph_attr)

        edge_feat, node_feat, graph_attr = self.conv(graph, edge_feat, node_feat, graph_attr)

        if self.dropout:
            edge_feat = self.dropout(edge_feat)  # pylint: disable=E1102
            node_feat = self.dropout(node_feat)  # pylint: disable=E1102
            graph_attr = self.dropout(graph_attr)  # pylint: disable=E1102

        if self.skip:
            edge_feat = edge_feat + inputs[0]
            node_feat = node_feat + inputs[1]
            graph_attr = graph_attr + inputs[2]

        return edge_feat, node_feat, graph_attr
