from collections import namedtuple

import dgl
import torch as th
import torch.nn as nn

from megnet.models.layers import MLP, MEGNetBlock, MEGNetGraphConv

Graph = namedtuple("Graph", "graph, graph_attr")


def build_graph(N, E, NDIM=5, EDIM=3, GDIM=10):
    graph = dgl.rand_graph(N, E)
    graph.ndata["node_feat"] = th.rand(N, NDIM)
    graph.edata["edge_feat"] = th.rand(E, EDIM)
    graph_attr = th.rand(1, GDIM)
    return Graph(graph, graph_attr)


def get_graphs(num_graphs, NDIM=5, EDIM=3, GDIM=10):
    Ns = th.randint(10, 30, (num_graphs,)).tolist()
    Es = th.randint(35, 100, (num_graphs,)).tolist()
    graphs = [build_graph(*gspec, NDIM, EDIM, GDIM) for gspec in zip(Ns, Es)]
    return graphs


def batch(graph_attrs_lists):
    graphs, attrs = list(zip(*graph_attrs_lists))
    batched_graph = dgl.batch(graphs)
    batched_attrs = th.vstack(attrs)
    return batched_graph, batched_attrs


def test_megnet_layer():
    graphs = get_graphs(5)
    batched_graph, attrs = batch(graphs)

    NDIM, EDIM, GDIM = 5, 3, 10
    edge_func = MLP(dims=[2 * NDIM + EDIM + GDIM, EDIM])
    node_func = MLP(dims=[EDIM + NDIM + GDIM, NDIM])
    attr_func = MLP(dims=[EDIM + NDIM + GDIM, GDIM])
    layer = MEGNetGraphConv(edge_func, node_func, attr_func)

    # one pass
    edge_feat = batched_graph.edata.pop("edge_feat")
    node_feat = batched_graph.ndata.pop("node_feat")
    out = layer(batched_graph, edge_feat, node_feat, attrs)
    return out


def test_megnet_block():
    DIM = 5
    N1, N2 = 64, 32
    block = MEGNetBlock(
        dims=[5, 10, 13],
        conv_hiddens=[N1, N1, N2],
        act=nn.SiLU(),
        skip=False,
    )
    graphs = get_graphs(5, NDIM=DIM, EDIM=DIM, GDIM=DIM)
    batched_graph, attrs = batch(graphs)

    # one pass
    edge_feat = batched_graph.edata.pop("edge_feat")
    node_feat = batched_graph.ndata.pop("node_feat")
    out = block(batched_graph, edge_feat, node_feat, attrs)
    return out


if __name__ == "__main__":
    test_megnet_layer()
    test_megnet_block()
