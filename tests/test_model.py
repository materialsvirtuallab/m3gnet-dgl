from collections import namedtuple

import dgl
import torch as th

from megnet.models import MEGNet

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


def test_megnet():
    DIM = 16
    N1, N2, N3 = 64, 32, 16
    graphs = get_graphs(5, NDIM=DIM, EDIM=DIM, GDIM=DIM)
    batched_graph, attrs = batch(graphs)

    model = MEGNet(
        in_dim=DIM,
        num_blocks=3,
        hiddens=[N1, N2],
        conv_hiddens=[N1, N1, N2],
        act="swish",
        s2s_num_layers=4,
        s2s_num_iters=3,
        output_hiddens=[N2, N3],
        is_classification=True,
    )

    # one pass
    edge_feat = batched_graph.edata.pop("edge_feat")
    node_feat = batched_graph.ndata.pop("node_feat")
    out = model(batched_graph, edge_feat, node_feat, attrs)
    return out


if __name__ == "__main__":
    test_megnet()
