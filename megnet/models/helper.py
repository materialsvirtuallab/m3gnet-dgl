import torch
from torch.nn import Module, Linear, ModuleList, LSTM
from dgl import broadcast_edges, softmax_edges, sum_edges, mean_edges

from ..types import List, Callable, Tensor


class MLP(Module):
    def __init__(
        self,
        dims: List[int],
        activation: Callable[[Tensor], Tensor] = None,
        activate_last: bool = False,
        bias_last: bool = True,
    ) -> None:
        super().__init__()
        self._depth = len(dims) - 1
        self.layers = ModuleList()

        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i < self._depth - 1:
                self.layers.append(Linear(in_dim, out_dim, bias=True))

                if activation is not None:
                    self.layers.append(activation)
            else:
                self.layers.append(Linear(in_dim, out_dim, bias=bias_last))

                if activation is not None and activate_last:
                    self.layers.append(activation)

    def __repr__(self):
        dims = []

        for layer in self.layers:
            if isinstance(layer, Linear):
                dims.append(f"{layer.in_features} \u2192 {layer.out_features}")
            else:
                dims.append(layer.__class__.__name__)

        return f'MLP({", ".join(dims)})'

    @property
    def last_linear(self) -> Linear:
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                return layer

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def in_features(self) -> int:
        return self.layers[0].in_features

    @property
    def out_features(self) -> int:
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                return layer.out_features
        raise RuntimeError

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs

        for layer in self.layers:
            x = layer(x)

        return x


class EdgeSet2Set(Module):
    def __init__(self, input_dim, n_iters, n_layers):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = LSTM(self.output_dim, self.input_dim, n_layers)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        self.lstm.reset_parameters()

    def forward(self, graph, feat):
        with graph.local_scope():
            batch_size = graph.batch_size

            h = (
                feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
                feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
            )

            q_star = feat.new_zeros(batch_size, self.output_dim)

            for _ in range(self.n_iters):
                q, h = self.lstm(q_star.unsqueeze(0), h)
                q = q.view(batch_size, self.input_dim)
                e = (feat * broadcast_edges(graph, q)).sum(dim=-1, keepdim=True)
                graph.edata["e"] = e
                alpha = softmax_edges(graph, "e")
                graph.edata["r"] = feat * alpha
                readout = sum_edges(graph, "r")
                q_star = torch.cat([q, readout], dim=-1)

            return q_star


class EdgeAvgPooling(Module):

    def forward(self, graph, feat):
        with graph.local_scope():
            graph.edata['h'] = feat
            readout = mean_edges(graph, 'h')
            return readout