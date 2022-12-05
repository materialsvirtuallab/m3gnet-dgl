"""
Compute three-body indices
"""
from __future__ import annotations

import dgl
import numpy as np
import torch

from ..utils.math import SphericalBesselFunction, SphericalHarmonicsFunction


def compute_3body(bond_atom_indices: np.array, n_atoms: np.array):
    """
    Calculate the three body indices from pair atom indices

    Args:
        bond_atom_indices (np.ndarray): pair atom indices
        n_atoms (list): number of atoms in each structure.

    Returns:
        triple_bond_indices (np.ndarray): bond indices that form three-body
        n_triple_ij (np.ndarray): number of three-body angles for each bond
        n_triple_i (np.ndarray): number of three-body angles each atom
        n_triple_s (np.ndarray): number of three-body angles for each structure
    """
    n_atoms_total = np.sum(n_atoms)
    first_col = bond_atom_indices[:, 0].reshape(-1, 1)
    all_indices = np.arange(n_atoms_total).reshape(1, -1)
    n_bond_per_atom = np.count_nonzero(first_col == all_indices, axis=0)
    n_triple_i = n_bond_per_atom * (n_bond_per_atom - 1)
    n_triple = np.sum(n_triple_i)
    n_triple_ij = np.repeat(n_bond_per_atom - 1, n_bond_per_atom)
    triple_bond_indices = np.empty(shape=(n_triple, 2), dtype=np.int32)

    start = 0
    cs = 0
    for i, n in enumerate(n_bond_per_atom):
        if n > 0:
            """
            triple_bond_indices is generated from all pair permutations of atom indices. The
            numpy version below does this with much greater efficiency. The equivalent slow
            code is:

            ```
            for j, k in itertools.permutations(range(n), 2):
                triple_bond_indices[index] = [start + j, start + k]
            ```
            """
            r = np.arange(n)
            x, y = np.meshgrid(r, r)
            c = np.stack([y.ravel(), x.ravel()], axis=1)
            final = c[c[:, 0] != c[:, 1]]
            triple_bond_indices[start : start + (n * (n - 1)), :] = final + cs
            start += n * (n - 1)
            cs += n

    n_triple_s = []
    i = 0
    for n in n_atoms:
        j = i + n
        n_triple_s.append(np.sum(n_triple_i[i:j]))
        i = j

    return (
        triple_bond_indices,
        n_triple_ij,
        n_triple_i,
        np.array(n_triple_s, dtype=np.int32),
    )


# def get_pair_vector_from_graph(graph: list):
#     """
#     Given a graph list return pair vectors that form the bonds
#     Args:
#         graph (List): graph list, obtained by MaterialGraph.as_list()
#
#     Returns: pair vector tf.Tensor
#
#     """
#     atom_positions = graph[Index.ATOM_POSITIONS]
#     lattices = graph[Index.LATTICES]
#     pbc_offsets = graph[Index.PBC_OFFSETS]
#     bond_atom_indices = graph[Index.BOND_ATOM_INDICES]
#     n_bonds = graph[Index.N_BONDS]
#     if lattices is not None:
#         lattices = tf.gather(lattices, get_segment_indices_from_n(n_bonds))
#         offset_vec = tf.keras.backend.batch_dot(tf.cast(pbc_offsets, DataType.tf_float), lattices)
#     else:
#         offset_vec = tf.constant([[0.0, 0.0, 0.0]], dtype=DataType.tf_float)
#     diff = (
#         tf.gather(atom_positions, bond_atom_indices[:, 1])
#         + offset_vec
#         - tf.gather(atom_positions, bond_atom_indices[:, 0])
#     )
#     return tf.cast(diff, DataType.tf_float)
#
#
# def tf_compute_distance_angle(graph: list):
#     """
#     Given a graph with pair, triplet indices, calculate the pair distance,
#     triplet angles, etc.
#     Args:
#         graph: MaterialGraph in List format
#
#     Returns: MaterialGraph in List format
#     """
#     graph = graph[:]
#     pair_vectors = get_pair_vector_from_graph(graph)
#     pair_rij = get_length(pair_vectors)
#     vij = tf.gather(pair_vectors, graph[Index.TRIPLE_BOND_INDICES][:, 0])
#     vik = tf.gather(pair_vectors, graph[Index.TRIPLE_BOND_INDICES][:, 1])
#     rij = tf.gather(pair_rij, graph[Index.TRIPLE_BOND_INDICES][:, 0])
#     rik = tf.gather(pair_rij, graph[Index.TRIPLE_BOND_INDICES][:, 1])
#     cos_jik = tf.reduce_sum(vij * vik, axis=1) / (rij * rik)
#     cos_jik = tf.clip_by_value(cos_jik, -1, 1)
#     graph[Index.BOND_WEIGHTS] = pair_rij
#     graph[Index.BONDS] = pair_rij[:, None]
#     graph[Index.TRIPLE_BOND_LENGTHS] = rik
#     graph[Index.THETA] = cos_jik  # theta is in fact costheta
#     graph[Index.PHI] = tf.zeros_like(cos_jik)  # dummy phi here
#     return graph
#
#
# def include_threebody_indices(graph: MaterialGraph | list, threebody_cutoff: float | None = None):
#     """
#     Given a graph without threebody indices, add the threebody indices
#     according to a threebody cutoff radius
#     Args:
#         graph: MaterialGraph
#         threebody_cutoff: float, threebody cutoff radius
#
#     Returns: new graph with added threebody indices
#
#     """
#     if isinstance(graph, MaterialGraph):
#         is_graph = True
#         graph_list: list = graph.as_list()
#     else:
#         is_graph = False
#         graph_list = graph
#
#     return _list_include_threebody_indices(graph_list, threebody_cutoff=threebody_cutoff, is_graph=is_graph)
#
#
# def _list_include_threebody_indices(graph: list, threebody_cutoff: float | None = None, is_graph: bool = False):
#     graph = graph[:]
#     bond_atom_indices = graph[Index.BOND_ATOM_INDICES]
#     n_bond = bond_atom_indices.shape[0]
#     if n_bond > 0 and threebody_cutoff is not None:
#         valid_three_body = graph[Index.BOND_WEIGHTS] <= threebody_cutoff
#         ij_reverse_map = np.where(valid_three_body)[0]
#         original_index = np.arange(n_bond)[valid_three_body]
#         bond_atom_indices = bond_atom_indices[valid_three_body, :]
#     else:
#         ij_reverse_map = None
#         original_index = np.arange(n_bond)
#     if bond_atom_indices.shape[0] > 0:
#
#         bond_indices, n_triple_ij, n_triple_i, n_triple_s = _compute_3body(
#             bond_atom_indices,
#             graph[Index.N_ATOMS],
#         )
#
#         if ij_reverse_map is not None:
#             n_triple_ij_ = np.zeros(shape=(n_bond,), dtype="int32")
#             n_triple_ij_[ij_reverse_map] = n_triple_ij
#             n_triple_ij = n_triple_ij_
#         bond_indices = original_index[bond_indices]
#         bond_indices = np.array(bond_indices, dtype="int32")
#     else:
#         bond_indices = np.reshape(np.array([], dtype="int32"), [-1, 2])
#         if n_bond == 0:
#             n_triple_ij = np.array([], dtype="int32")
#         else:
#             n_triple_ij = np.array([0] * n_bond, dtype="int32")
#         n_triple_i = np.array([0] * len(graph[Index.ATOMS]), dtype="int32")
#         n_triple_s = np.array([0], dtype="int32")
#     graph[Index.TRIPLE_BOND_INDICES] = bond_indices
#     graph[Index.N_TRIPLE_IJ] = n_triple_ij
#     graph[Index.N_TRIPLE_I] = n_triple_i
#     graph[Index.N_TRIPLE_S] = n_triple_s
#     if is_graph:
#         graph = MaterialGraph.from_list(graph)
#     return graph


class SphericalBesselWithHarmonics(dgl.Module):
    """
    Spherical bessel function as radial function and spherical harmonics as
    the angular function
    """

    def __init__(
        self, max_n: int, max_l: int, cutoff: float = 5.0, use_phi: bool = False, smooth: bool = False, **kwargs
    ):
        """

        Args:
            max_n (int): maximum number for radial expansion
            max_l (int): maximum number of angular expansion
            cutoff (float): cutoff radius
            use_phi (bool): whether to use phi. So far, we do not have the phi
                angle yet.
            smooth (bool): whether to use the smooth version of the radial
                function
            **kwargs:
        """
        super().__init__(**kwargs)
        self.sbf = SphericalBesselFunction(max_l=max_l, max_n=max_n, cutoff=cutoff, smooth=smooth)
        self.shf = SphericalHarmonicsFunction(max_l=max_l, use_phi=use_phi)
        self.max_n = max_n
        self.max_l = max_l
        self.cutoff = cutoff
        self.use_phi = use_phi

    def forward(self, graph: dgl.DGLGraph, **kwargs) -> torch.Tensor:  # noqa
        """
        Args:
            graph (list): the list representation of a graph
            **kwargs:
        Returns: combined radial and spherical harmonic expansion of the
            distance and angle
        """
        # sbf = self.sbf(graph[Index.TRIPLE_BOND_LENGTHS])
        # shf = self.shf(graph[Index.THETA], graph[Index.PHI])
        # return combine_sbf_shf(sbf, shf, max_n=self.max_n, max_l=self.max_l, use_phi=self.use_phi)


class ThreeBodyInteraction(dgl.Module):
    """
    Include 3-body interactions in the bond update
    """

    def __init__(self, update_network_atom: dgl.Module, update_network_bond: dgl.Module, **kwargs):
        r"""
        Args:
            update_network_atom (dgl.Module): Module for updating the atom attributes before merging with 3-body
                interactions.
            update_network_bond (dgl.Module): Module for updating the bond information after merging with 3-body
                interactions
            **kwargs: Passthrough to dgl.Module superclass.
        """
        super().__init__(**kwargs)
        self.update_network_atom = update_network_atom
        self.update_network_bond = update_network_bond

    def forward(self, graph: dgl.DGLGraph, three_basis: torch.Tensor, three_cutoff: float, **kwargs) -> dgl.DGLGraph:
        """

        Args:
            graph (dgl.DGLGraph): Input graph
            three_basis (tf.Tensor): three body basis expansion
            three_cutoff (float): cutoff radius
            **kwargs:
        Returns: Updated dgl.DGLGraph
        """
        # graph = graph[:]
        # end_atom_index = tf.gather(graph[Index.BOND_ATOM_INDICES][:, 1], graph[Index.TRIPLE_BOND_INDICES][:, 1])
        # atoms = self.update_network(graph[Index.ATOMS])
        # atoms = tf.gather(atoms, end_atom_index)
        # basis = three_basis * atoms
        # n_bonds = tf.reduce_sum(graph[Index.N_BONDS])
        # weights = tf.reshape(tf.gather(three_cutoff, graph[Index.TRIPLE_BOND_INDICES]), (-1, 2))
        # weights = tf.math.reduce_prod(weights, axis=-1)
        # basis = basis * weights[:, None]
        # new_bonds = tf.math.unsorted_segment_sum(
        #     basis,
        #     get_segment_indices_from_n(graph[Index.N_TRIPLE_IJ]),
        #     num_segments=n_bonds,
        # )
        # graph[Index.BONDS] = graph[Index.BONDS] + self.update_network2(new_bonds)
        return graph
