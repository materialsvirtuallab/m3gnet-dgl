"""
Tools to convert materials representations from Pymatgen and other codes to DGLGraphs.
"""
from __future__ import annotations

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from dgl.backend import tensor
from pymatgen.core import Element, Molecule, Structure
from pymatgen.optimization.neighbors import find_points_in_spheres


def get_element_list(train_structures: list[Structure | Molecule]) -> tuple[str]:
    """Get the dictionary containing elements in the training set for atomic features

    Args:
        train_structures: pymatgen Molecule/Structure object

    Returns:
        Tuple of elements covered in training set
    """
    elements = set()
    for s in train_structures:
        elements.update(s.composition.get_el_amt_dict().keys())
    return tuple(sorted(elements, key=lambda el: Element(el).Z))  # type: ignore


class GaussianExpansion(nn.Module):
    r"""
    Gaussian Radial Expansion.
    The bond distance is expanded to a vector of shape [m],
    where m is the number of Gaussian basis centers
    """

    def __init__(
        self,
        initial: float = 0.0,
        final: float = 4.0,
        num_centers: int = 20,
        width: None | float = 0.5,
    ):
        """
        Parameters
        ----------
        initial : float
                Location of initial Gaussian basis center.
        final : float
                Location of final Gaussian basis center
        number : int
                Number of Gaussian Basis functions
        width : float
                Width of Gaussian Basis functions
        """
        super().__init__()
        self.centers = np.linspace(initial, final, num_centers)
        self.centers = nn.Parameter(torch.tensor(self.centers).float(), requires_grad=False)
        if width is None:
            self.width = float(1.0 / np.diff(self.centers).mean())
        else:
            self.width = width

    def reset_parameters(self):
        """Reinitialize model parameters."""
        device = self.centers.device
        self.centers = nn.Parameter(self.centers.clone().detach().float(), requires_grad=False).to(device)

    def forward(self, bond_dists):
        """Expand distances.

        Parameters
        ----------
        bond_dists :
            Bond (edge) distances between two atoms (nodes)

        Returns
        -------
        A vector of expanded distance with shape [num_centers]
        """
        diff = bond_dists - self.centers
        return torch.exp(-self.width * (diff**2))


class Pmg2Graph:
    """
    Construct a DGL graph from Pymatgen Molecules or Structures.
    """

    def __init__(
        self,
        element_types: tuple[str],
        cutoff: float = 5.0,
        initial: float = 0.0,
        final: float = 4.0,
        num_centers: int = 20,
        width: float = 0.5,
    ):
        """
        Parameters
        ----------
        element_types: List of elements present in dataset for graph conversion. This ensures all graphs are
            constructed with the same dimensionality of features.
        cutoff: Cutoff radius for graph representation
        initial: Initial location of center for Gaussian expansion
        final: Final location of center for Gaussian expansion
        num_centers: Number of centers for Gaussian expansion
        width: Width of Gaussian function
        """
        self.element_types = tuple(element_types)
        self.cutoff = cutoff
        self.initial = initial
        self.final = final
        self.num_centers = num_centers
        self.width = width

    def get_graph_from_molecule(self, mol: Molecule) -> tuple[dgl.DGLGraph, list]:
        """
        Get a DGL graph from an input molecule.

        :param mol: pymatgen molecule object
        :return: (dgl graph, state features)
        """
        natoms = len(mol)
        R = mol.cart_coords
        element_types = self.element_types
        Z = [np.eye(len(element_types))[element_types.index(site.specie.symbol)] for site in mol]
        Z = np.array(Z)
        weight = mol.composition.weight / len(mol)
        dist = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
        dist_converter = GaussianExpansion(
            initial=self.initial,
            final=self.final,
            num_centers=self.num_centers,
            width=self.width,
        )
        dists = mol.distance_matrix.flatten()
        nbonds = (np.count_nonzero(dists <= self.cutoff) - natoms) / 2
        nbonds /= natoms
        adj = sp.csr_matrix(dist <= self.cutoff) - sp.eye(natoms, dtype=np.bool_)
        adj = adj.tocoo()
        u, v = tensor(adj.row), tensor(adj.col)
        edge_rbf_list = []
        g = dgl.graph((u, v))
        for i in range(g.num_edges()):
            rbf_dist = dist_converter(dist[u[i]][v[i]]).detach().numpy()
            edge_rbf_list += [rbf_dist]
        edge_rbf_list = np.array(edge_rbf_list).astype(np.float64)
        g = dgl.to_bidirected(g)
        g.edata["edge_attr"] = tensor(edge_rbf_list)
        g.ndata["attr"] = tensor(Z)
        state_attr = [weight, nbonds]
        return g, state_attr

    def get_graph_from_structure(self, structure: Structure) -> tuple[dgl.DGLGraph, list]:
        """
        Get a DGL graph from an input Structure.

        :param structure: pymatgen structure object
        :return:
            g: dgl graph
            state_attr: state features
        """

        numerical_tol = 1.0e-8
        pbc = np.array([1, 1, 1], dtype=int)
        element_types = self.element_types
        Z = [np.eye(len(element_types))[element_types.index(site.specie.symbol)] for site in structure]
        Z = np.array(Z)
        lattice_matrix = np.ascontiguousarray(np.array(structure.lattice.matrix), dtype=float)
        cart_coords = np.ascontiguousarray(np.array(structure.cart_coords), dtype=float)
        src_id, dst_id, images, bond_dist = find_points_in_spheres(
            cart_coords,
            cart_coords,
            r=self.cutoff,
            pbc=pbc,
            lattice=lattice_matrix,
            tol=numerical_tol,
        )
        exclude_self = (src_id != dst_id) | (bond_dist > numerical_tol)
        src_id, dst_id, bond_dist = (
            src_id[exclude_self],
            dst_id[exclude_self],
            bond_dist[exclude_self],
        )

        dist_converter = GaussianExpansion(
            initial=self.initial,
            final=self.final,
            num_centers=self.num_centers,
            width=self.width,
        )
        edge_rbf_list = []
        u, v = tensor(src_id), tensor(dst_id)
        g = dgl.graph((u, v))
        for i in range(g.num_edges()):
            rbf_dist = dist_converter(bond_dist[i]).detach().numpy()
            edge_rbf_list += [rbf_dist]
        edge_rbf_list = np.array(edge_rbf_list).astype(np.float64)
        g.edata["edge_attr"] = tensor(edge_rbf_list)
        g.ndata["attr"] = tensor(Z)
        state_attr = [0.0, 0.0]

        return g, state_attr
