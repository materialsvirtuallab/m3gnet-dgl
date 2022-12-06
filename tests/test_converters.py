import os
import unittest

import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.util.testing import PymatgenTest

from megnet.graph.converters import GaussianExpansion, Pmg2Graph, get_element_list

module_dir = os.path.dirname(os.path.abspath(__file__))


class GaussianExpansionTest(unittest.TestCase):
    def test_call(self):
        bond_dist = 1.0
        dist_converter = GaussianExpansion()
        expanded_dist = dist_converter(bond_dist)
        # check the shape of a vector
        self.assertTrue(np.allclose(expanded_dist.shape, [20]))
        # check the first value of expanded distance
        self.assertTrue(np.allclose(expanded_dist[0], np.exp(-0.5 * np.power(1.0 - 0.0, 2.0))))
        # check the last value of expanded distance
        self.assertTrue(np.allclose(expanded_dist[-1], np.exp(-0.5 * np.power(1.0 - 4.0, 2.0))))


class Pmg2GraphTest(PymatgenTest):
    def test_get_graph_from_molecule(self):
        coords = [
            [0.000000, 0.000000, 0.000000],
            [0.000000, 0.000000, 1.089000],
            [1.026719, 0.000000, -0.363000],
            [-0.513360, -0.889165, -0.363000],
            [-0.513360, 0.889165, -0.363000],
        ]
        methane = Molecule(["C", "H", "H", "H", "H"], coords)
        element_types = get_element_list([methane])
        p2g = Pmg2Graph(element_types=element_types, cutoff=1.5)
        graph, state = p2g.get_graph_from_molecule(methane)
        # check the number of nodes
        self.assertTrue(np.allclose(graph.num_nodes(), 5))
        # check the number of edges
        self.assertTrue(np.allclose(graph.num_edges(), 8))
        # check the src_ids
        self.assertTrue(np.allclose(graph.edges()[0].numpy(), [0, 0, 0, 0, 1, 2, 3, 4]))
        # check the dst_ids
        self.assertTrue(np.allclose(graph.edges()[1].numpy(), [1, 2, 3, 4, 0, 0, 0, 0]))
        # check the atomic features of atom C
        self.assertTrue(np.allclose(graph.ndata["attr"][0], [0, 1]))
        # check the atomic features of atom H
        self.assertTrue(np.allclose(graph.ndata["attr"][1], [1, 0]))
        # check the edge features of atom 0 and atom 1
        dist_converter = GaussianExpansion()
        self.assertTrue(np.allclose(graph.edata["edge_attr"][0].numpy(), dist_converter(1.089).numpy()))
        # check the shape of state features
        self.assertTrue(np.allclose(len(state), 2))
        # check the value of state features
        self.assertTrue(np.allclose(state, [3.208492, 0.8]))

    def test_get_graph_from_structure(self):
        structure_LiFePO4 = self.get_structure("LiFePO4")
        element_types = get_element_list([structure_LiFePO4])
        p2g = Pmg2Graph(element_types=element_types, cutoff=4.0)
        graph, state = p2g.get_graph_from_structure(structure_LiFePO4)
        # check the number of nodes
        self.assertTrue(np.allclose(graph.num_nodes(), structure_LiFePO4.num_sites))
        # check the atomic feature of atom 0
        self.assertTrue(np.allclose(graph.ndata["attr"][0].numpy(), [1, 0, 0, 0]))
        # check the atomic feature of atom 4
        self.assertTrue(np.allclose(graph.ndata["attr"][4].numpy(), [0, 0, 0, 1]))
        # check the number of bonds
        self.assertTrue(np.allclose(graph.num_edges(), 704))
        # check the edge features of bond between atom 0 and 6
        self.assertTrue(
            np.allclose(
                graph.edata["edge_attr"][0].numpy(),
                [
                    0.00403916509822011,
                    0.007947498932480812,
                    0.014959634281694889,
                    0.026937847957015038,
                    0.046404093503952026,
                    0.07647180557250977,
                    0.1205584779381752,
                    0.18182168900966644,
                    0.26232829689979553,
                    0.3620729148387909,
                    0.4780776798725128,
                    0.6038823127746582,
                    0.7297223806381226,
                    0.8435572981834412,
                    0.9328739047050476,
                    0.9869219660758972,
                    0.9988359212875366,
                    0.9670679569244385,
                    0.895717978477478,
                    0.793664813041687,
                ],
            )
        )
        # check the state features
        self.assertTrue(np.allclose(state, [0.0, 0.0]))
        structure_BaTiO3 = Structure.from_prototype("perovskite", ["Ba", "Ti", "O"], a=4.04)
        element_types = get_element_list([structure_BaTiO3])
        p2g = Pmg2Graph(element_types=element_types, cutoff=4.0)
        graph, state = p2g.get_graph_from_structure(structure_BaTiO3)
        # check the number of nodes
        self.assertTrue(np.allclose(graph.num_nodes(), structure_BaTiO3.num_sites))
        # check the atomic features of atom 0
        self.assertTrue(np.allclose(graph.ndata["attr"][0], [0, 0, 1]))
        # check the atomic features of atom 1
        self.assertTrue(np.allclose(graph.ndata["attr"][1], [0, 1, 0]))
        # check the number of edges
        self.assertTrue(np.allclose(graph.num_edges(), 76))
        # check the edge features of bond between atom 0 and 1
        self.assertTrue(
            np.allclose(
                graph.edata["edge_attr"][0].numpy(),
                [
                    0.002197137800976634,
                    0.004488740116357803,
                    0.008772904984652996,
                    0.016402624547481537,
                    0.029338326305150986,
                    0.05020054057240486,
                    0.08217374235391617,
                    0.12867948412895203,
                    0.1927689015865326,
                    0.27625882625579834,
                    0.378744900226593,
                    0.4967397451400757,
                    0.6232503056526184,
                    0.7480794191360474,
                    0.858982503414154,
                    0.9435663819313049,
                    0.9915441274642944,
                    0.996788740158081,
                    0.9586182832717896,
                    0.881941556930542,
                ],
            )
        )
        # check the state features
        self.assertTrue(np.allclose(state, [0.0, 0.0]))

    def test_get_element_list(self):
        cscl = Structure.from_spacegroup("Pm-3m", Lattice.cubic(3), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        naf = Structure.from_spacegroup("Pm-3m", Lattice.cubic(3), ["Na", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        elem_list = get_element_list([cscl, naf])
        self.assertEqual(elem_list, ("F", "Na", "Cl", "Cs"))


if __name__ == "__main__":
    unittest.main()
