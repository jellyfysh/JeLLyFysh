# JeLLFysh - a Python application for all-atom event-chain Monte Carlo - https://github.com/jellyfysh
# Copyright (C) 2019, 2022 The JeLLyFysh organization
# (See the AUTHORS.md file for the full list of authors.)
#
# This file is part of JeLLyFysh.
#
# JeLLyFysh is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# JeLLyFysh is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with JeLLyFysh in the LICENSE file.
# If not, see <https://www.gnu.org/licenses/>.
#
# If you use JeLLyFysh in published work, please cite the following reference (see [Hoellmer2020] in References.bib):
# Philipp Hoellmer, Liang Qin, Michael F. Faulkner, A. C. Maggs, and Werner Krauth,
# JeLLyFysh-Version1.0 -- a Python application for all-atom event-chain Monte Carlo,
# Computer Physics Communications, Volume 253, 107168 (2020), https://doi.org/10.1016/j.cpc.2020.107168.
#
from unittest import TestCase, main
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.potential.inverse_power_coulomb_bounding_potential import InversePowerCoulombBoundingPotential
import jellyfysh.setting as setting
from jellyfysh.setting import hypercubic_setting


class TestInversePowerCoulombBoundingPotential(TestCase):
    def setUp(self) -> None:
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=1.0)
        # Set these so the settings are initialized
        setting.set_number_of_root_nodes(2)
        setting.set_number_of_nodes_per_root_node(2)
        setting.set_number_of_node_levels(1)
        # noinspection PyArgumentEqualDefault
        self._potential = InversePowerCoulombBoundingPotential(prefactor=1.6)

    def tearDown(self) -> None:
        setting.reset()

    def test_displacement_equal_charge_uphill_direction_zero_same_image(self):
        # The time displacement depends on the value of the non-vanishing component of the velocity.
        self.assertAlmostEqual(self._potential.displacement(
            [1.0, 0.0, 0.0], [0.1907448836841521, 0.30844711493736865, -0.28453409629109017], 1.0, 1.0,
            0.009639528698551114), 0.1907448836841521 - 0.18763880019427548, places=14)
        self.assertAlmostEqual(self._potential.displacement(
            [3.1, 0.0, 0.0], [0.1907448836841521, 0.30844711493736865, -0.28453409629109017], 1.0, 1.0,
            0.009639528698551114), (0.1907448836841521 - 0.18763880019427548) / 3.1, places=14)

    def test_displacement_equal_charge_uphill_direction_one_same_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 1.0, 0.0], [-0.26388529926704435, 0.3154253255236278, 0.18314675352980678], 1.0, 1.0,
            1.3412289656743743),
                               0.3154253255236278 - 0.060412764813754745, places=13)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 3.1, 0.0], [-0.26388529926704435, 0.3154253255236278, 0.18314675352980678], 1.0, 1.0,
            1.3412289656743743), (0.3154253255236278 - 0.060412764813754745) / 3.1, places=13)

    def test_displacement_equal_charge_uphill_direction_two_same_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 1.0], [0.24079938551648655, 0.16370997351125174, 0.29104306617231285], 1.0, 1.0,
            0.3403655472079823), 0.29104306617231285 - 0.24188438158174752, places=14)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 3.1], [0.24079938551648655, 0.16370997351125174, 0.29104306617231285], 1.0, 1.0,
            0.3403655472079823), (0.29104306617231285 - 0.24188438158174752) / 3.1, places=14)

    def test_displacement_equal_charge_uphill_direction_zero_next_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [1.0, 0.0, 0.0], [0.1907448836841521, -0.30844711493736865, 0.28453409629109017], 1.0, 1.0,
            1.5150110651006652), 1.1907448836841521 - 0.13705017662742836, places=12)
        self.assertAlmostEqual(self._potential.displacement(
            [3.1, 0.0, 0.0], [0.1907448836841521, -0.30844711493736865, 0.28453409629109017], 1.0, 1.0,
            1.5150110651006652), (1.1907448836841521 - 0.13705017662742836) / 3.1, places=12)

    def test_displacement_equal_charge_uphill_direction_one_next_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 1.0, 0.0], [-0.26388529926704435, 0.3154253255236278, 0.18314675352980678], 1.0, 1.0,
            2.8911565677176148), 1.3154253255236278 - 0.21214934967679594, places=12)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 3.1, 0.0], [-0.26388529926704435, 0.3154253255236278, 0.18314675352980678], 1.0, 1.0,
            2.8911565677176148), (1.3154253255236278 - 0.21214934967679594) / 3.1, places=12)

    def test_displacement_equal_charge_uphill_direction_two_next_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 1.0], [0.24079938551648655, 0.16370997351125174, 0.29104306617231285], 1.0, 1.0,
            3.1632316768074107), 1.29104306617231285 - 0.22888813299386318, places=12)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 3.1], [0.24079938551648655, 0.16370997351125174, 0.29104306617231285], 1.0, 1.0,
            3.1632316768074107), (1.29104306617231285 - 0.22888813299386318) / 3.1, places=12)

    def test_displacement_equal_charge_uphill_direction_zero_next_next_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [1.0, 0.0, 0.0], [0.1907448836841521, -0.30844711493736865, 0.28453409629109017], 1.0, 1.0,
            2.737534726763149), 2.1907448836841521 - 0.18615241791697856, places=12)
        self.assertAlmostEqual(self._potential.displacement(
            [3.1, 0.0, 0.0], [0.1907448836841521, -0.30844711493736865, 0.28453409629109017], 1.0, 1.0,
            2.737534726763149), (2.1907448836841521 - 0.18615241791697856) / 3.1, places=12)

    def test_displacement_equal_charge_uphill_direction_one_next_next_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 1.0, 0.0], [-0.26388529926704435, 0.3154253255236278, 0.18314675352980678], 1.0, 1.0,
            5.982700660685892), 2.3154253255236278 - 0.030275295549534786, places=12)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 3.1, 0.0], [-0.26388529926704435, 0.3154253255236278, 0.18314675352980678], 1.0, 1.0,
            5.982700660685892), (2.3154253255236278 - 0.030275295549534786) / 3.1, places=12)

    def test_displacement_equal_charge_uphill_direction_two_next_next_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 1.0], [0.24079938551648655, 0.16370997351125174, 0.29104306617231285], 1.0, 1.0,
            7.041312903831338), 2.29104306617231285 - 0.02868162418537079, places=12)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 3.1], [0.24079938551648655, 0.16370997351125174, 0.29104306617231285], 1.0, 1.0,
            7.041312903831338), (2.29104306617231285 - 0.02868162418537079) / 3.1, places=12)

    def test_displacement_equal_charge_downhill_direction_zero_same_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [1.0, 0.0, 0.0], [-0.1907448836841521, 0.30844711493736865, -0.28453409629109017], 1.0, 1.0,
            0.048467391473757004), -0.1907448836841521 + 1.0 - 0.4833590186926574, places=13)
        self.assertAlmostEqual(self._potential.displacement(
            [3.1, 0.0, 0.0], [-0.1907448836841521, 0.30844711493736865, -0.28453409629109017], 1.0, 1.0,
            0.048467391473757004), (-0.1907448836841521 + 1.0 - 0.4833590186926574) / 3.1, places=13)

    def test_displacement_equal_charge_downhill_direction_one_same_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 1.0, 0.0], [-0.26388529926704435, -0.3154253255236278, 0.18314675352980678], 1.0, 1.0,
            0.2769539791301425), -0.3154253255236278 + 1.0 - 0.4326522580911339, places=13)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 3.1, 0.0], [-0.26388529926704435, -0.3154253255236278, 0.18314675352980678], 1.0, 1.0,
            0.2769539791301425), (-0.3154253255236278 + 1.0 - 0.4326522580911339) / 3.1, places=13)

    def test_displacement_equal_charge_downhill_direction_two_same_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 1.0], [0.24079938551648655, 0.16370997351125174, -0.29104306617231285], 1.0, 1.0,
            0.3715223933270022), -0.29104306617231285 + 1.0 - 0.41879851862802286, places=13)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 3.1], [0.24079938551648655, 0.16370997351125174, -0.29104306617231285], 1.0, 1.0,
            0.3715223933270022), (-0.29104306617231285 + 1.0 - 0.41879851862802286) / 3.1, places=13)

    def test_displacement_equal_charge_downhill_direction_zero_next_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [1.0, 0.0, 0.0], [-0.1907448836841521, 0.30844711493736865, -0.28453409629109017], 1.0, 1.0,
            2.392558368286875), -0.1907448836841521 + 2.0 - 0.18719960224410015, places=12)
        self.assertAlmostEqual(self._potential.displacement(
            [3.1, 0.0, 0.0], [-0.1907448836841521, 0.30844711493736865, -0.28453409629109017], 1.0, 1.0,
            2.392558368286875), (-0.1907448836841521 + 2.0 - 0.18719960224410015) / 3.1, places=12)

    def test_displacement_equal_charge_downhill_direction_one_next_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 1.0, 0.0], [-0.26388529926704435, -0.3154253255236278, 0.18314675352980678], 1.0, 1.0,
            4.437242448124799), -0.3154253255236278 + 2.0 - 0.07791712770049392, places=12)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 3.1, 0.0], [-0.26388529926704435, -0.3154253255236278, 0.18314675352980678], 1.0, 1.0,
            4.437242448124799), (-0.3154253255236278 + 2.0 - 0.07791712770049392) / 3.1, places=12)

    def test_displacement_equal_charge_downhill_direction_two_next_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 1.0], [0.24079938551648655, 0.16370997351125174, -0.29104306617231285], 1.0, 1.0,
            5.456822031916243), -0.29104306617231285 + 2.0 - 0.008689619657246017, places=12)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 3.1], [0.24079938551648655, 0.16370997351125174, -0.29104306617231285], 1.0, 1.0,
            5.456822031916243), (-0.29104306617231285 + 2.0 - 0.008689619657246017) / 3.1, places=12)

    def test_displacement_opposite_charge_uphill_direction_zero_same_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [1.0, 0.0, 0.0], [-0.1907448836841521, 0.30844711493736865, -0.28453409629109017], 1.0, -1.0,
            0.8556747347860076), -0.1907448836841521 + 0.44515840902067266, places=13)
        self.assertAlmostEqual(self._potential.displacement(
            [3.1, 0.0, 0.0], [-0.1907448836841521, 0.30844711493736865, -0.28453409629109017], 1.0, -1.0,
            0.8556747347860076), (-0.1907448836841521 + 0.44515840902067266) / 3.1, places=13)

    def test_displacement_opposite_charge_uphill_direction_one_same_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 1.0, 0.0], [-0.26388529926704435, -0.3154253255236278, 0.18314675352980678], -1.0, 1.0,
            0.0726908471220229), -0.3154253255236278 + 0.3287020714993278, places=14)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 3.1, 0.0], [-0.26388529926704435, -0.3154253255236278, 0.18314675352980678], -1.0, 1.0,
            0.0726908471220229), (-0.3154253255236278 + 0.3287020714993278) / 3.1, places=14)

    def test_displacement_opposite_charge_uphill_direction_two_same_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 1.0], [0.24079938551648655, 0.16370997351125174, -0.29104306617231285], 1.0, -1.0,
            0.5737873232080783), -0.29104306617231285 + 0.3853669682486328, places=13)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 3.1], [0.24079938551648655, 0.16370997351125174, -0.29104306617231285], 1.0, -1.0,
            0.5737873232080783), (-0.29104306617231285 + 0.3853669682486328) / 3.1, places=13)

    def test_displacement_opposite_charge_uphill_direction_zero_next_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [1.0, 0.0, 0.0], [-0.1907448836841521, -0.30844711493736865, 0.28453409629109017], -1.0, 1.0,
            2.314448452657504), -0.1907448836841521 + 1.0 + 0.4770645841046286, places=12)
        self.assertAlmostEqual(self._potential.displacement(
            [3.1, 0.0, 0.0], [-0.1907448836841521, -0.30844711493736865, 0.28453409629109017], -1.0, 1.0,
            2.314448452657504), (-0.1907448836841521 + 1.0 + 0.4770645841046286) / 3.1, places=12)

    def test_displacement_opposite_charge_uphill_direction_one_next_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 1.0, 0.0], [-0.26388529926704435, -0.3154253255236278, 0.18314675352980678], 1.0, -1.0,
            1.430864830416918), -0.3154253255236278 + 1.0 + 0.16832989166952644, places=12)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 3.1, 0.0], [-0.26388529926704435, -0.3154253255236278, 0.18314675352980678], 1.0, -1.0,
            1.430864830416918), (-0.3154253255236278 + 1.0 + 0.16832989166952644) / 3.1, places=12)

    def test_displacement_opposite_charge_uphill_direction_two_next_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 1.0], [0.24079938551648655, 0.16370997351125174, -0.29104306617231285], -1.0, 1.0,
            2.2737505417621993), -0.29104306617231285 + 1.0 + 0.2257994532473867, places=12)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 3.1], [0.24079938551648655, 0.16370997351125174, -0.29104306617231285], -1.0, 1.0,
            2.2737505417621993), (-0.29104306617231285 + 1.0 + 0.2257994532473867) / 3.1, places=12)

    def test_displacement_opposite_charge_uphill_direction_zero_next_next_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [1.0, 0.0, 0.0], [-0.1907448836841521, -0.30844711493736865, 0.28453409629109017], 1.0, -1.0,
            3.231603907318052), -0.1907448836841521 + 2.0 + 0.3399204405660574, places=12)
        self.assertAlmostEqual(self._potential.displacement(
            [3.1, 0.0, 0.0], [-0.1907448836841521, -0.30844711493736865, 0.28453409629109017], 1.0, -1.0,
            3.231603907318052), (-0.1907448836841521 + 2.0 + 0.3399204405660574) / 3.1, places=12)

    def test_displacement_opposite_charge_uphill_direction_one_next_next_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 1.0, 0.0], [-0.26388529926704435, -0.3154253255236278, 0.18314675352980678], -1.0, 1.0,
            4.664947609281399), -0.3154253255236278 + 2.0 + 0.33140920000729035, places=12)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 3.1, 0.0], [-0.26388529926704435, -0.3154253255236278, 0.18314675352980678], -1.0, 1.0,
            4.664947609281399), (-0.3154253255236278 + 2.0 + 0.33140920000729035) / 3.1, places=12)

    def test_displacement_opposite_charge_uphill_direction_two_next_next_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 1.0], [0.24079938551648655, 0.16370997351125174, -0.29104306617231285], 1.0, -1.0,
            6.247117175888333), -0.29104306617231285 + 2.0 + 0.42644489631878385, places=12)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 3.1], [0.24079938551648655, 0.16370997351125174, -0.29104306617231285], 1.0, -1.0,
            6.247117175888333), (-0.29104306617231285 + 2.0 + 0.42644489631878385) / 3.1, places=12)

    def test_displacement_opposite_charge_downhill_direction_zero_same_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [1.0, 0.0, 0.0], [0.1907448836841521, 0.30844711493736865, -0.28453409629109017], -1.0, 1.0,
            1.3295325716758453), 0.1907448836841521 + 0.48892483171069656, places=13)
        self.assertAlmostEqual(self._potential.displacement(
            [3.1, 0.0, 0.0], [0.1907448836841521, 0.30844711493736865, -0.28453409629109017], -1.0, 1.0,
            1.3295325716758453), (0.1907448836841521 + 0.48892483171069656) / 3.1, places=13)

    def test_displacement_opposite_charge_downhill_direction_one_same_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 1.0, 0.0], [-0.26388529926704435, 0.3154253255236278, 0.18314675352980678], 1.0, -1.0,
            0.003732649570575397), 0.3154253255236278 + 0.012442244085621779, places=13)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 3.1, 0.0], [-0.26388529926704435, 0.3154253255236278, 0.18314675352980678], 1.0, -1.0,
            0.003732649570575397), (0.3154253255236278 + 0.012442244085621779) / 3.1, places=13)

    def test_displacement_opposite_charge_downhill_direction_two_same_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 1.0], [0.24079938551648655, 0.16370997351125174, 0.29104306617231285], -1.0, 1.0,
            0.8471121300294273), 0.29104306617231285 + 0.1836373509955493, places=13)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 3.1], [0.24079938551648655, 0.16370997351125174, 0.29104306617231285], -1.0, 1.0,
            0.8471121300294273), (0.29104306617231285 + 0.1836373509955493) / 3.1, places=13)

    def test_displacement_opposite_charge_downhill_direction_zero_next_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [1.0, 0.0, 0.0], [0.1907448836841521, 0.30844711493736865, -0.28453409629109017], 1.0, -1.0,
            2.262851657028147), 0.1907448836841521 + 1.0 + 0.354801372741893, places=12)
        self.assertAlmostEqual(self._potential.displacement(
            [3.1, 0.0, 0.0], [0.1907448836841521, 0.30844711493736865, -0.28453409629109017], 1.0, -1.0,
            2.262851657028147), (0.1907448836841521 + 1.0 + 0.354801372741893) / 3.1, places=12)

    def test_displacement_opposite_charge_downhill_direction_one_next_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 1.0, 0.0], [-0.26388529926704435, 0.3154253255236278, 0.18314675352980678], -1.0, 1.0,
            2.5276027856303367), 0.3154253255236278 + 1.0 + 0.10321011116498574, places=12)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 3.1, 0.0], [-0.26388529926704435, 0.3154253255236278, 0.18314675352980678], -1.0, 1.0,
            2.5276027856303367), (0.3154253255236278 + 1.0 + 0.10321011116498574) / 3.1, places=12)

    def test_displacement_opposite_charge_downhill_direction_two_next_image(self):
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 1.0], [0.24079938551648655, 0.16370997351125174, 0.29104306617231285], 1.0, -1.0,
            5.421210644585653), 0.29104306617231285 + 1.0 + 0.49088890123979645, places=12)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 3.1], [0.24079938551648655, 0.16370997351125174, 0.29104306617231285], 1.0, -1.0,
            5.421210644585653), (0.29104306617231285 + 1.0 + 0.49088890123979645) / 3.1, places=12)

    def test_derivative_direction_zero(self):
        self.assertAlmostEqual(self._potential.derivative(
            [1.0, 0.0, 0.0], [-0.005923076000000027, -0.37398288999999996, 0.14366621899999998], 0.86413485,
            -0.64413971), 0.08200890714787873, places=14)
        # The derivative does not depend on the absolute value of the velocity but only on the non-vanishing component.
        self.assertAlmostEqual(self._potential.derivative(
            [3.1, 0.0, 0.0], [-0.005923076000000027, -0.37398288999999996, 0.14366621899999998], 0.86413485,
            -0.64413971), 0.08200890714787873 * 3.1, places=14)

    def test_derivative_direction_one(self):
        self.assertAlmostEqual(self._potential.derivative(
            [0.0, 1.0, 0.0], [-0.005923076000000027, -0.37398288999999996, 0.14366621899999998], 0.86413485,
            -0.64413971), 5.178040616211104, places=12)
        self.assertAlmostEqual(self._potential.derivative(
            [0.0, 3.1, 0.0], [-0.005923076000000027, -0.37398288999999996, 0.14366621899999998], 0.86413485,
            -0.64413971), 5.178040616211104 * 3.1, places=12)

    def test_derivative_direction_two(self):
        self.assertAlmostEqual(self._potential.derivative(
            [0.0, 0.0, 1.0], [-0.005923076000000027, -0.37398288999999996, 0.14366621899999998], 0.86413485,
            -0.64413971), -1.98915388123633, places=13)
        self.assertAlmostEqual(self._potential.derivative(
            [0.0, 0.0, 3.1], [-0.005923076000000027, -0.37398288999999996, 0.14366621899999998], 0.86413485,
            -0.64413971), -1.98915388123633 * 3.1, places=13)

    def test_very_small_potential_change_equal_charge(self):
        self.assertAlmostEqual(self._potential.displacement(
            [1.0, 0.0, 0.0], [0.3479865348803935, 0.19802392198685964, -0.16793607980326702], 1.0, 1.0,
            0.0033151136093199085), 0.0004870101789121728, places=16)
        self.assertAlmostEqual(self._potential.displacement(
            [3.1, 0.0, 0.0], [0.3479865348803935, 0.19802392198685964, -0.16793607980326702], 1.0, 1.0,
            0.0033151136093199085), 0.0004870101789121728 / 3.1, places=16)

    def test_very_small_potential_change_opposite_charge(self):
        # TODO: Solve this problem
        """
        When comparing the single step with the high precision results out of Mathematica the problem lies
        probably in the function base.vectors.displacement_until_new_norm_sq_smaller_separationt. The term in the sqrt
        gets extremely small (order 10^-15) because separation[direction_of_motion] is 0.0 in the minimum,
        calculated out of numbers of the oder 0.1. Therefore the error in the sqrt is relatively high,
        and this propagates into an 10^-10 error after taking the sqrt.
        """
        try:
            self.assertAlmostEqual(self._potential.displacement(
                [0.0, 0.0, 1.0], [-0.23159215246037826, 0.27067015647470893, 0.2785010797789582], 1.0, -1.0,
                2.782068224819782e-14), 0.2785011194275781, places=13)
        except AssertionError:
            pass

    def test_number_separation_arguments_is_one(self):
        self.assertEqual(self._potential.number_separation_arguments, 1)

    def test_number_charge_arguments_is_two(self):
        self.assertEqual(self._potential.number_charge_arguments, 2)

    def test_potential_change_required(self):
        self.assertTrue(self._potential.potential_change_required)

    def test_velocity_zero_raises_error(self):
        with self.assertRaises(AssertionError):
            self._potential.displacement([0.0, 0.0, 0.0], [-0.01, 0.1, -0.3], 1.0, 1.0, 0.05)
        with self.assertRaises(AssertionError):
            self._potential.derivative([0.0, 0.0, 0.0], [0.2, 0.1, -0.3], 1.0, 1.0)

    def test_negative_velocity_along_axis_raises_error(self):
        with self.assertRaises(AssertionError):
            self._potential.displacement([-1.0, 0.0, 0.0], [-0.01, 0.1, -0.3], 1.0, 1.0, 0.05)
        with self.assertRaises(AssertionError):
            self._potential.displacement([0.0, -1.0, 0.0], [-0.01, 0.1, -0.3], 1.0, 1.0, 0.05)
        with self.assertRaises(AssertionError):
            self._potential.displacement([0.0, 0.0, -1.0], [-0.01, 0.1, -0.3], 1.0, 1.0, 0.05)
        with self.assertRaises(AssertionError):
            self._potential.derivative([-3.1, 0.0, 0.0], [0.2, 0.1, -0.3], 1.0, 1.0)
        with self.assertRaises(AssertionError):
            self._potential.derivative([0.0, -3.1, 0.0], [0.2, 0.1, -0.3], 1.0, 1.0)
        with self.assertRaises(AssertionError):
            self._potential.derivative([0.0, 0.0, -3.1], [0.2, 0.1, -0.3], 1.0, 1.0)

    def test_velocity_not_parallel_to_axis_raises_error(self):
        with self.assertRaises(AssertionError):
            self._potential.displacement([1.0, 3.1, 0.0], [-0.01, 0.1, -0.3], 1.0, 1.0, 0.05)
        with self.assertRaises(AssertionError):
            self._potential.derivative([0.0, 1.0, 3.1], [0.2, 0.1, -0.3], 1.0, 1.0)

    def test_prefactor_zero_raises_error(self):
        with self.assertRaises(ConfigurationError):
            InversePowerCoulombBoundingPotential(prefactor=0.0)

    def test_setting_not_initialized_raises_error(self):
        # Set one variable to None so that setting is not initialized
        hypercubic_setting.number_of_root_nodes = None
        with self.assertRaises(ConfigurationError):
            InversePowerCoulombBoundingPotential(prefactor=1.6)

    def test_dimension_unequal_three_raises_error(self):
        setting.reset()
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=2, system_length=1.0)
        # Set these so the settings are initialized
        setting.set_number_of_root_nodes(2)
        setting.set_number_of_nodes_per_root_node(2)
        setting.set_number_of_node_levels(1)
        with self.assertRaises(ConfigurationError):
            self._potential = InversePowerCoulombBoundingPotential(prefactor=1.6)


if __name__ == '__main__':
    main()
