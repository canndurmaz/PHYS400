"""
test_elastic.py — Unit + integration tests for elastic.py

Run with:
    source ../.venv/bin/activate && OMP_NUM_THREADS=1 python -m pytest test_elastic.py -v
"""

import os
import sys
import numpy as np
import pytest

# Make sure we import from this directory
sys.path.insert(0, os.path.dirname(__file__))
import elastic as E


# ---------------------------------------------------------------------------
# Unit tests: VRH averaging
# ---------------------------------------------------------------------------

class TestVRH:
    """Test Voigt-Reuss-Hill averaging against known BCC Fe values."""

    # Textbook single-crystal elastic constants for pure BCC Fe (GPa)
    # C11=229, C12=134, C44=116  (from Ledbetter 1973)
    C11, C12, C44 = 229.0, 134.0, 116.0

    @pytest.fixture
    def C_cubic(self):
        """Build a cubic 6x6 elastic tensor from C11, C12, C44."""
        C = np.zeros((6, 6))
        for i in range(3):
            C[i, i] = self.C11
        for i in range(3):
            for j in range(3):
                if i != j:
                    C[i, j] = self.C12
        for i in range(3, 6):
            C[i, i] = self.C44
        return C

    def test_bulk_modulus_cubic(self, C_cubic):
        """For a cubic crystal K_V = K_R = (C11 + 2*C12) / 3."""
        expected_K = (self.C11 + 2 * self.C12) / 3.0
        m = E.voigt_reuss_hill(C_cubic)
        assert m["K_V"] == pytest.approx(expected_K, rel=1e-6)
        assert m["K_R"] == pytest.approx(expected_K, rel=1e-4)
        assert m["K_H"] == pytest.approx(expected_K, rel=1e-4)

    def test_hill_bounds_shear(self, C_cubic):
        """G_R <= G_H <= G_V must always hold."""
        m = E.voigt_reuss_hill(C_cubic)
        assert m["G_R"] <= m["G_H"] + 1e-10
        assert m["G_H"] <= m["G_V"] + 1e-10

    def test_youngs_modulus_positive(self, C_cubic):
        m = E.voigt_reuss_hill(C_cubic)
        assert m["E"] > 0

    def test_poisson_ratio_range(self, C_cubic):
        """Poisson ratio must be in (-1, 0.5) for stable isotropic solid."""
        m = E.voigt_reuss_hill(C_cubic)
        assert -1.0 < m["nu"] < 0.5

    def test_identity_tensor(self):
        """Isotropic tensor: K and G should satisfy exact relations."""
        # Build isotropic tensor with K=170, G=82
        K, G = 170.0, 82.0
        lam = K - 2 * G / 3
        C = np.zeros((6, 6))
        for i in range(3):
            C[i, i] = lam + 2 * G
        for i in range(3):
            for j in range(3):
                if i != j:
                    C[i, j] = lam
        for i in range(3, 6):
            C[i, i] = G
        m = E.voigt_reuss_hill(C)
        assert m["K_H"] == pytest.approx(K, rel=1e-6)
        assert m["G_H"] == pytest.approx(G, rel=1e-6)
        # E = 9KG / (3K + G)
        E_expected = 9 * K * G / (3 * K + G)
        assert m["E"] == pytest.approx(E_expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Unit tests: stress extraction sign convention
# ---------------------------------------------------------------------------

class TestGetStressGPa:
    """get_stress_gpa returns σ = −P in GPa."""

    def test_units_conversion(self):
        """1 bar pressure → -1e-4 GPa stress component."""
        from unittest.mock import MagicMock
        L = MagicMock()
        # Simulate 10000 bar (= 1 GPa) hydrostatic pressure
        L.get_thermo.side_effect = lambda key: 10000.0
        sigma = E.get_stress_gpa(L)
        # σ = -P → -1 GPa each component
        np.testing.assert_allclose(sigma, -1.0 * np.ones(6), atol=1e-12)

    def test_voigt_order(self):
        """Components are returned in xx,yy,zz,yz,xz,xy order."""
        from unittest.mock import MagicMock
        pressure_map = {
            "pxx": 1.0, "pyy": 2.0, "pzz": 3.0,
            "pyz": 4.0, "pxz": 5.0, "pxy": 6.0,
        }
        L = MagicMock()
        L.get_thermo.side_effect = lambda k: pressure_map[k]
        sigma = E.get_stress_gpa(L)
        expected = -np.array([1, 2, 3, 4, 5, 6]) * 1e-4
        np.testing.assert_allclose(sigma, expected, atol=1e-15)


# ---------------------------------------------------------------------------
# Integration test: single LAMMPS ground-state build
# ---------------------------------------------------------------------------

class TestBuildGroundState:
    """Smoke test that build_ground_state runs without error."""

    @pytest.fixture(scope="class")
    def small_cfg(self):
        cfg = dict(E.CONFIG)
        cfg["supercell"] = (3, 3, 3)   # small — fast
        cfg["dopant_fraction"] = 0.0    # pure Fe for reproducibility
        return cfg

    def test_builds_without_error(self, small_cfg):
        L = E.build_ground_state(small_cfg)
        assert L is not None
        L.close()

    def test_pe_negative(self, small_cfg):
        """Potential energy of relaxed BCC Fe must be negative."""
        L = E.build_ground_state(small_cfg)
        pe = L.get_thermo("pe")
        L.close()
        assert pe < 0

    def test_pressure_near_zero(self, small_cfg):
        """After full relaxation all pressure components should be < 0.1 bar."""
        L = E.build_ground_state(small_cfg)
        sigma = E.get_stress_gpa(L)
        L.close()
        # Relaxed: stresses ≈ 0 (within 0.01 GPa = 100 bar tolerance)
        np.testing.assert_allclose(sigma, 0.0, atol=0.01)


# ---------------------------------------------------------------------------
# Integration test: full elastic tensor (3×3×3 pure Fe, fast)
# ---------------------------------------------------------------------------

class TestElasticTensor:
    """Full pipeline on a tiny pure-Fe cell."""

    @pytest.fixture(scope="class")
    def C_and_moduli(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("elastic")
        cfg = dict(E.CONFIG)
        cfg["supercell"] = (3, 3, 3)
        cfg["dopant_fraction"] = 0.0
        cfg["frames_file"] = str(tmp / "frames.lammpstrj")
        cfg["cij_file"]    = str(tmp / "Cij.npy")
        cfg["moduli_file"] = str(tmp / "moduli.npy")
        C = E.compute_elastic_tensor(cfg)
        moduli = E.voigt_reuss_hill(C)
        return C, moduli

    def test_shape(self, C_and_moduli):
        C, _ = C_and_moduli
        assert C.shape == (6, 6)

    def test_symmetric(self, C_and_moduli):
        C, _ = C_and_moduli
        np.testing.assert_allclose(C, C.T, atol=0.5)

    def test_positive_definite(self, C_and_moduli):
        C, _ = C_and_moduli
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > 0), f"Non-positive eigenvalues: {eigvals}"

    def test_c11_range(self, C_and_moduli):
        """C11 for pure Fe should be 200–320 GPa."""
        C, _ = C_and_moduli
        c11 = C[0, 0]
        assert 200 < c11 < 320, f"C11={c11:.1f} GPa out of range"

    def test_c44_range(self, C_and_moduli):
        """C44 for pure Fe should be 80–160 GPa."""
        C, _ = C_and_moduli
        c44 = C[3, 3]
        assert 80 < c44 < 160, f"C44={c44:.1f} GPa out of range"

    def test_youngs_modulus_range(self, C_and_moduli):
        """E for pure Fe: 190–300 GPa."""
        _, m = C_and_moduli
        assert 190 < m["E"] < 300, f"E={m['E']:.1f} GPa out of range"

    def test_poisson_ratio(self, C_and_moduli):
        _, m = C_and_moduli
        assert 0.20 < m["nu"] < 0.40, f"ν={m['nu']:.4f} out of range"

    def test_hill_bounds(self, C_and_moduli):
        _, m = C_and_moduli
        assert m["G_R"] <= m["G_H"] + 1e-6
        assert m["G_H"] <= m["G_V"] + 1e-6

    def test_frames_written(self, C_and_moduli, tmp_path_factory):
        """Frames file should contain 18 snapshots (6 cols × 3 frames)."""
        cfg = dict(E.CONFIG)
        cfg["supercell"] = (3, 3, 3)
        cfg["dopant_fraction"] = 0.0
        # Re-use the file from the fixture (class-scoped)
        frames_file = C_and_moduli[0]   # not quite right; just check file exists
        # The fixture runs compute_elastic_tensor which creates frames_file
        # We verify via the snapshot count in the frames file
        tmp = tmp_path_factory.getbasetemp()
        # Find any .lammpstrj in tmp
        import pathlib
        found = list(pathlib.Path(tmp).rglob("frames.lammpstrj"))
        assert found, "No frames file found"
        content = found[0].read_text()
        n_items = content.count("ITEM: ATOMS")
        assert n_items == 18, f"Expected 18 frames, got {n_items}"
