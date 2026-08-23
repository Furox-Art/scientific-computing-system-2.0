"""Tests for cds2.scientific constants and formulas."""

import math

import pytest

from cds2 import scientific


class TestConstants:
    def test_speed_of_light_exact(self) -> None:
        assert scientific.speed_of_light() == 299_792_458.0

    def test_constants_table_complete(self) -> None:
        expected = {
            "c",
            "h",
            "hbar",
            "G",
            "k_B",
            "e",
            "N_A",
            "epsilon_0",
            "mu_0",
            "g",
            "m_e",
            "m_p",
        }
        assert set(scientific.CONSTANTS) == expected

    def test_accessor_functions_match_table(self) -> None:
        assert scientific.planck_constant() == scientific.CONSTANTS["h"]
        assert scientific.boltzmann_constant() == scientific.CONSTANTS["k_B"]
        assert scientific.gravitational_constant() == scientific.CONSTANTS["G"]
        assert scientific.elementary_charge() == scientific.CONSTANTS["e"]
        assert scientific.avogadro_number() == scientific.CONSTANTS["N_A"]
        assert scientific.vacuum_permittivity() == scientific.CONSTANTS["epsilon_0"]
        assert scientific.vacuum_permeability() == scientific.CONSTANTS["mu_0"]
        assert scientific.standard_gravity() == scientific.CONSTANTS["g"]


class TestMechanicsFormulas:
    def test_kinetic_energy(self) -> None:
        assert scientific.kinetic_energy(2.0, 3.0) == pytest.approx(9.0)

    def test_kinetic_energy_negative_mass(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            scientific.kinetic_energy(-1.0, 1.0)

    def test_gravitational_force_inverse_square(self) -> None:
        force = scientific.gravitational_force(1.0, 1.0, 2.0)
        assert force == pytest.approx(scientific.CONSTANTS["G"] / 4.0)

    def test_gravitational_distance_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            scientific.gravitational_force(1.0, 1.0, 0.0)

    def test_escape_velocity_earth(self) -> None:
        earth_mass = 5.972e24
        earth_radius = 6.371e6
        assert scientific.escape_velocity(earth_mass, earth_radius) == pytest.approx(
            11_186.0, rel=1e-3
        )


class TestElectromagnetismAndQuantum:
    def test_coulomb_signed_force(self) -> None:
        like = scientific.coulomb_force(1e-6, 1e-6, 0.1)
        unlike = scientific.coulomb_force(1e-6, -1e-6, 0.1)
        assert like > 0
        assert unlike == pytest.approx(-like)

    def test_photon_energy_visible_light(self) -> None:
        energy = scientific.photon_energy(500e-9)
        assert energy == pytest.approx(3.97e-19, rel=1e-3)

    def test_de_broglie_electron(self) -> None:
        wavelength = scientific.de_broglie_wavelength(scientific.CONSTANTS["m_e"], 1e6)
        assert wavelength == pytest.approx(7.27e-10, rel=1e-3)

    def test_lorentz_factor_limits(self) -> None:
        assert scientific.lorentz_factor(0.0) == pytest.approx(1.0)
        half_c = scientific.lorentz_factor(scientific.CONSTANTS["c"] / 2)
        assert half_c == pytest.approx(2.0 / math.sqrt(3.0))

    def test_lorentz_rejects_superluminal(self) -> None:
        with pytest.raises(ValueError, match="speed of light"):
            scientific.lorentz_factor(scientific.CONSTANTS["c"] * 1.1)

    def test_ideal_gas_pressure(self) -> None:
        pressure = scientific.ideal_gas_pressure(1.0, 273.15, 1.0)
        assert pressure == pytest.approx(2271.1, rel=1e-2)

    def test_thermodynamic_validation(self) -> None:
        with pytest.raises(ValueError, match="volume"):
            scientific.ideal_gas_pressure(1.0, 300.0, 0.0)
        with pytest.raises(ValueError, match="temperature"):
            scientific.ideal_gas_pressure(1.0, 0.0, 1.0)
