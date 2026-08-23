"""Physical constants and formula helpers."""

from __future__ import annotations

import math

__all__ = [
    "CONSTANTS",
    "speed_of_light",
    "planck_constant",
    "boltzmann_constant",
    "gravitational_constant",
    "elementary_charge",
    "avogadro_number",
    "vacuum_permittivity",
    "vacuum_permeability",
    "standard_gravity",
    "kinetic_energy",
    "gravitational_force",
    "coulomb_force",
    "photon_energy",
    "de_broglie_wavelength",
    "lorentz_factor",
    "ideal_gas_pressure",
    "escape_velocity",
]

CONSTANTS: dict[str, float] = {
    "c": 299_792_458.0,
    "h": 6.62607015e-34,
    "hbar": 1.0545718176461565e-34,
    "G": 6.67430e-11,
    "k_B": 1.380649e-23,
    "e": 1.602176634e-19,
    "N_A": 6.02214076e23,
    "epsilon_0": 8.8541878128e-12,
    "mu_0": 1.25663706212e-6,
    "g": 9.80665,
    "m_e": 9.1093837015e-31,
    "m_p": 1.67262192369e-27,
}


def speed_of_light() -> float:
    """Speed of light in vacuum, m/s."""
    return CONSTANTS["c"]


def planck_constant() -> float:
    """Planck constant, J*s."""
    return CONSTANTS["h"]


def boltzmann_constant() -> float:
    """Boltzmann constant, J/K."""
    return CONSTANTS["k_B"]


def gravitational_constant() -> float:
    """Newtonian constant of gravitation."""
    return CONSTANTS["G"]


def elementary_charge() -> float:
    """Elementary charge, C."""
    return CONSTANTS["e"]


def avogadro_number() -> float:
    """Avogadro constant, 1/mol."""
    return CONSTANTS["N_A"]


def vacuum_permittivity() -> float:
    """Vacuum electric permittivity, F/m."""
    return CONSTANTS["epsilon_0"]


def vacuum_permeability() -> float:
    """Vacuum magnetic permeability, H/m."""
    return CONSTANTS["mu_0"]


def standard_gravity() -> float:
    """Standard Earth gravity, m/s^2."""
    return CONSTANTS["g"]


def kinetic_energy(mass: float, velocity: float) -> float:
    """Classical kinetic energy, J."""
    if mass < 0:
        msg = "mass cannot be negative"
        raise ValueError(msg)
    return 0.5 * mass * velocity**2


def gravitational_force(mass_a: float, mass_b: float, distance: float) -> float:
    """Newtonian gravitational attraction between two point masses, N."""
    if distance <= 0:
        msg = "distance must be positive"
        raise ValueError(msg)
    return CONSTANTS["G"] * mass_a * mass_b / distance**2


def coulomb_force(charge_a: float, charge_b: float, distance: float) -> float:
    """Electrostatic force between two point charges, N (signed)."""
    if distance <= 0:
        msg = "distance must be positive"
        raise ValueError(msg)
    return charge_a * charge_b / (4.0 * math.pi * CONSTANTS["epsilon_0"] * distance**2)


def photon_energy(wavelength_m: float) -> float:
    """Energy of a photon from its vacuum wavelength, J."""
    if wavelength_m <= 0:
        msg = "wavelength must be positive"
        raise ValueError(msg)
    return CONSTANTS["h"] * CONSTANTS["c"] / wavelength_m


def de_broglie_wavelength(mass: float, velocity: float) -> float:
    """Matter wavelength of a moving particle, m."""
    if mass <= 0:
        msg = "mass must be positive"
        raise ValueError(msg)
    return CONSTANTS["h"] / (mass * abs(velocity))


def lorentz_factor(velocity: float) -> float:
    """Special-relativistic gamma for a speed in m/s."""
    ratio = abs(velocity) / CONSTANTS["c"]
    if ratio >= 1.0:
        msg = "speed must be below the speed of light"
        raise ValueError(msg)
    return 1.0 / math.sqrt(1.0 - ratio * ratio)


def ideal_gas_pressure(moles: float, temperature_k: float, volume_m3: float) -> float:
    """Ideal gas law pressure, Pa (``moles`` in mol, converted via Avogadro)."""
    if volume_m3 <= 0:
        msg = "volume must be positive"
        raise ValueError(msg)
    if temperature_k <= 0:
        msg = "temperature must be positive"
        raise ValueError(msg)
    molecule_count = moles * CONSTANTS["N_A"]
    return molecule_count * CONSTANTS["k_B"] * temperature_k / volume_m3


def escape_velocity(mass_kg: float, radius_m: float) -> float:
    """Escape velocity from a spherical body, m/s."""
    if mass_kg <= 0 or radius_m <= 0:
        msg = "mass and radius must be positive"
        raise ValueError(msg)
    return math.sqrt(2.0 * CONSTANTS["G"] * mass_kg / radius_m)
