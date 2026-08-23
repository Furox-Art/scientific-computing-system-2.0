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
    "convert_units",
    "list_units",
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


# ------------------------------------------------------------ unit system ---
# factor maps each unit to its SI base; temperature uses offsets instead.
UNIT_FACTORS: dict[str, float] = {
    # length (m)
    "nm": 1e-9,
    "um": 1e-6,
    "mm": 1e-3,
    "cm": 1e-2,
    "m": 1.0,
    "km": 1e3,
    "inch": 0.0254,
    "ft": 0.3048,
    "mile": 1609.344,
    # mass (kg)
    "mg": 1e-6,
    "g": 1e-3,
    "kg": 1.0,
    "tonne": 1e3,
    "lb": 0.45359237,
    "oz": 0.028349523125,
    # time (s)
    "ms": 1e-3,
    "s": 1.0,
    "min": 60.0,
    "h": 3600.0,
    "day": 86400.0,
    # energy (J)
    "J": 1.0,
    "kJ": 1e3,
    "cal": 4.184,
    "kcal": 4184.0,
    "Wh": 3600.0,
    "kWh": 3.6e6,
    "eV": 1.602176634e-19,
    # pressure (Pa)
    "Pa": 1.0,
    "kPa": 1e3,
    "bar": 1e5,
    "atm": 101325.0,
    "mmHg": 133.322387415,
    # angle (rad)
    "rad": 1.0,
    "deg": math.pi / 180.0,
}

UNIT_DIMENSIONS: dict[str, str] = {
    **{unit: "length" for unit in ("nm", "um", "mm", "cm", "m", "km", "inch", "ft", "mile")},
    **{unit: "mass" for unit in ("mg", "g", "kg", "tonne", "lb", "oz")},
    **{unit: "time" for unit in ("ms", "s", "min", "h", "day")},
    **{unit: "energy" for unit in ("J", "kJ", "cal", "kcal", "Wh", "kWh", "eV")},
    **{unit: "pressure" for unit in ("Pa", "kPa", "bar", "atm", "mmHg")},
    **{unit: "angle" for unit in ("rad", "deg")},
}

TEMPERATURE_UNITS = {"C", "F", "K"}


def list_units() -> list[str]:
    """All supported unit symbols, including the three temperatures."""
    return sorted([*UNIT_FACTORS.keys(), *TEMPERATURE_UNITS])


def _to_kelvin(value: float, unit: str) -> float:
    if unit == "C":
        return value + 273.15
    if unit == "F":
        return (value - 32.0) * 5.0 / 9.0 + 273.15
    return value


def _from_kelvin(value: float, unit: str) -> float:
    if unit == "C":
        return value - 273.15
    if unit == "F":
        return (value - 273.15) * 9.0 / 5.0 + 32.0
    return value


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert ``value`` between supported units of one physical dimension.

    Covers length, mass, time, energy, pressure and angle via ``UNIT_FACTORS``
    plus temperatures in C/F/K. Mixing dimensions raises ``ValueError``.
    """
    source = from_unit.strip()
    target = to_unit.strip()
    source_is_temp = source in TEMPERATURE_UNITS
    target_is_temp = target in TEMPERATURE_UNITS
    if source not in UNIT_FACTORS and not source_is_temp:
        msg = f"unknown unit: {source!r}"
        raise ValueError(msg)
    if target not in UNIT_FACTORS and not target_is_temp:
        msg = f"unknown unit: {target!r}"
        raise ValueError(msg)
    if source_is_temp != target_is_temp:
        msg = (
            f"cannot convert between temperature and non-temperature units ({source!r}, {target!r})"
        )
        raise ValueError(msg)
    if source_is_temp:
        return _from_kelvin(_to_kelvin(value, source), target)
    if UNIT_DIMENSIONS[source] != UNIT_DIMENSIONS[target]:
        msg = (
            f"cannot convert {UNIT_DIMENSIONS[source]} ({source!r}) to "
            f"{UNIT_DIMENSIONS[target]} ({target!r})"
        )
        raise ValueError(msg)
    base_value = value * UNIT_FACTORS[source]
    return base_value / UNIT_FACTORS[target]
