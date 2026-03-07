class Element:
    """Represents a MEAM element with its physical properties."""

    def __init__(self, symbol, lattice_type, coord_number, atomic_number,
                 mass, lattice_constant, meam_index):
        self.symbol = symbol
        self.lattice_type = lattice_type
        self.coord_number = coord_number
        self.atomic_number = atomic_number
        self.mass = mass
        self.lattice_constant = lattice_constant
        self.meam_index = meam_index

    def __repr__(self):
        return f"Element({self.symbol}, {self.lattice_type}, a={self.lattice_constant} Å)"


# Predefined instances — data from library.meam
# meam_index is the position in FeMnNiTiCuCrCoAl.meam (1-8)
Fe = Element("Fe", "bcc", 8,  26, 55.845,   2.8285, 1)
Mn = Element("Mn", "hcp", 12,  1, 54.940,   2.4708, 2)  # atomic_number placeholder in library.meam
Ni = Element("Ni", "fcc", 12, 28, 58.6934,  3.5214, 3)
Ti = Element("Ti", "hcp", 12,  1, 47.880,   2.9200, 4)  # atomic_number placeholder in library.meam
Cu = Element("Cu", "fcc", 12, 29, 63.546,   3.6200, 5)
Cr = Element("Cr", "bcc", 8,   1, 51.960,   2.8810, 6)  # atomic_number placeholder in library.meam
Co = Element("Co", "hcp", 12, 27, 58.933,   2.5000, 7)
Al = Element("Al", "fcc", 12, 13, 26.9815,  4.0500, 8)

# Ordered list matching FeMnNiTiCuCrCoAl.meam numbering
MEAM_ELEMENT_ORDER = ["Fe", "Mn", "Ni", "Ti", "Cu", "Cr", "Co", "Al"]

# Lookup by symbol
ELEMENTS = {
    "Fe": Fe, "Mn": Mn, "Ni": Ni, "Ti": Ti,
    "Cu": Cu, "Cr": Cr, "Co": Co, "Al": Al,
}
