"""
Datos sembrados de las 48 selecciones del Mundial 2026.
Ratings Elo iniciales basados en eloratings.net (junio 2026, aproximación informada).
Stats xG/xGA por partido basadas en clasificatorias 2023-2026.
"""

# Formato: code, name, confederation, elo, xg_for, xg_against, attack_strength, defense_strength
# elo: rating Elo (referencia ~2100 = top mundial, ~1600 = bajo)
# xg_for: goles esperados por partido en clasificatorias/amistosos 2024-2026
# xg_against: goles esperados en contra por partido
TEAMS = [
    # CONMEBOL (6)
    ("ARG", "Argentina",     "CONMEBOL", 2143, 2.10, 0.80),
    ("BRA", "Brasil",        "CONMEBOL", 2050, 1.95, 0.95),
    ("URU", "Uruguay",       "CONMEBOL", 1955, 1.75, 1.05),
    ("COL", "Colombia",      "CONMEBOL", 1935, 1.70, 1.10),
    ("ECU", "Ecuador",       "CONMEBOL", 1820, 1.40, 1.15),
    ("PAR", "Paraguay",      "CONMEBOL", 1735, 1.20, 1.25),

    # UEFA (16)
    ("FRA", "Francia",       "UEFA",     2090, 2.05, 0.90),
    ("ESP", "España",        "UEFA",     2078, 2.10, 0.85),
    ("ENG", "Inglaterra",    "UEFA",     2040, 1.90, 0.95),
    ("POR", "Portugal",      "UEFA",     2030, 1.95, 1.00),
    ("NED", "Países Bajos",  "UEFA",     2005, 1.85, 1.05),
    ("GER", "Alemania",      "UEFA",     1985, 1.90, 1.10),
    ("ITA", "Italia",        "UEFA",     1965, 1.75, 1.00),
    ("BEL", "Bélgica",       "UEFA",     1945, 1.80, 1.10),
    ("CRO", "Croacia",       "UEFA",     1925, 1.65, 1.05),
    ("SUI", "Suiza",         "UEFA",     1885, 1.55, 1.10),
    ("DEN", "Dinamarca",     "UEFA",     1875, 1.60, 1.15),
    ("AUT", "Austria",       "UEFA",     1850, 1.55, 1.20),
    ("POL", "Polonia",       "UEFA",     1830, 1.45, 1.25),
    ("SRB", "Serbia",        "UEFA",     1820, 1.50, 1.25),
    ("TUR", "Turquía",       "UEFA",     1805, 1.55, 1.30),
    ("NOR", "Noruega",       "UEFA",     1790, 1.70, 1.30),

    # CONCACAF (6, 3 anfitriones + 3)
    ("MEX", "México",        "CONCACAF", 1810, 1.60, 1.15),
    ("USA", "Estados Unidos","CONCACAF", 1795, 1.55, 1.20),
    ("CAN", "Canadá",        "CONCACAF", 1755, 1.45, 1.30),
    ("CRC", "Costa Rica",    "CONCACAF", 1635, 1.20, 1.40),
    ("PAN", "Panamá",        "CONCACAF", 1605, 1.15, 1.45),
    ("JAM", "Jamaica",       "CONCACAF", 1585, 1.10, 1.50),

    # AFC (8)
    ("JPN", "Japón",         "AFC",      1825, 1.65, 1.10),
    ("IRN", "Irán",          "AFC",      1790, 1.55, 1.20),
    ("KOR", "Corea del Sur", "AFC",      1775, 1.50, 1.20),
    ("AUS", "Australia",     "AFC",      1740, 1.40, 1.25),
    ("KSA", "Arabia Saudita","AFC",      1660, 1.20, 1.45),
    ("QAT", "Qatar",         "AFC",      1640, 1.15, 1.45),
    ("UZB", "Uzbekistán",    "AFC",      1625, 1.10, 1.50),
    ("JOR", "Jordania",      "AFC",      1565, 1.05, 1.55),

    # CAF (9)
    ("MAR", "Marruecos",     "CAF",      1810, 1.55, 1.15),
    ("SEN", "Senegal",       "CAF",      1790, 1.60, 1.25),
    ("EGY", "Egipto",        "CAF",      1745, 1.50, 1.30),
    ("ALG", "Argelia",       "CAF",      1735, 1.45, 1.30),
    ("CIV", "Costa de Marfil","CAF",     1720, 1.50, 1.35),
    ("NGA", "Nigeria",       "CAF",      1715, 1.55, 1.40),
    ("CMR", "Camerún",       "CAF",      1685, 1.40, 1.40),
    ("TUN", "Túnez",         "CAF",      1670, 1.30, 1.40),
    ("GHA", "Ghana",         "CAF",      1655, 1.40, 1.45),

    # OFC (1)
    ("NZL", "Nueva Zelanda", "OFC",      1545, 1.10, 1.55),

    # Repechaje (2 simulados)
    ("BOL", "Bolivia",       "CONMEBOL", 1530, 1.05, 1.60),
    ("HAI", "Haití",         "CONCACAF", 1490, 1.00, 1.65),
]
