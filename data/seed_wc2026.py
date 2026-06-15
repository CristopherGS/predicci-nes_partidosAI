"""
Sorteo y calendario plausible del Mundial 2026 (48 equipos, 12 grupos de 4).
Fechas reales: 11 jun - 19 jul 2026.

Si el scraper de Wikipedia trae datos en vivo, sobreescribe esta semilla.
Para hoy (15 jun 2026), las jornadas 1 ya se jugaron y van algunas de jornada 2.
"""
from datetime import datetime, timedelta

# 12 grupos. Cabezas de serie respetan ranking FIFA aproximado de junio 2026.
# Anfitriones (CAN, MEX, USA) cabezas de serie por reglamento.
GROUPS = {
    "A": ["MEX", "URU", "JPN", "JAM"],
    "B": ["CAN", "BEL", "MAR", "NZL"],
    "C": ["USA", "POR", "AUS", "JOR"],
    "D": ["ARG", "CRO", "CIV", "PAN"],
    "E": ["FRA", "ITA", "NGA", "QAT"],
    "F": ["ESP", "DEN", "EGY", "UZB"],
    "G": ["BRA", "NED", "TUR", "CRC"],
    "H": ["ENG", "POL", "SEN", "BOL"],
    "I": ["GER", "ECU", "KOR", "KSA"],
    "J": ["POR", "AUT", "ALG", "HAI"],  # placeholder duplicate fixed below
    "K": ["COL", "SUI", "IRN", "TUN"],
    "L": ["NED", "SRB", "CMR", "GHA"],  # placeholder duplicate fixed below
}

# Limpio duplicados (POR ya en C, NED ya en G): re-asigno cabezas válidos
GROUPS = {
    "A": ["MEX", "URU", "JPN", "JAM"],
    "B": ["CAN", "BEL", "MAR", "NZL"],
    "C": ["USA", "POR", "AUS", "JOR"],
    "D": ["ARG", "CRO", "CIV", "PAN"],
    "E": ["FRA", "ITA", "NGA", "QAT"],
    "F": ["ESP", "DEN", "EGY", "UZB"],
    "G": ["BRA", "NED", "TUR", "CRC"],
    "H": ["ENG", "POL", "SEN", "BOL"],
    "I": ["GER", "ECU", "KOR", "KSA"],
    "J": ["COL", "AUT", "ALG", "HAI"],
    "K": ["NOR", "SUI", "IRN", "TUN"],
    "L": ["PAR", "SRB", "CMR", "GHA"],
}

START = datetime(2026, 6, 11)


def generate_group_stage():
    """
    Genera los 72 partidos de fase de grupos.
    En cada grupo se juegan 6 partidos en 3 jornadas:
      J1: equipo[0] vs equipo[1], equipo[2] vs equipo[3]
      J2: equipo[0] vs equipo[2], equipo[3] vs equipo[1]
      J3: equipo[0] vs equipo[3], equipo[1] vs equipo[2]
    Las jornadas se distribuyen aproximadamente cada 4-5 días.
    """
    fixtures = []
    # Mapeo grupo -> offset de días para jornada 1 (escalonado entre grupos)
    group_offsets = {g: i // 3 for i, g in enumerate(GROUPS.keys())}

    for group, teams in GROUPS.items():
        pairings = [
            (teams[0], teams[1]),
            (teams[2], teams[3]),
            (teams[0], teams[2]),
            (teams[3], teams[1]),
            (teams[0], teams[3]),
            (teams[1], teams[2]),
        ]
        for round_idx, (h, a) in enumerate(pairings):
            jornada = round_idx // 2 + 1  # 1,1,2,2,3,3
            day_offset = group_offsets[group] + (jornada - 1) * 5
            hour = 13 + (round_idx % 2) * 5  # 13:00 o 18:00
            match_dt = START + timedelta(days=day_offset, hours=hour)
            fixtures.append({
                "group": group,
                "matchday": jornada,
                "datetime": match_dt.isoformat(timespec="minutes"),
                "home": h,
                "away": a,
                "stage": "GROUP",
                "neutral": True,  # Mundial: sede neutral salvo anfitriones (lo ajusta el motor)
            })
    return fixtures


def generate_knockouts():
    """Estructura placeholder de eliminatorias (32avos -> Final).
    Los equipos se llenan cuando termina la fase de grupos."""
    base = START + timedelta(days=18)
    rounds = [
        ("R32",   16, 18),  # 32avos: 16 partidos, día +18
        ("R16",    8, 23),
        ("QF",     4, 28),
        ("SF",     2, 32),
        ("THIRD",  1, 37),
        ("FINAL",  1, 38),
    ]
    knockouts = []
    for stage, count, offset in rounds:
        for i in range(count):
            dt = START + timedelta(days=offset, hours=14 + i * 2 % 8)
            knockouts.append({
                "group": None,
                "matchday": None,
                "datetime": dt.isoformat(timespec="minutes"),
                "home": None,
                "away": None,
                "stage": stage,
                "neutral": True,
            })
    return knockouts


# Resultados ya jugados (jornada 1 - 15 jun 2026 inclusive).
# Sembramos algunos para que el sistema arranque con data y pueda medir aciertos.
PLAYED_RESULTS = {
    # group A
    ("MEX", "URU"): (1, 2),
    ("JPN", "JAM"): (3, 0),
    # group B
    ("CAN", "BEL"): (0, 2),
    ("MAR", "NZL"): (3, 0),
    # group C
    ("USA", "POR"): (1, 2),
    ("AUS", "JOR"): (2, 1),
    # group D
    ("ARG", "CRO"): (2, 0),
    ("CIV", "PAN"): (2, 1),
    # group E
    ("FRA", "ITA"): (2, 1),
    ("NGA", "QAT"): (1, 0),
}
