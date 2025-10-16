# parsers/td3.py
from typing import Dict, Optional, Tuple

_ALPHA_FROM_DIGIT = str.maketrans({"0":"O","1":"I","2":"Z","5":"S","8":"B"})

_WEIGHTS = [7, 3, 1]

def _char_value(ch: str) -> int:
    if ch == '<': return 0
    if ch.isdigit(): return ord(ch) - ord('0')
    if 'A' <= ch <= 'Z': return 10 + (ord(ch) - ord('A'))
    return 0



def _checksum(s: str) -> str:
    return str(sum(_char_value(ch) * _WEIGHTS[i % 3] for i, ch in enumerate(s)) % 10)

def _yyMMdd_to_iso(s: str) -> Optional[str]:
    if len(s) != 6 or not s.isdigit(): return None
    yy = int(s[:2]); mm = s[2:4]; dd = s[4:6]
    cent = 1900 if yy >= 50 else 2000
    return f"{cent+yy:04d}-{mm}-{dd}"

def _parse_names(field: str) -> Tuple[str, str]:
    parts = field.split("<<", 1)
    surname = parts[0].replace('<', ' ').strip()
    given = "" if len(parts) == 1 else parts[1].replace('<', ' ').strip()
    return surname, given

def parse_td3(L1: str, L2: str) -> Dict:
    """
    Strict TD3 MRZ parser for two lines (each 44 chars). Pads/trims to 44,
    returns structured fields and checksum booleans.
    """
    L1 = (L1.upper().replace(" ", "") + "<"*44)[:44]
    L2 = (L2.upper().replace(" ", "") + "<"*44)[:44]
    if len(L1) != 44 or len(L2) != 44:
        raise ValueError("TD3 lines must be 44 chars each")

    # Line 1
    doc_type   = L1[0]
    country    = L1[2:5]
    name_field = L1[5:44]

    # Line 2
    passport_no = L2[0:9]; c1 = L2[9]
    nationality = L2[10:13].translate(_ALPHA_FROM_DIGIT)
    birth_date  = L2[13:19]; c2 = L2[19]
    sex         = L2[20]
    expiry_date = L2[21:27]; c3 = L2[27]
    personal_no = L2[28:42]; c4 = L2[42]
    c_final     = L2[43]
    
    

    # Checksums
    ok1 = _checksum(passport_no) == c1
    ok2 = _checksum(birth_date)  == c2
    ok3 = _checksum(expiry_date) == c3
    ok4 = _checksum(personal_no) == c4
    composite = passport_no + c1 + birth_date + c2 + expiry_date + c3 + personal_no + c4
    okF = _checksum(composite) == c_final

    surname, given = _parse_names(name_field)

    return {
        "doc_type": doc_type,
        "country": country,
        "surname": surname,
        "given_names": given,
        "passport_number": passport_no,
        "passport_number_checksum_ok": ok1,
        "nationality": nationality,
        "birth_date": _yyMMdd_to_iso(birth_date),
        "birth_date_checksum_ok": ok2,
        "sex": sex if sex in ("M", "F") else "X",
        "expiry_date": _yyMMdd_to_iso(expiry_date),
        "expiry_date_checksum_ok": ok3,
        "personal_number": (personal_no.replace('<', ' ').strip() or None),
        "personal_number_checksum_ok": ok4,
        "final_checksum_ok": okF,
        "raw": {"L1": L1, "L2": L2},
    }
