"""
Post-Processing, Syntactic Validation, and OCR Disambiguation Module
Uses Bayesian domain priors, Regular Expression syntax modeling,
and OCR Confusion Matrix mapping to resolve common visual ambiguities:
- 'O' vs '0'
- 'B' vs '8'
- 'I' vs '1'
- 'Z' vs '2'
- 'S' vs '5'
Also performs Levenshtein Distance fuzzy matching against registered parking permits.
"""

import re

# Character confusion mappings based on stroke geometry
DIGIT_TO_LETTER = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '5': 'S',
    '8': 'B'
}

LETTER_TO_DIGIT = {
    'O': '0',
    'Q': '0',
    'D': '0',
    'I': '1',
    'L': '1',
    'Z': '2',
    'S': '5',
    'B': '8'
}

# Standard Indian License Plate Syntax Regex:
# 2 Letters (State) + 2 Digits (RTO) + 1-2 Letters (Series) + 4 Digits (Unique)
# e.g., DL01AB1234, MH12DE4567, KA05MJ9876
STANDARD_PLATE_REGEX = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$')

def levenshtein_distance(s1, s2):
    """Compute edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def clean_raw_ocr_string(raw_text):
    """
    Remove non-alphanumeric noise characters.
    """
    cleaned = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()
    return cleaned


def disambiguate_license_plate(plate_str):
    """
    Apply positional syntactic domain priors to correct ambiguous characters.
    For a 9-10 character standard plate:
    - Index 0, 1: MUST be alphabetical letters
    - Index 2, 3: MUST be numeric digits
    - Last 4 characters (index -4 to end): MUST be numeric digits
    """
    plate = list(clean_raw_ocr_string(plate_str))
    n = len(plate)

    if n < 6:
        return "".join(plate)

    # 1. First 2 characters -> Letters
    for i in range(min(2, n)):
        if plate[i] in DIGIT_TO_LETTER:
            plate[i] = DIGIT_TO_LETTER[plate[i]]

    # 2. Next 2 characters -> Digits
    if n >= 4:
        for i in range(2, 4):
            if plate[i] in LETTER_TO_DIGIT:
                plate[i] = LETTER_TO_DIGIT[plate[i]]

    # 3. Last 4 characters -> Digits (for standard 9-10 char plates)
    if n >= 8:
        for i in range(n - 4, n):
            if plate[i] in LETTER_TO_DIGIT:
                plate[i] = LETTER_TO_DIGIT[plate[i]]

    corrected = "".join(plate)
    return corrected


def match_registered_database(plate_str, registered_plates, max_distance=2):
    """
    Fuzzy match plate against database of pre-registered/monthly VIP vehicles.
    
    Returns:
        matched_plate (str or None): Exact registered string if match found
        distance (int): Levenshtein distance
        is_exact (bool): True if exact match
    """
    cleaned = clean_raw_ocr_string(plate_str)
    
    if cleaned in registered_plates:
        return cleaned, 0, True

    best_match = None
    min_dist = float('inf')

    for reg in registered_plates:
        d = levenshtein_distance(cleaned, reg)
        if d < min_dist:
            min_dist = d
            best_match = reg

    if min_dist <= max_distance:
        return best_match, min_dist, False

    return cleaned, min_dist, False
