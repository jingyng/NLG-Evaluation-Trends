#!/usr/bin/env python3
"""Normalize language stats with pycountry + aliases + ISO hints."""

import csv
import unicodedata
from collections import Counter
from pathlib import Path
import re

try:
    import pycountry  # pip install pycountry
except ImportError:
    pycountry = None

BASE = Path(__file__).parent.parent
INPUT = BASE / "metadata_unique_counts" / "languages_stats.csv"
OUTPUT = BASE / "metadata_unique_counts" / "languages_stats_normalized.csv"
MAP_OUTPUT = BASE / "metadata_unique_counts" / "languages_normalization_mapping.csv"
MERGES_OUTPUT = BASE / "metadata_unique_counts" / "languages_normalization_merges.csv"

ALIASES = {
    # regional/dialect
    "brazilian portuguese": "Portuguese",
    "egyptian arabic": "Arabic",
    "modern standard arabic": "Arabic",
    "moroccan arabic": "Arabic",
    "tunisian arabic": "Arabic",
    "mandarin": "Chinese",
    "mandarin chinese": "Chinese",
    "simplified chinese": "Chinese",
    "traditional chinese": "Chinese",
    "cantonese": "Cantonese",
    "hokkien": "Min Nan Chinese",
    "min nan": "Min Nan Chinese",
    "swiss german": "German",
    "ancient greek": "Greek",
    "modern greek": "Greek",
    "farsi": "Persian",
    "oriya": "Odia",
    "slovene": "Slovenian",
    "norwegian bokmal": "Norwegian",
    "norwegian bokmål": "Norwegian",
    "bokmal": "Norwegian",
    "bokmål": "Norwegian",
    "multilingual": "Multilingual",
    "multiple languages": "Multilingual",
    "multilingual (100+ languages)": "Multilingual",
    "multilingual (15 languages)": "Multilingual",
    "multilingual (16 languages)": "Multilingual",
    "code switching": "Code-Switching",
    "code-switching": "Code-Switching",
    "code mixed": "Code-Mixed",
    "code-mixed": "Code-Mixed",
    "sign language": "Sign Languages",
    "american sign language": "American Sign Language",
}

ISO_CODE_MAP = {
    "en": "English", "eng": "English",
    "zh": "Chinese", "zho": "Chinese", "cmn": "Chinese", "yue": "Cantonese",
    "de": "German", "deu": "German",
    "fr": "French", "fra": "French",
    "es": "Spanish", "spa": "Spanish",
    "pt": "Portuguese", "por": "Portuguese",
    "ru": "Russian", "rus": "Russian",
    "ja": "Japanese", "jpn": "Japanese",
    "ko": "Korean", "kor": "Korean",
    "ar": "Arabic", "ara": "Arabic",
    "hi": "Hindi", "hin": "Hindi",
    "bn": "Bengali", "ben": "Bengali",
    "tr": "Turkish", "tur": "Turkish",
    "vi": "Vietnamese", "vie": "Vietnamese",
    "id": "Indonesian", "ind": "Indonesian",
    "th": "Thai", "tha": "Thai",
    "pl": "Polish", "pol": "Polish",
    "fa": "Persian", "fas": "Persian",
    "ur": "Urdu", "urd": "Urdu",
    "ta": "Tamil", "tam": "Tamil",
    "te": "Telugu", "tel": "Telugu",
    "it": "Italian", "ita": "Italian",
    "uk": "Ukrainian", "ukr": "Ukrainian",
    "nl": "Dutch", "nld": "Dutch",
    "fi": "Finnish", "fin": "Finnish",
    "sv": "Swedish", "swe": "Swedish",
    "no": "Norwegian", "nor": "Norwegian",
    "cs": "Czech", "ces": "Czech",
    "hu": "Hungarian", "hun": "Hungarian",
    "ro": "Romanian", "ron": "Romanian",
    "el": "Greek", "ell": "Greek",
    "he": "Hebrew", "heb": "Hebrew",
    "sr": "Serbian", "srp": "Serbian",
    "lt": "Lithuanian", "lit": "Lithuanian",
    "lv": "Latvian", "lav": "Latvian",
    "da": "Danish", "dan": "Danish",
    "is": "Icelandic", "isl": "Icelandic",
    "ms": "Malay", "msa": "Malay",
    "mr": "Marathi", "mar": "Marathi",
    "gu": "Gujarati", "guj": "Gujarati",
    "be": "Belarusian", "bel": "Belarusian",
    "bg": "Bulgarian", "bul": "Bulgarian",
    "et": "Estonian", "est": "Estonian",
    "km": "Khmer", "khm": "Khmer",
    "my": "Burmese", "mya": "Burmese",
    "qu": "Quechua", "que": "Quechua",
    "am": "Amharic", "amh": "Amharic",
    "zu": "Zulu", "zul": "Zulu",
    "arp": "Arapaho", "aym": "Aymara", "cni": "Ashaninka", "cpa": "Chin",
    "cre": "Cree", "ddo": "Tsez", "eus": "Basque", "evn": "Evenki",
}

ISO_FOR_LANG = {
    "English": ("en", "eng"), "Chinese": ("zh", "zho"),
    "Cantonese": ("yue", "yue"), "Min Nan Chinese": ("nan", "nan"),
    "German": ("de", "deu"), "French": ("fr", "fra"), "Spanish": ("es", "spa"),
    "Portuguese": ("pt", "por"), "Arabic": ("ar", "ara"), "Russian": ("ru", "rus"),
    "Japanese": ("ja", "jpn"), "Korean": ("ko", "kor"),
    "Hindi": ("hi", "hin"), "Bengali": ("bn", "ben"), "Turkish": ("tr", "tur"),
    "Vietnamese": ("vi", "vie"), "Indonesian": ("id", "ind"),
    "Thai": ("th", "tha"), "Polish": ("pl", "pol"), "Italian": ("it", "ita"),
    "Ukrainian": ("uk", "ukr"), "Dutch": ("nl", "nld"), "Finnish": ("fi", "fin"),
    "Swedish": ("sv", "swe"), "Norwegian": ("no", "nor"), "Czech": ("cs", "ces"),
    "Hungarian": ("hu", "hun"), "Romanian": ("ro", "ron"), "Greek": ("el", "ell"),
    "Hebrew": ("he", "heb"), "Persian": ("fa", "fas"), "Urdu": ("ur", "urd"),
    "Tamil": ("ta", "tam"), "Telugu": ("te", "tel"), "Malay": ("ms", "msa"),
    "Malayalam": ("ml", "mal"), "Marathi": ("mr", "mar"), "Gujarati": ("gu", "guj"),
    "Belarusian": ("be", "bel"), "Bulgarian": ("bg", "bul"), "Estonian": ("et", "est"),
    "Lithuanian": ("lt", "lit"), "Latvian": ("lv", "lav"), "Danish": ("da", "dan"),
    "Icelandic": ("is", "isl"), "Quechua": ("qu", "que"), "Zulu": ("zu", "zul"),
    "Basque": ("eu", "eus"), "Arapaho": ("", "arp"), "Aymara": ("", "aym"),
    "Ashaninka": ("", "cni"), "Chin": ("", "cpa"), "Cree": ("cr", "cre"),
    "Tsez": ("", "ddo"), "Evenki": ("ev", "evn"),
}

def strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))

def canonical(name: str) -> str:
    name = strip_accents(name or "")
    name = name.lower().strip()
    # drop parentheses content for canonical form
    name = re.sub(r"\([^)]*\)", " ", name)
    for ch in ["_", "-", "/"]:
        name = name.replace(ch, " ")
    name = " ".join(name.split())
    return name

def pycountry_lookup(name: str):
    if not pycountry:
        return None
    # Try by name, then alpha_2, then alpha_3
    lang = pycountry.languages.get(name=name)
    if not lang:
        lang = pycountry.languages.get(alpha_2=name.lower())
    if not lang:
        lang = pycountry.languages.get(alpha_3=name.lower())
    if lang:
        iso1 = getattr(lang, "alpha_2", "")
        iso3 = getattr(lang, "alpha_3", "")
        canonical = lang.name
        return canonical, iso1, iso3
    return None

def normalize_lang(name: str):
    if not name:
        return None, "", ""
    key = canonical(name)
    if key in ALIASES:
        canonical_name = ALIASES[key]
        iso1, iso3 = ISO_FOR_LANG.get(canonical_name, ("", ""))
        return canonical_name, iso1, iso3
    if key in ISO_CODE_MAP:
        canonical_name = ISO_CODE_MAP[key]
        iso1, iso3 = ISO_FOR_LANG.get(canonical_name, ("", ""))
        return canonical_name, iso1, iso3
    pc = pycountry_lookup(name.strip())
    if pc:
        canonical_name, iso1, iso3 = pc
        return canonical_name, iso1, iso3
    return key.title(), "", ""

def main():
    aggregated = Counter()
    mapping_rows = []
    merges_rows = []
    norm_to_orig = {}
    with open(INPUT, newline="") as f:
        for row in csv.DictReader(f):
            original = row.get("language", "")
            count = int(row.get("count", 0) or 0)
            canonical_name, iso1, iso3 = normalize_lang(original)
            if not canonical_name:
                continue
            aggregated[canonical_name] += count
            mapping_rows.append((original, canonical_name, iso1, iso3, count))
            norm_to_orig.setdefault(canonical_name, Counter()).update({original: count})

    rows = sorted(aggregated.items(), key=lambda x: (-x[1], x[0]))
    with open(OUTPUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["language", "iso_639_1", "iso_639_3", "count"])
        for lang, cnt in rows:
            iso1, iso3 = ISO_FOR_LANG.get(lang, ("", ""))
            w.writerow([lang, iso1, iso3, cnt])

    with open(MAP_OUTPUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["original", "normalized", "iso_639_1", "iso_639_3", "count"])
        w.writerows(mapping_rows)

    # Build merges report (exact normalized label with multiple originals)
    for norm, ctr in norm_to_orig.items():
        if len(ctr) <= 1:
            continue
        total = sum(ctr.values())
        variants = "; ".join([f"{o} ({c})" for o, c in sorted(ctr.items(), key=lambda x: (-x[1], x[0]))])
        merges_rows.append((norm, total, len(ctr), variants))
    merges_rows.sort(key=lambda x: (-x[1], -x[2], x[0]))

    with open(MERGES_OUTPUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["normalized_language", "total_count", "num_variants", "variants_with_counts"])
        w.writerows(merges_rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT}")
    print(f"Wrote {len(mapping_rows)} mapping rows to {MAP_OUTPUT}")
    print(f"Wrote {len(merges_rows)} merge groups to {MERGES_OUTPUT}")
    print("\nTop 15 normalized languages:")
    for lang, cnt in rows[:15]:
        iso1, iso3 = ISO_FOR_LANG.get(lang, ("", ""))
        suffix = f" ({iso1}/{iso3})" if iso1 or iso3 else ""
        print(f"  {lang}{suffix}: {cnt}")

if __name__ == "__main__":
    main()
