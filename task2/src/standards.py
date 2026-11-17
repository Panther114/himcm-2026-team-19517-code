"""Design-standard occupant densities and a machine-readable citation registry.

This module makes the density numbers auditable end to end:

    data/standards/<standard table>.csv      tabulated default occupant densities
    data/standards/building_type_map.csv     building type -> standard row
    data/standards/citations.csv             stable [n] keys + BibTeX keys
    data/density_profile.csv                 hourly occupancy shape phi(h)
            |
            v
    build_density() -> rho by building type + density_provenance.csv
            |
            v
    citations.txt / references.bib          ready-to-paste reference list

The two encoded tables are the rows relevant to this study, transcribed from
the public reproductions listed as references [6] and [7]. The primary
standards [1] and [2] are the authoritative sources to cite.

Unit convention
---------------
ASHRAE 62.1-2022 Table 6-1 heads its occupant-density column
"#/1000 ft2 or #/100 m2". We take the SI reading, so a tabulated number N
means N people per 100 m2, i.e. rho_peak = N/100 persons/m2.

California Title 24 Part 6 Table 120.1-A states persons/1000 ft2. For the
residential and "all others" rows we apply the same design number per 100 m2
so the whole profile stays on one SI basis. This is a modelling choice, not an
exact unit conversion: a strict conversion would be N/92.90304. Both values
are written to density_provenance.csv so the choice is visible.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

M2_PER_1000_FT2 = 92.90304

TABLE_FILES = {
    "ASHRAE62_1_2022": "ashrae_62_1_2022_table_6_1.csv",
    "CA_T24_2025": "ca_title24_2025_table_120_1_a.csv",
}

UNIT_BASIS = {
    "ashrae_si_column": (
        "ASHRAE 62.1-2022 Table 6-1 heads the column '#/1000 ft2 or "
        "#/100 m2'; the SI reading gives rho_peak = N/100 persons/m2."),
    "title24_design_number": (
        "Table 120.1-A states persons/1000 ft2; we apply the same design "
        "number per 100 m2 to keep one SI basis. Strict conversion would be "
        "rho_peak = N/92.90304."),
}

_BIBTEX_ESCAPE = {
    "\\": r"\textbackslash{}", "#": r"\#", "%": r"\%", "&": r"\&",
    "_": r"\_", "$": r"\$", "{": r"\{", "}": r"\}",
}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_citations(standards_dir):
    """key -> citation record (ref_number, short, bibtex_key, ...)."""
    rows = read_csv(Path(standards_dir) / "citations.csv")
    out = {}
    for r in rows:
        if r["key"] in out:
            raise SystemExit(f"duplicate citation key: {r['key']}")
        out[r["key"]] = r
    return out


def load_tables(standards_dir):
    """citation key -> {'rows', 'index', 'filename'}."""
    out = {}
    for key, name in TABLE_FILES.items():
        rows = read_csv(Path(standards_dir) / name)
        index = {}
        for r in rows:
            cat = r["occupancy_category"].strip().lower()
            grp = r["section_group"].strip().lower()
            index[(cat, grp)] = r
            index.setdefault(cat, r)
        out[key] = dict(rows=rows, index=index, filename=name)
    return out


def rho_peak_from_density(n, unit_basis):
    """Turn a tabulated default density number N into persons/m2."""
    if unit_basis not in UNIT_BASIS:
        raise SystemExit(f"unknown unit_basis: {unit_basis}")
    return float(n) / 100.0


def build_density(standards_dir, profile_path):
    """Resolve every building type to a standard row and a 24-h mean rho.

    Returns a dict with 'rho_by_type', 'provenance', 'citations', 'tables'
    and 'validation'.  Raises SystemExit if the mapping, the encoded table
    and density_profile.csv disagree: the standards are the source of truth.
    """
    citations = load_citations(standards_dir)
    tables = load_tables(standards_dir)
    mapping = read_csv(Path(standards_dir) / "building_type_map.csv")
    profile = read_csv(profile_path)

    rho_by_type, provenance = {}, []
    max_diff = 0.0
    for m in mapping:
        bt = m["building_type"].strip().lower()
        key = m["citation_key"]
        if key not in citations:
            raise SystemExit(f"{bt}: unknown citation_key {key}")
        if key not in tables:
            raise SystemExit(f"{bt}: no encoded table for {key}")
        tab = tables[key]
        cat = m["occupancy_category"].strip().lower()
        grp = m["section_group"].strip().lower()
        row = tab["index"].get((cat, grp)) or tab["index"].get(cat)
        if row is None:
            raise SystemExit(f"{bt}: '{m['occupancy_category']}' absent from "
                             f"{tab['filename']}")
        n_table = float(row["default_density"])
        n_map = float(m["default_density"])
        if abs(n_table - n_map) > 1e-9:
            raise SystemExit(f"{bt}: default density mismatch "
                             f"{n_map} (map) vs {n_table} ({tab['filename']})")
        peak = rho_peak_from_density(n_table, m["unit_basis"])
        if abs(peak - float(m["rho_peak"])) > 1e-9:
            raise SystemExit(f"{bt}: rho_peak mismatch in building_type_map.csv")

        hours = [p for p in profile
                 if p["building_type"].strip().lower() == bt]
        if not hours:
            raise SystemExit(f"{bt}: no rows in density_profile.csv")
        mean_phi = sum(float(p["phi"]) for p in hours) / len(hours)
        csv_mean_rho = sum(float(p["rho"]) for p in hours) / len(hours)
        expected_rho = peak * mean_phi
        diff = abs(csv_mean_rho - expected_rho)
        max_diff = max(max_diff, diff)
        if diff > 1e-9:
            raise SystemExit(f"{bt}: density_profile.csv mean rho "
                             f"{csv_mean_rho} disagrees with standard-derived "
                             f"{expected_rho}")

        phi_key = (m["phi_citation_key"] or "").strip()
        phi_cite = citations.get(phi_key, {}).get("short", m["phi_source"])
        ref = citations[key]
        provenance.append(dict(
            building_type=bt,
            standard=m["standard"], table=m["table"],
            section_group=m["section_group"],
            occupancy_category=m["occupancy_category"],
            default_density=n_table, density_unit=row["density_unit"],
            unit_basis=m["unit_basis"],
            unit_convention=UNIT_BASIS[m["unit_basis"]],
            rho_peak=peak,
            strict_imperial_rho_peak=n_table / M2_PER_1000_FT2,
            mean_phi=mean_phi, n_hours=len(hours),
            mean_rho=expected_rho, csv_mean_rho=csv_mean_rho,
            profile_abs_diff=diff,
            phi_source=m["phi_source"],
            phi_citation_key=phi_key, phi_short_citation=phi_cite,
            citation_key=key, ref_number=ref["ref_number"],
            short_citation=ref["short"],
            bibtex_key=ref["bibtex_key"],
            in_text_citation=f"[{ref['ref_number']}, {m['table']}, "
                             f"{m['occupancy_category']}]",
            note=m["note"]))
        rho_by_type[bt] = expected_rho

    return dict(rho_by_type=rho_by_type, provenance=provenance,
                citations=citations, tables=tables,
                validation=dict(n_types=len(provenance),
                                max_abs_diff=max_diff,
                                tolerance=1e-9,
                                tables={k: len(v["rows"])
                                        for k, v in tables.items()}))


def _esc(text):
    return re.sub(r"[\\#%&_${}]",
                  lambda m: _BIBTEX_ESCAPE[m.group()], text or "")


def format_bibtex(citations):
    """Generate out/references.bib from the registry."""
    lines = [
        "% Generated by src/standards.py from data/standards/citations.csv.",
        "% Do not edit by hand. UTF-8: compile with biblatex/biber,",
        "% XeLaTeX or LuaLaTeX to preserve diacritics.",
        "",
    ]
    for c in sorted(citations.values(), key=lambda r: int(r["ref_number"])):
        kind = c["entry_type"] or "misc"
        lines.append(f"@{kind}{{{c['bibtex_key']},")
        fields = [("author", c["authors"]), ("title", c["title"])]
        if kind == "article":
            fields += [("journal", c["container"]), ("year", c["year"]),
                       ("volume", c["volume"]), ("pages", c["pages"])]
        else:
            fields += [("howpublished", c["container"] or c["publisher"]),
                       ("year", c["year"])]
        fields += [("doi", c["doi"]), ("url", c["url"])]
        note_bits = [f"Ref. [{c['ref_number']}] in this study; key "
                     f"{c['key']}; role {c['role']}"]
        if c["accessed"]:
            note_bits.append(f"Accessed {c['accessed']}")
        if c["note"]:
            note_bits.append(c["note"])
        fields += [("note", ". ".join(note_bits))]
        for name, value in fields:
            if value:
                lines.append(f"  {name:<12} = {{{_esc(value)}}},")
        lines.append("}")
        lines.append("")
    return "\n".join(lines)


def format_plain(citations, provenance):
    """Generate out/citations.txt: numbered references + a usage trace."""
    lines = [
        "=" * 79,
        "REFERENCE LIST AND CITATION SCHEME",
        "=" * 79,
        "Cite a density as:  [n, Table X, occupancy category]  (see",
        "out/density_provenance.csv for the exact string per building type).",
        "Primary standards [1] and [2] are authoritative; [6] and [7] are",
        "public reproductions used only to transcribe the tables.",
        "BibTeX keys are in out/references.bib.  Registry source of truth:",
        "data/standards/citations.csv.",
        "",
    ]
    for c in sorted(citations.values(), key=lambda r: int(r["ref_number"])):
        bits = [f"[{c['ref_number']}] {c['authors']} ({c['year']}). "
                f"{c['title']}."]
        if c["container"]:
            extra = c["container"]
            if c["volume"]:
                extra += f" {c['volume']}"
            if c["pages"]:
                extra += f", {c['pages']}"
            bits.append(f"    {extra}.")
        if c["publisher"]:
            bits.append(f"    {c['publisher']}.")
        if c["doi"]:
            bits.append(f"    doi:{c['doi']}")
        if c["url"]:
            bits.append(f"    {c['url']}")
        if c["accessed"]:
            bits.append(f"    Accessed {c['accessed']}.")
        bits.append(f"    Role: {c['role']} | Key: {c['key']} | "
                    f"BibTeX: {c['bibtex_key']}")
        uses = [p for p in provenance if p["citation_key"] == c["key"]]
        if uses:
            bits.append("    Used for: " + "; ".join(
                f"{p['building_type']} = {p['default_density']:g} "
                f"({p['occupancy_category']})" for p in uses) + ".")
        phi_uses = [p for p in provenance
                    if p["phi_citation_key"] == c["key"]]
        if phi_uses:
            bits.append("    Justifies phi(h) for: " +
                        ", ".join(p["building_type"] for p in phi_uses) + ".")
        if c["note"]:
            bits.append(f"    Note: {c['note']}")
        lines += bits + [""]
    lines += ["=" * 79, "EXAMPLE IN-TEXT CITATIONS", "=" * 79]
    for p in provenance:
        phi_label = (p["phi_short_citation"] if p["phi_citation_key"]
                     else "flat assumption (no GBOBD schedule)")
        lines.append(
            f"  rho_peak({p['building_type']}) = {p['rho_peak']:g} persons/m2 "
            f"{p['in_text_citation']}; phi from {phi_label}.")
    lines += ["", "=" * 79]
    return "\n".join(lines) + "\n"
