"""
Citation and Reference Validator Module
Validates reference format and matches citations to references.
Optimized for ISPRS formatting and PDF extraction artifacts (newlines/hyphenation).
"""

import re
import difflib
import unicodedata
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

# Optional spaCy import for NER-based citation parsing
try:
    import spacy
    import subprocess
    SPACY_AVAILABLE = True
    try:
        _nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Try to download the model
        try:
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            SPACY_AVAILABLE = False
            _nlp = None
except Exception:
    # Catches ImportError as well as runtime failures from spaCy's C extensions
    # (e.g. numpy ABI mismatches that surface as ValueError on import).
    SPACY_AVAILABLE = False
    _nlp = None


@dataclass
class ParsedCitation:
    """Represents a parsed in-text citation."""
    text: str
    primary_surname: str  # The surname used for matching (e.g., "Smith")
    year: Optional[str]
    page: int
    bbox: Tuple[float, float, float, float]
    citation_type: str = "reference"  # 'reference', 'figure', 'table'
    year_suffix: Optional[str] = None  # 'a', 'b', 'c' for multiple publications same author/year
    format_issues: List[str] = field(default_factory=list)  # spacing/punctuation problems

    @property
    def full_year(self) -> Optional[str]:
        """Return year with suffix if present (e.g., '2019a')."""
        if self.year:
            return f"{self.year}{self.year_suffix}" if self.year_suffix else self.year
        return None


@dataclass
class ParsedReference:
    """Represents a parsed reference from the bibliography."""
    original_text: str
    primary_surname: str  # First author's surname
    all_authors: List[str]
    year: Optional[str]
    title: Optional[str]
    additional_info: Optional[str]  # Journal, volume, pages, DOI, etc.
    is_valid_format: bool
    format_issues: List[str] = field(default_factory=list)
    year_suffix: Optional[str] = None  # 'a', 'b', 'c' for multiple publications same author/year
    page: Optional[int] = None  # Page number for highlighting
    bbox: Optional[Tuple[float, float, float, float]] = None  # Bounding box for highlighting

    @property
    def full_year(self) -> Optional[str]:
        """Return year with suffix if present (e.g., '2019a')."""
        if self.year:
            return f"{self.year}{self.year_suffix}" if self.year_suffix else self.year
        return None


@dataclass
class CitationMatch:
    """Represents a match between citation and reference."""
    citation: ParsedCitation
    reference: Optional[ParsedReference]
    matched: bool
    reason: str = ""
    confidence: float = 0.0


class CitationValidator:
    """Validates citations and references using ISPRS and standard academic patterns."""

    # Regex components - captures optional letter suffix (2019a, 2019b, etc.)
    YEAR_PATTERN = r'\b(19\d{2}|20\d{2})([a-z])?\b'

    # Unicode character classes for author names:
    #   _CAP = uppercase Latin (ASCII + Latin-1 Supplement + Latin Extended-A/B)
    #   _LET = any Latin letter (upper or lower, including extended)
    _CAP = r'A-ZÀ-ÖØ-Þ\u0100-\u024F'
    _LET = r'a-zA-ZÀ-ÖØ-öø-ÿ\u0100-\u024F'

    # 1. (Author, Year) or (Author & Author, Year) or (Author et al., Year)
    # The author group is intentionally ASCII/Latin-only without internal whitespace
    # so that "et al."/"& X"/"and X" cannot be silently absorbed into the surname
    # via greedy backtracking. Multi-word particles ("van der Waals") are handled
    # by the optional lowercase-particle prefix.
    PARENTHETICAL_PATTERN = re.compile(
        r'\('                                       # Open paren
        r'(?P<author>'
        r'(?:(?:van der|van den|van|von|de la|de|del|della|der|den|du|da|dos|el|la|le|ten|ter)\s+)?'
        r'[' + _CAP + r'][' + _LET + r'\-]+'        # Surname (Cap + letters, no spaces)
        r')'
        r'(?:\s+et\.?\s*al\.?'                     # et al / et. al.
        r'|\s*&\s*[' + _CAP + r'][' + _LET + r'\-]+' # & Author
        r'|\s+and\s+[' + _CAP + r'][' + _LET + r'\-]+' # and Author
        r')?'
        r',?\s+'                                    # Comma/Space
        r'(?P<year>19\d{2}|20\d{2})'                # Year
        r'(?:[a-z])?'                               # Optional year suffix (2020a)
        r'\)?',                                     # Close paren (optional — Document AI may truncate)
        re.MULTILINE | re.DOTALL
    )

    # 2. Author (Year) - Narrative citation
    # The particle prefix is included INSIDE the author group so the captured
    # surname matches the reference list (e.g. "von Neumann" sorts/matches
    # against the bibliography entry "von Neumann, J., 1955.").
    NARRATIVE_PATTERN = re.compile(
        r'(?P<author>'
        r'(?:\b(?:van der|van den|van|von|de la|de|del|della|der|den|du|da|dos|el|la|le|ten|ter)\s+)?'
        r'[' + _CAP + r'][' + _LET + r'\-]+'        # Surname (no spaces)
        r')'
        r'(?:\s+et\.?\s*al\.?'
        r'|\s+and\s+[' + _CAP + r'][' + _LET + r'\-]+'
        r'|\s*&\s*[' + _CAP + r'][' + _LET + r'\-]+'
        r')?'
        r'\s*\('                                    # Open paren
        r'(?P<year>19\d{2}|20\d{2})'                # Year
        r'(?:[a-z])?'                               # Optional year suffix
        r'\)?',                                     # Close paren (optional — Document AI may truncate)
        re.MULTILINE | re.DOTALL
    )

    # 3. "Author et al., Year" - Simple format without parentheses
    SIMPLE_ET_AL_PATTERN = re.compile(
        r'\b(?P<author>[' + _CAP + r'][' + _LET + r'\-]+)'           # Author name (Unicode support)
        r'\s+et\.?\s*al\.?,\s*'                       # et al., et. al., or et al, (period optional)
        r'(?P<year>19\d{2}|20\d{2})'                # Year
        r'(?:[a-z])?'                               # Optional year suffix
    )

    # 3b. "Author et al., (Year)" - et al. with parentheses around year
    SIMPLE_ET_AL_PAREN_PATTERN = re.compile(
        r'\b(?P<author>[' + _CAP + r'][' + _LET + r'\-]+)'           # Author name (Unicode support)
        r'\s+et\.?\s*al\.?,\s*\('                    # et al., ( or et. al., ( (period optional)
        r'(?P<year>19\d{2}|20\d{2})'                # Year
        r'(?:[a-z])?\)?'                            # Optional year suffix and ) (optional — Document AI may truncate)
    )

    # 4. "Author, Year" - Simple comma-separated format
    SIMPLE_COMMA_PATTERN = re.compile(
        r'\b(?P<author>[' + _CAP + r'][' + _LET + r'\-]+)'           # Author name (Unicode support)
        r',\s*'                                     # Comma
        r'(?P<year>19\d{2}|20\d{2})'                # Year
        r'(?:[a-z])?'                               # Optional year suffix
    )

    # 5. "Author and Author, Year" or "Author & Author, Year" - Two authors without parentheses
    # Optional lowercase particle prefix on the first surname (e.g. "van der Waals and Smith, 2020").
    SIMPLE_TWO_AUTHORS_PATTERN = re.compile(
        r'(?P<author>'
        r'(?:\b(?:van der|van den|van|von|de la|de|del|della|der|den|du|da|dos|el|la|le|ten|ter)\s+)?'
        r'[' + _CAP + r'][' + _LET + r'\-]+'                           # First author surname
        r')'
        r'\s+(?:and|&)\s+'
        r'(?:\b(?:van der|van den|van|von|de la|de|del|della|der|den|du|da|dos|el|la|le|ten|ter)\s+)?'
        r'[' + _CAP + r'][' + _LET + r'\-]+,\s*'                       # Second author surname
        r'(?P<year>19\d{2}|20\d{2})'                # Year
        r'(?:[a-z])?'                               # Optional year suffix
    )

    # Lenient scanner used to find parenthetical citations Document AI may
    # have missed inside Main_Text. Tolerates "et al.YEAR" with no comma/
    # space, "Smith,YEAR" with no space, etc. Requires either an et-al/&-
    # author group OR an explicit comma/space separator before the year so
    # bare "Smith2020" doesn't false-match.
    SCAN_PARENTHETICAL_PATTERN = re.compile(
        r'\(\s*'
        r'(?P<author>'
        r'(?:(?:van der|van den|van|von|de la|de|del|della|der|den|du|da|dos|el|la|le|ten|ter)\s+)?'
        r'[' + _CAP + r'][' + _LET + r'\-]+'
        r')'
        r'(?:'
        r'\s+et\.?\s*al\.?\s*,?\s*'                                          # et al, with/without comma
        r'|'
        r'\s*(?:and|&)\s*'
        r'(?:(?:van der|van den|van|von|de la|de|del|della|der|den|du|da|dos|el|la|le|ten|ter)\s+)?'
        r'[' + _CAP + r'][' + _LET + r'\-]+\s*,?\s*'                         # and/& Author
        r'|'
        r'[,\s]+'                                                            # plain separator
        r')'
        r'(?P<year>19\d{2}|20\d{2})(?:[a-z])?'
        r'\s*\)',
        re.MULTILINE | re.DOTALL,
    )

    # 6. "Author, n.d." / "Author and Author, n.d." / "Author et al., n.d."
    # Citations with no date — extract the surname; year remains None and
    # surname-only matching handles the lookup.
    NODATE_CITATION_PATTERN = re.compile(
        r'\b(?P<author>'
        r'(?:(?:van der|van den|van|von|de la|de|del|della|der|den|du|da|dos|el|la|le|ten|ter)\s+)?'
        r'[' + _CAP + r'][' + _LET + r'\-]+'
        r')'
        r'(?:\s+et\.?\s*al\.?'
        r'|\s+and\s+[' + _CAP + r'][' + _LET + r'\-]+'
        r'|\s*&\s*[' + _CAP + r'][' + _LET + r'\-]+'
        r')?'
        r',?\s*(?i:n\.?\s*d\.?)(?!\w)'
    )

    # Author-start detector for splitting joined citations. Allows a lowercase
    # nobility/locative particle prefix ("van der Waals", "von Neumann",
    # "de la Rosa") before the capital surname.
    _AUTHOR_START_RE = re.compile(
        r'\b(?:van der|van den|van|von|de la|de|del|della|der|den|du|da|dos|el|la|le|ten|ter)\s+'
        r'[' + _CAP + r']'
        r'|[' + _CAP + r']'
    )

    # Editor-continuation detector. A book-chapter reference body looks like:
    #     ..., in: Awe, O.O., A. Vance, E. (Eds.), Practical ...
    # Inside the editor list the comma+initial+period sequence looks like a
    # reference boundary, so the merge-splitter can produce a false positive.
    # If the first ~100 chars after a candidate split contain "(Eds.)" /
    # "(Ed.)" / "editors", it is a co-editor continuation rather than a new
    # bibliography entry.
    _EDITOR_CONTINUATION_RE = re.compile(
        r'^.{0,100}?(?:\(\s*Eds?\.?\s*\)|\beditors?\b)',
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(self):
        self.figure_patterns = [re.compile(r'\bFig(ure)?\.?\s*(\d+)', re.IGNORECASE)]
        self.table_patterns = [re.compile(r'\bTable\s*(\d+)', re.IGNORECASE)]

    def _clean_text(self, text: str) -> str:
        """Normalizes text by removing extra whitespace and newlines."""
        return ' '.join(text.split())

    # Characters that NFKD decomposition does not reduce to ASCII base letters.
    # These appear in academic author names extracted from PDFs and must map to
    # their plain Latin equivalents for matching.
    _EXTRA_CHAR_MAP = str.maketrans({
        'đ': 'd', 'Đ': 'D',   # d with stroke  (Croatian/Serbian)
        'ð': 'd', 'Ð': 'D',   # eth             (Icelandic)
        'ø': 'o', 'Ø': 'O',   # o with stroke   (Danish/Norwegian)
        'ł': 'l', 'Ł': 'L',   # l with stroke   (Polish)
        'ß': 'ss',             # sharp s         (German)
        'æ': 'ae', 'Æ': 'AE', # ligature ae
        'œ': 'oe', 'Œ': 'OE', # ligature oe
        'þ': 'th', 'Þ': 'Th', # thorn           (Icelandic)
        'ı': 'i',              # dotless i        (Turkish)
    })

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Strip accents and normalize a name for comparison.
        E.g., 'Cedeño' -> 'Cedeno', 'Müller' -> 'Muller', 'Đorđević' -> 'Dordevic'."""
        # First, replace characters that NFKD cannot decompose
        name = name.translate(CitationValidator._EXTRA_CHAR_MAP)
        nfkd = unicodedata.normalize('NFKD', name)
        return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower()

    @staticmethod
    def _looks_like_prose_paragraph(text: str) -> bool:
        """True if a "References" element is actually a body-text paragraph
        (e.g. a conclusion or summary) Document AI swept into the References
        stream. Real references either:
          - have an early comma (after the first word's surname/initial), or
          - lead with two consecutive capitalized words (organisation name).
        Prose paragraphs typically start with a pronoun/article ("This",
        "We", "The", ...) followed by a lowercase word and lack the early
        comma signal.
        """
        if not text:
            return False
        head = text.strip()
        first_two = re.match(r'^(\S+)\s+(\S+)', head)
        if not first_two:
            return False
        w1, w2 = first_two.group(1), first_two.group(2)
        if not w1 or not w2:
            return False
        # Real refs almost always have a comma within the first ~50 chars
        # (after the surname or organisation name). Its absence is suspicious.
        has_early_comma = ',' in head[:50]
        # Two consecutive capitalised tokens signal an organisational author
        # ("Food Systems...", "United Nations..."). Use isupper on the first
        # alphabetic char to allow tokens with leading punctuation.
        def _starts_upper(token: str) -> bool:
            for ch in token:
                if ch.isalpha():
                    return ch.isupper()
            return False
        starts_titlecase_pair = _starts_upper(w1) and _starts_upper(w2)
        if not has_early_comma and not starts_titlecase_pair:
            return True
        return False

    @staticmethod
    def _looks_like_figure_content(text: str) -> bool:
        """True if a "References" element is actually misclassified figure
        content (caption, panel labels, chart data). Document AI sometimes
        sweeps these into the References stream; treating them as real refs
        produces phantom "uncited" entries.

        Signals:
        - Starts with a figure/panel marker: "Figure N.", "(a) Figure", "(d) ...".
        - "Figure N." appears very early in the text (within ~80 chars).
        - The leading text has a high density of standalone numbers and
          square-bracketed units like "[s]" / "[m]" — typical of axis labels.
        """
        if not text:
            return False
        head = text.strip()
        # Panel marker like "(a) Figure 5." or "(d) ..."
        if re.match(r'^\([a-z]\)\s+Figure\s*\d', head, re.IGNORECASE):
            return True
        # "Figure N." right at the start
        if re.match(r'^Figure\s+\d+\.', head, re.IGNORECASE):
            return True
        # "Figure N." appearing in the first ~200 chars indicates figure
        # content rather than a reference whose title happens to mention
        # a figure (real refs lead with author surnames first).
        early = head[:200]
        if re.search(r'\bFigure\s+\d+\.', early, re.IGNORECASE):
            return True
        # Chart-data signature: bracketed units in the leading text
        # (e.g. "[s]", "[m]", "[%]") suggest axis labels, not a reference.
        snippet = head[:300]
        bracket_units = len(re.findall(r'\[[a-zA-Z%]{1,3}\]', snippet))
        if bracket_units >= 2:
            return True
        # Chart-axis signature: many standalone numeric tokens in the
        # leading text (axis tick values like "200 400 600 800"). Real
        # references rarely carry 5+ bare numbers within the first 150
        # chars; skip the check entirely if the text already opens with a
        # plausible author-list pattern ("Surname, I." or
        # "Organisation Name, YEAR") so titles that legitimately contain
        # numerals don't false-match.
        starts_with_author = bool(re.match(
            r'^[A-ZÀ-ɏ][a-zA-ZÀ-ɏ\-\'’ ]+,\s+(?:[A-Z]\.|[A-Z][a-z]+,\s+\d{4})',
            head,
        ))
        if not starts_with_author:
            standalone_digit_groups = len(re.findall(r'(?<!\w)\d+(?!\w)', head[:150]))
            if standalone_digit_groups >= 5:
                return True
        return False

    @staticmethod
    def _normalize_citation_key(text: str) -> str:
        """Build a comparison key for citation dedup. Collapses whitespace,
        strips outer punctuation, lowercases, and squashes spacing defects
        so "Sun et al.2024" and "Sun et al., 2024" map to the same key.
        """
        if not text:
            return ''
        s = ' '.join(text.split()).lower()
        s = re.sub(r'\s*[,;]\s*', ' ', s)
        s = re.sub(r'[\.\(\)]', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    @staticmethod
    def _fix_line_break_hyphens(text: str) -> str:
        """Fix hyphens introduced by line breaks in PDF extraction.
        E.g., 'Shah- mohamadi' -> 'Shahmohamadi'.

        Only merge when the character after the space is lowercase — a
        capitalized word after the hyphen typically signals a new
        sentence or proper noun (e.g. 'data- Caesar' is *not* a
        hyphenation break) and merging across that boundary destroys
        downstream reference parsing.
        """
        return re.sub(r'(\w)- ([a-zà-öø-ÿā-ɏ])', r'\1\2', text)

    # Author-year reference pattern used to find the start of a real
    # reference inside an element that Document AI bundled with non-
    # reference content (figure captions, abstract paragraphs, etc.).
    _REF_INITIAL = r'[A-Z]\.(?:\s*[A-Z]\.)*'
    _REF_SURNAME = r"[A-Z][a-zA-Z\-']+"
    _REF_AUTHOR = _REF_SURNAME + r',\s+' + _REF_INITIAL
    _REF_START_PATTERN = re.compile(
        r'(?<![A-Za-z\-])'
        + _REF_AUTHOR
        + r'(?:,\s+' + _REF_AUTHOR + r')*'
        + r',?\s+(?:19\d{2}|20\d{2})[a-z]?\.'
    )

    @classmethod
    def _strip_non_reference_prefix(cls, text: str) -> str:
        """If the text *clearly* starts with non-reference content
        (figure caption, abstract paragraph, sentence prose) but contains
        a real Surname, I., ..., YYYY. cluster further in, trim the
        leading non-reference content.

        Trim only when the leading text is unambiguously not a reference
        (figure sub-label like '(a)' / '(d)', 'Figure'/'Table' caption,
        or starts with a lowercase letter / digit / opening quote).
        Refs like 'Jefriza, Yusoff, I.M., 2020' where the first author
        has a non-standard format must be left alone.
        """
        stripped = text.lstrip()
        if not stripped:
            return text

        looks_non_reference = bool(
            re.match(r'^\([a-zA-Z]\)\s', stripped)        # "(a) ", "(d) "
            or re.match(r'^(?:Figure|Table|Fig\.|Tab\.)\s', stripped, re.IGNORECASE)
            or re.match(r'^[a-z\d"“‘\'`]', stripped)      # lowercase / digit / quote
        )
        if not looks_non_reference:
            return text

        m = cls._REF_START_PATTERN.search(text)
        if m and m.start() > 0:
            return text[m.start():]
        return text

    @staticmethod
    def _split_on_top_level_semicolons(text: str) -> List[str]:
        """Split on semicolons that are not nested inside parentheses/brackets."""
        parts = []
        depth = 0
        start = 0
        for i, ch in enumerate(text):
            if ch in '([':
                depth += 1
            elif ch in ')]':
                if depth > 0:
                    depth -= 1
            elif ch == ';' and depth == 0:
                parts.append(text[start:i])
                start = i + 1
        parts.append(text[start:])
        return parts

    def _split_multi_citations(self, citation_text: str) -> List[str]:
        """Split a Document AI entity containing multiple citations into individual parts.

        Handles:
        - Semicolon-separated: 'Brauer et al., 2016; Castell et al., 2017'
        - Comma-separated in parens: '(Griffin et al., 2019, Van Geffen et al., 2020)'
        - Joined citations: '(Huang et al., 2023). Vavassori et al. (2024)'
        - Mixed dated/undated: 'Verma and Jana, n.d.; Wang et al., 2022'
        """
        normalized = ' '.join(citation_text.split())
        normalized = self._fix_line_break_hyphens(normalized)

        # Strip balanced outer parens so that internal semicolons aren't
        # treated as "nested" by the depth-aware splitter. A whole-paren
        # multi-citation like "(A et al., 2022; B et al., n.d.; C, 2020)"
        # has every ';' at depth 1 — without stripping, the semicolon
        # split below produces only one part.
        inner = normalized
        if inner.startswith('(') and inner.endswith(')'):
            candidate = inner[1:-1]
            if candidate.count('(') == candidate.count(')'):
                inner = candidate

        # Split on top-level semicolons first. Semicolons unambiguously
        # separate author-year citations and don't normally appear inside a
        # single citation, so this catches cases the year-anchored split
        # below misses (e.g. "n.d." has no year for the regex to anchor on).
        semi_parts = self._split_on_top_level_semicolons(inner)
        if len(semi_parts) > 1:
            results = []
            for part in semi_parts:
                cleaned = part.strip(' ;,.\t')
                if not cleaned:
                    continue
                results.extend(self._split_multi_citations(cleaned))
            return results if results else [normalized]

        # Find all 4-digit years in the text
        year_matches = list(re.finditer(r'\b(19|20)\d{2}[a-z]?\b', normalized))

        if len(year_matches) <= 1:
            # Single or no year — return as-is (single citation)
            return [normalized]

        # Multiple years found — split into individual citations
        # For each year, find the author text preceding it
        citations = []
        for i, ym in enumerate(year_matches):
            if i == 0:
                start = 0
            else:
                # Start after the previous year; find the next capital letter
                # (start of next author name)
                prev_end = year_matches[i - 1].end()
                remaining = normalized[prev_end:]
                author_start = self._AUTHOR_START_RE.search(remaining)
                if author_start:
                    start = prev_end + author_start.start()
                else:
                    start = prev_end

            # End is after this year — include trailing ), ], . that belong to this citation
            end = ym.end()
            while end < len(normalized) and normalized[end] in ')].':
                end += 1

            cit_text = normalized[start:end].strip()
            cit_text = cit_text.strip(';,. \t')
            # Remove balanced outer parens: "(Smith, 2020)" → "Smith, 2020"
            if cit_text.startswith('(') and cit_text.endswith(')'):
                inner = cit_text[1:-1]
                if inner.count('(') == inner.count(')'):
                    cit_text = inner.strip(';,. \t')
            # Strip unmatched leading paren: "(Smith, 2020" → "Smith, 2020"
            if cit_text.startswith('(') and cit_text.count('(') > cit_text.count(')'):
                cit_text = cit_text[1:].strip(';,. \t')
            if cit_text:
                citations.append(cit_text)

        # If a chunk ended up as just a bare year (e.g. "Smith et al.,
        # 2025, 2020" → ["Smith et al., 2025", "2020"]), inherit the
        # previous chunk's author prefix so it parses as a real citation.
        bare_year_re = re.compile(r'^\(?\s*(?:19|20)\d{2}[a-z]?\)?$')
        for i in range(1, len(citations)):
            if bare_year_re.match(citations[i]):
                prev = citations[i - 1]
                prev_year = re.search(r'\b(?:19|20)\d{2}[a-z]?\b', prev)
                if prev_year:
                    author_prefix = prev[:prev_year.start()].rstrip(' ,;.')
                    if author_prefix:
                        citations[i] = f"{author_prefix}, {citations[i]}"

        return citations if citations else [normalized]

    @staticmethod
    def _author_from_ner_entity(ent_text: str) -> str:
        """Extract the first-author surname from a spaCy PERSON entity.

        ISPRS citation format always puts the first author's surname at
        the start (e.g. "Smith, 2020", "Smith et al.", "Smith and Jones").
        spaCy may chunk multi-author or partial citation text into a
        single entity such as "Khalid and Khan" or "Khalid and"; in both
        cases the surname we want is what comes BEFORE the conjunction.
        """
        s = (ent_text or '').strip()
        # Stop at the first conjunction
        s = re.split(r'\s+(?:and|&)\s+', s, maxsplit=1)[0]
        # Trim trailing punctuation/initials commonly stuck to the entity
        return s.strip('.,;()')

    def _parse_citation_with_ner(self, citation_text: str) -> Tuple[str, Optional[str]]:
        """
        Parse citation using spaCy NER to handle varied formats.
        This handles cases like:
        - "Smith described in 2020"
        - "In 2020, Smith found..."
        - "As noted by Smith (2020)"
        - "Smith et al., (2020)"

        Returns:
            Tuple of (primary_surname, year)
        """
        if not SPACY_AVAILABLE or not _nlp:
            return "", None

        doc = _nlp(citation_text)

        # Extract PERSON entities (author names) and clean each one to
        # its first-author surname. Reject candidates whose first token
        # is not capitalised — that filters lowercase artefacts like
        # "and"/"or" that spaCy occasionally hands back when it chunks
        # multi-author citations as one entity.
        skip_tokens = {'al', 'et', 'and', 'or', 'the', 'of', 'in'}
        cleaned_entities: List[str] = []
        for ent in doc.ents:
            if ent.label_ != 'PERSON':
                continue
            cand = self._author_from_ner_entity(ent.text)
            if not cand or len(cand) <= 2:
                continue
            first_token = cand.split()[0]
            if first_token.lower() in skip_tokens:
                continue
            if not first_token[:1].isupper():
                continue
            cleaned_entities.append(cand)

        person_entities = cleaned_entities

        # Extract DATE entities and look for 4-digit years
        # Pattern allows optional year suffix (e.g., 2019a, 2019b) for proper matching
        year = None
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                year_match = re.search(r'\b(19\d{2}|20\d{2})([a-z])?\b', ent.text)
                if year_match:
                    year = year_match.group(1)
                    break

        # If no DATE entity found, search entire text for year
        if not year:
            year_match = re.search(r'\b(19\d{2}|20\d{2})([a-z])?\b', citation_text)
            if year_match:
                year = year_match.group(1)

        # Use first PERSON entity as primary surname (already cleaned by
        # _author_from_ner_entity above — multi-author chunks like
        # "Khalid and Khan" have already been reduced to "Khalid").
        primary_surname = person_entities[0] if person_entities else ""

        return primary_surname, year

    def parse_reference(
        self,
        reference_text: str,
        page: Optional[int] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> ParsedReference:
        """
        Parses a bibliography entry.
        ISPRS Format assumption: Author(s) (Year). Title...
        """
        cleaned_text = self._clean_text(reference_text)
        # Fix line-break hyphens (e.g., "Houwel- ing" -> "Houweling")
        cleaned_text = self._fix_line_break_hyphens(cleaned_text)
        # If Document AI bundled non-reference text (figure captions,
        # abstract paragraphs) ahead of the actual reference, isolate the
        # real Surname, I., ..., YYYY. cluster and drop the leading garbage.
        cleaned_text = self._strip_non_reference_prefix(cleaned_text)
        issues = []
        is_valid = True

        # Check for numbered reference format (incorrect for ISPRS)
        # Pattern matches: [1], [12], (1), (12), 1., 12. at the start of reference
        numbered_pattern = re.match(r'^\s*[\[\(]?\d+[\]\)]?\.?\s+', cleaned_text)
        if numbered_pattern:
            issues.append("Reference uses numbered format (e.g., [1]) instead of required Author-Year format")
            is_valid = False

        # 1. Extract Year (Look for the first 4-digit year bounded by word boundaries)
        # We prioritize the year appearing earlier in the string (after authors)
        year_match = re.search(self.YEAR_PATTERN, cleaned_text)

        # If no year is found, fall back to "n.d." (no date) as the marker
        # that separates the author segment from the title segment.
        nodate_match = None
        if not year_match:
            nodate_match = re.search(r'\bn\.?\s*d\.?(?=\s|[.,;])', cleaned_text, re.IGNORECASE)

        primary_surname = "Unknown"
        all_authors = []
        year = None
        year_suffix = None
        title = None
        additional_info = None

        if year_match:
            year = year_match.group(1)
            year_suffix = year_match.group(2)  # 'a', 'b', 'c' or None

            # Split text into Author segment (pre-year) and Title segment (post-year)
            start_index = year_match.start()
            author_segment = cleaned_text[:start_index].strip()
            title_segment = cleaned_text[year_match.end():].strip()

            # Clean up punctuation at end of author segment (trailing period or comma)
            author_segment = author_segment.rstrip('.,;:')

            # 2. Extract Primary Surname
            # Heuristic: The first word before a comma, or the very first word if no comma
            if ',' in author_segment:
                # E.g., "Smith, J., Jones, B." -> Smith
                primary_surname = author_segment.split(',')[0].strip()
            else:
                # E.g., "Smith et al." -> Smith
                primary_surname = author_segment.split()[0].strip() if author_segment.split() else "Unknown"

            # 3. Extract all authors
            # Pattern matches surnames including those with mid-word capitals (e.g., "McLeod", "O'Brien")
            # Matches: Capital letter followed by any letters, apostrophes, or hyphens
            surname_pattern = re.compile(r'\b([A-Z\u00C0-\u024F][a-zA-Z\u00C0-\u024F\'\-]+)\b')
            all_authors = surname_pattern.findall(author_segment)
            # Filter out single-letter initials (e.g., "G", "R", "S")
            all_authors = [a for a in all_authors if len(a) > 1]
            # Remove common words that might be captured
            all_authors = [a for a in all_authors if a.lower() not in ['et', 'al', 'and', 'the', 'in', 'of']]

            # 4. Extract Title and Additional Info
            # Title is the first sentence after the year. Naive "first period"
            # splitting breaks on abbreviated journal names like
            # "Photogramm. Remote Sens. J." — those internal periods are followed
            # by another capitalised abbreviation, not a new sentence.
            # Heuristic: split only at a period that is followed by whitespace
            # and a Capital-then-lowercase word (real sentence start), or at
            # end-of-string.
            additional_info = None
            cleaned_title_seg = title_segment.lstrip('.,:;\"\' \t')
            sentence_split = re.search(
                r'\.\s+(?=[A-ZÀ-ÖØ-ÞĀ-ɏ][a-zÀ-ɏ])',
                cleaned_title_seg
            )
            if sentence_split:
                title = cleaned_title_seg[:sentence_split.start()].strip()
                remaining = cleaned_title_seg[sentence_split.end():].strip()
                if remaining:
                    additional_info = remaining
            elif cleaned_title_seg.endswith('.'):
                # Whole title segment is one sentence ending in a period
                title = cleaned_title_seg.rstrip('.').strip()
            else:
                title = cleaned_title_seg[:50] + "..." if len(cleaned_title_seg) > 50 else cleaned_title_seg

        elif nodate_match:
            # "n.d." used in place of a year — extract author segment from
            # before the marker so the surname can still be matched.
            start_index = nodate_match.start()
            author_segment = cleaned_text[:start_index].strip().rstrip('.,;:')
            if ',' in author_segment:
                primary_surname = author_segment.split(',')[0].strip()
            elif author_segment.split():
                primary_surname = author_segment.split()[0].strip()

            surname_pattern = re.compile(r'\b([A-ZÀ-ɏ][a-zA-ZÀ-ɏ\'\-]+)\b')
            all_authors = [
                a for a in surname_pattern.findall(author_segment)
                if len(a) > 1 and a.lower() not in ('et', 'al', 'and', 'the', 'in', 'of')
            ]

            title_segment = cleaned_text[nodate_match.end():].lstrip('.,:;\"\' \t')
            sentence_split = re.search(
                r'\.\s+(?=[A-ZÀ-ÖØ-ÞĀ-ɏ][a-zÀ-ɏ])',
                title_segment
            )
            if sentence_split:
                title = title_segment[:sentence_split.start()].strip()
                remaining = title_segment[sentence_split.end():].strip()
                if remaining:
                    additional_info = remaining
            elif title_segment.endswith('.'):
                title = title_segment.rstrip('.').strip()
            else:
                title = title_segment[:50] + "..." if len(title_segment) > 50 else title_segment

        else:
            issues.append("No valid year found")
            is_valid = False

        if not primary_surname or len(primary_surname) < 2:
            issues.append("Could not identify primary author")
            is_valid = False

        return ParsedReference(
            original_text=reference_text,
            primary_surname=primary_surname,
            all_authors=all_authors,
            year=year,
            title=title,
            additional_info=additional_info,
            is_valid_format=is_valid,
            format_issues=issues,
            year_suffix=year_suffix,
            page=page,
            bbox=bbox
        )

    def parse_citation(self, citation_text: str, page: int, bbox: Tuple, citation_type: str = 'reference') -> ParsedCitation:
        """
        Parses an in-text citation string.
        """
        primary_surname = ""
        year = None
        year_suffix = None

        # Handle Figures/Tables immediately
        if citation_type in ['figure', 'table']:
            return ParsedCitation(citation_text, citation_text, None, page, bbox, citation_type)

        # Normalize whitespace (replace newlines with spaces) for pattern matching
        # This handles cases like "Cronk\net al, 2006" where text wraps across lines
        normalized_text = ' '.join(citation_text.split())
        # Fix line-break hyphens (e.g., "Shah- mohamadi" -> "Shahmohamadi")
        normalized_text = self._fix_line_break_hyphens(normalized_text)

        # Repair common spacing defects so the patterns below can still match
        # the underlying citation. Each defect is recorded so the validator
        # can flag it as a citation-format issue separately from matching.
        format_issues: List[str] = []
        # "et al.YEAR" — period of "al." touches the year (no space/comma).
        # Insert ", " so the existing "et al., YEAR" patterns recognize it.
        if re.search(r'\bal\.(?=\d{4}\b)', normalized_text):
            format_issues.append("Missing comma/space between 'al.' and year")
            normalized_text = re.sub(r'(\bal\.)(\d{4}\b)', r'\1, \2', normalized_text)
        # "et al. YEAR" — space but no comma between "al." and year. Insert
        # the comma so the comma-anchored patterns match.
        if re.search(r'\bal\.\s+(?=\d{4}\b)', normalized_text):
            if "Missing comma after 'et al.'" not in format_issues:
                format_issues.append("Missing comma after 'et al.'")
            normalized_text = re.sub(r'(\bal\.)\s+(\d{4}\b)', r'\1, \2', normalized_text)
        # "Surname,YEAR" — comma touches the year directly.
        if re.search(r'[' + self._CAP + self._LET + r'],(?=\d{4}\b)', normalized_text):
            format_issues.append("Missing space after comma before year")
            normalized_text = re.sub(r'(,)(\d{4}\b)', r'\1 \2', normalized_text)
        # Malformed year: a 5+ digit number adjacent to a year-shaped prefix
        # (e.g. "20122" — typo for 2012). The year regex below would silently
        # take the leading "2012" and match a real reference; flag instead so
        # the citation surfaces as an orphan with a clear reason.
        malformed_year_match = re.search(r'\b((?:19|20)\d{3,})\b', normalized_text)
        if malformed_year_match:
            format_issues.append(
                f"Malformed year '{malformed_year_match.group(1)}' "
                f"(expected 4 digits)"
            )

        # Try NER first (handles all variable formats like "Smith described in 2020")
        if SPACY_AVAILABLE and _nlp:
            primary_surname, year = self._parse_citation_with_ner(normalized_text)

        # Fall back to regex patterns if NER didn't find both author and year
        if not primary_surname or not year:
            # Try patterns in order of specificity
            # 1. Check Parenthetical: (Smith, 2020)
            p_match = self.PARENTHETICAL_PATTERN.search(normalized_text)
            if p_match:
                primary_surname = p_match.group('author').strip()
                year = p_match.group('year')
            else:
                # 2. Check Narrative: Smith (2020)
                n_match = self.NARRATIVE_PATTERN.search(normalized_text)
                if n_match:
                    primary_surname = n_match.group('author').strip()
                    year = n_match.group('year')
                else:
                    # 3. Check Simple et al. with parens: Boyle et al., (2024)
                    et_al_paren_match = self.SIMPLE_ET_AL_PAREN_PATTERN.search(normalized_text)
                    if et_al_paren_match:
                        primary_surname = et_al_paren_match.group('author').strip()
                        year = et_al_paren_match.group('year')
                    else:
                        # 4. Check Simple et al.: Boyle et al., 2024
                        et_al_match = self.SIMPLE_ET_AL_PATTERN.search(normalized_text)
                        if et_al_match:
                            primary_surname = et_al_match.group('author').strip()
                            year = et_al_match.group('year')
                        else:
                            # 5. Check Two authors: Smith and Jones, 2020 or Smith & Jones, 2020
                            two_authors_match = self.SIMPLE_TWO_AUTHORS_PATTERN.search(normalized_text)
                            if two_authors_match:
                                primary_surname = two_authors_match.group('author').strip()
                                year = two_authors_match.group('year')
                            else:
                                # 6. Check Simple comma: Smith, 2020
                                comma_match = self.SIMPLE_COMMA_PATTERN.search(normalized_text)
                                if comma_match:
                                    primary_surname = comma_match.group('author').strip()
                                    year = comma_match.group('year')

        # No-date fallback: "Author, n.d." style citations have no year, so
        # the year-based patterns above can't anchor on them. Year stays None
        # and surname-only matching takes over.
        if not primary_surname:
            nd_match = self.NODATE_CITATION_PATTERN.search(normalized_text)
            if nd_match:
                primary_surname = nd_match.group('author').strip()

        # Surname-only fallback: if no patterns matched, try to extract just the author name
        # Handles truncated citations like "Berends et al.," with no year
        if not primary_surname:
            surname_only_match = re.match(
                r'\b(?P<author>[' + self._CAP + r'][' + self._LET + r'\-]{2,})'
                r'(?:\s+et\.?\s*al\.?)?\s*[,;.]',
                normalized_text
            )
            if surname_only_match:
                primary_surname = surname_only_match.group('author').strip()

        # Cleanup surname. Take the FIRST word as the surname for legacy
        # multi-word artefacts (e.g. "Trotman and Faraway" -> "Trotman"),
        # but preserve particle-prefixed surnames intact ("van der Waals"
        # must not collapse to "van"). The particle list mirrors the one
        # used in PARENTHETICAL_PATTERN / NARRATIVE_PATTERN.
        if primary_surname:
            parts = primary_surname.split()
            _PARTICLES = {
                'van', 'von', 'de', 'del', 'della', 'der', 'den',
                'du', 'da', 'dos', 'el', 'la', 'le', 'ten', 'ter',
            }
            if parts and parts[0].lower() not in _PARTICLES:
                primary_surname = parts[0]
            primary_surname = primary_surname.strip('.,;()')

        # Extract year suffix if present (e.g., 2019a, 2019b)
        if year:
            suffix_match = re.search(rf'{year}([a-z])\b', normalized_text)
            if suffix_match:
                year_suffix = suffix_match.group(1)

        return ParsedCitation(
            text=citation_text,
            primary_surname=primary_surname,
            year=year,
            page=page,
            bbox=bbox,
            citation_type=citation_type,
            year_suffix=year_suffix,
            format_issues=format_issues,
        )

    def match_citation_to_reference(self, citation: ParsedCitation, references: List[ParsedReference]) -> CitationMatch:
        """
        Matches a citation to a reference list entry using Year + Surname Similarity.
        Supports year suffix matching (e.g., 2019a, 2019b).
        """
        if citation.citation_type != 'reference':
            return CitationMatch(citation, None, True, "Non-reference citation")

        # Malformed year (e.g. "20122") — refuse to match. The year regex
        # would silently truncate to a real-looking year and then fuzzy-match
        # an adjacent reference, masking the typo from the reviewer.
        for issue in (citation.format_issues or []):
            if issue.startswith("Malformed year"):
                return CitationMatch(citation, None, False, issue)

        if not citation.primary_surname:
            return CitationMatch(citation, None, False, "Could not parse author from citation")
        if not citation.year:
            return self._match_citation_surname_only(citation, references)

        best_match = None
        highest_score = 0.0
        best_suffix_match = False  # Track if we found an exact suffix match

        cit_surname_clean = citation.primary_surname.lower()
        cit_surname_norm = self._normalize_name(citation.primary_surname)

        for ref in references:
            if not ref.year or not ref.primary_surname:
                continue

            # 1. Year Check (Hard Filter) - base year must match
            if ref.year != citation.year:
                continue

            # 2. Year Suffix Check (Soft preference for exact match)
            suffix_match = False
            if citation.year_suffix and ref.year_suffix:
                # Both have suffix - must match exactly for priority
                suffix_match = (citation.year_suffix == ref.year_suffix)
            elif not citation.year_suffix and not ref.year_suffix:
                # Neither has suffix - good match
                suffix_match = True
            # else: one has suffix, other doesn't - can still match but lower priority

            # 3. Surname Matching (with accent normalization)
            ref_surname_clean = ref.primary_surname.lower()
            ref_surname_norm = self._normalize_name(ref.primary_surname)

            # Exact Match (case-insensitive)
            if cit_surname_clean == ref_surname_clean:
                score = 1.0
            # Exact match after accent normalization (Cedeño == Cedeno)
            elif cit_surname_norm == ref_surname_norm:
                score = 1.0
            # Containment (e.g., "Van der Waal" vs "Waal")
            elif cit_surname_clean in ref_surname_clean or ref_surname_clean in cit_surname_clean:
                score = 0.9
            elif cit_surname_norm in ref_surname_norm or ref_surname_norm in cit_surname_norm:
                score = 0.9
            # Fuzzy Match (SequenceMatcher) - handles OCR typos
            else:
                # Use accent-normalized forms for fuzzy matching
                score = difflib.SequenceMatcher(None, cit_surname_norm, ref_surname_norm).ratio()

            # 4. If primary match is weak, check ALL authors in reference.
            # Skip short tokens (≤3 chars) and Dutch/Romance/etc. particles
            # so e.g. "De" / "Van" / "Le" don't substring-match into common
            # surnames like "Mulder", "Vance", "Lee".
            _PARTICLE_TOKENS = {
                'van', 'von', 'de', 'del', 'della', 'der', 'den', 'du',
                'da', 'dos', 'el', 'la', 'le', 'ten', 'ter',
            }
            if score < 0.85 and ref.all_authors:
                for author in ref.all_authors:
                    author_clean = author.lower()
                    if len(author_clean) <= 3 or author_clean in _PARTICLE_TOKENS:
                        continue
                    author_norm = self._normalize_name(author)
                    if cit_surname_clean == author_clean or cit_surname_norm == author_norm:
                        score = max(score, 0.95)  # Slightly lower than exact primary
                        break
                    elif (cit_surname_clean in author_clean or author_clean in cit_surname_clean
                          or cit_surname_norm in author_norm or author_norm in cit_surname_norm):
                        score = max(score, 0.85)
                        break

            # 5. Update best match - prefer suffix matches when scores are close
            if suffix_match:
                # Suffix matches get priority
                if score > highest_score or (not best_suffix_match and score >= highest_score - 0.05):
                    highest_score = score
                    best_match = ref
                    best_suffix_match = True
            elif not best_suffix_match and score > highest_score:
                # Only update if we haven't found a suffix match yet
                highest_score = score
                best_match = ref

        # Threshold for acceptance
        if highest_score >= 0.85:
            # Include suffix in match reason if present
            year_display = best_match.full_year or best_match.year
            return CitationMatch(
                citation=citation,
                reference=best_match,
                matched=True,
                reason=f"Matched: {best_match.primary_surname} ({year_display})",
                confidence=highest_score
            )

        # Second pass: ±1 year tolerance with near-exact surname match
        # Handles preprint-vs-published year discrepancies and author typos
        best_fuzzy_match = None
        best_fuzzy_score = 0.0

        for ref in references:
            if not ref.year or not ref.primary_surname:
                continue

            try:
                year_diff = abs(int(ref.year) - int(citation.year))
            except (ValueError, TypeError):
                continue

            if year_diff != 1:
                continue

            # Require near-exact surname match (>=0.95) for year-tolerant matching
            ref_surname_norm = self._normalize_name(ref.primary_surname)

            if cit_surname_clean == ref.primary_surname.lower() or cit_surname_norm == ref_surname_norm:
                adj_score = 1.0 * 0.95  # Penalty for year mismatch
            elif cit_surname_norm in ref_surname_norm or ref_surname_norm in cit_surname_norm:
                adj_score = 0.9 * 0.95
            else:
                sim = difflib.SequenceMatcher(None, cit_surname_norm, ref_surname_norm).ratio()
                if sim < 0.95:
                    continue
                adj_score = sim * 0.95

            if adj_score > best_fuzzy_score:
                best_fuzzy_score = adj_score
                best_fuzzy_match = ref

        if best_fuzzy_score >= 0.85 and best_fuzzy_match:
            year_display = best_fuzzy_match.full_year or best_fuzzy_match.year
            return CitationMatch(
                citation=citation,
                reference=best_fuzzy_match,
                matched=True,
                reason=f"Matched: {best_fuzzy_match.primary_surname} ({year_display}) (year ±1)",
                confidence=best_fuzzy_score
            )

        return CitationMatch(
            citation=citation,
            reference=None,
            matched=False,
            reason=f"No matching reference for {citation.primary_surname} ({citation.year})",
            confidence=highest_score
        )

    def _match_citation_surname_only(
        self,
        citation: ParsedCitation,
        references: List[ParsedReference]
    ) -> CitationMatch:
        """Match a citation that has a surname but no year.

        This handles truncated citations like "Berends et al.," where
        Document AI failed to capture the year. Requires a very high
        surname similarity threshold (>=0.95) and applies a 0.9x
        confidence penalty since there is no year confirmation.
        """
        cit_surname_norm = self._normalize_name(citation.primary_surname)
        cit_surname_clean = citation.primary_surname.lower()

        best_match = None
        best_score = 0.0

        for ref in references:
            if not ref.primary_surname:
                continue

            ref_surname_clean = ref.primary_surname.lower()
            ref_surname_norm = self._normalize_name(ref.primary_surname)

            if cit_surname_clean == ref_surname_clean or cit_surname_norm == ref_surname_norm:
                score = 1.0
            elif cit_surname_norm in ref_surname_norm or ref_surname_norm in cit_surname_norm:
                score = 0.95
            else:
                score = difflib.SequenceMatcher(None, cit_surname_norm, ref_surname_norm).ratio()

            if score >= 0.95 and score > best_score:
                best_score = score
                best_match = ref

        if best_match:
            confidence = best_score * 0.9  # Penalty for no year confirmation
            year_display = best_match.full_year or best_match.year or '?'
            return CitationMatch(
                citation=citation,
                reference=best_match,
                matched=True,
                reason=f"Matched (surname only): {best_match.primary_surname} ({year_display})",
                confidence=confidence
            )

        return CitationMatch(
            citation=citation,
            reference=None,
            matched=False,
            reason=f"No matching reference for {citation.primary_surname} (no year)",
            confidence=0.0
        )

    def search_main_text_for_citation(
        self,
        reference: ParsedReference,
        main_text_elements: List
    ) -> bool:
        """
        Search in main text for citation of a reference (fallback method).
        Looks for first author name (word-boundary match) followed by year
        in a citation context (parenthetical, after comma, or after "et al.").
        """
        if not reference.primary_surname or not reference.year:
            return False

        first_author = reference.primary_surname.strip('.,;:')
        year = reference.year

        # Word-boundary regex for author name (case-insensitive)
        author_pattern = re.compile(r'\b' + re.escape(first_author) + r'\b', re.IGNORECASE)

        # Citation-context regex for year — require it to appear in a citation pattern,
        # not just as bare digits (avoids matching DOIs, page numbers, dates).
        # The trailing-comma form catches multi-citation lists like
        # "(Getis and Ord, 1992, Ord and Getis, 1995, Kumagai, 2011, ...)" —
        # each comma-separated entry's year is followed by another comma.
        citation_year_pattern = re.compile(
            r'(?:'
            r'\(\s*' + re.escape(year) + r'[a-z]?\s*\)?'             # (2011) or (2011 — truncated
            r'|'
            r',\s*' + re.escape(year) + r'[a-z]?\s*[,);\]]'          # , 2011, or , 2011) or , 2011;
            r'|'
            r'et\.?\s+al\.?,?\s*\(?\s*' + re.escape(year) + r'[a-z]?\s*\)?'  # et al., 2011 or et. al. (2011)
            r')'
        )

        # Search each main text element
        for element in main_text_elements:
            if not hasattr(element, 'text'):
                continue

            # Normalize text: fix line-break hyphens so "Shah- mohamadi" -> "Shahmohamadi"
            text = self._fix_line_break_hyphens(element.text)
            # Collapse runs of whitespace (incl. newlines) to single spaces.
            # Table cells often split a citation across lines, e.g.
            # "Cheng et al.\n, 2022)" — the citation_year_pattern below expects
            # commas adjacent to the year, so we flatten first.
            text = re.sub(r'\s+', ' ', text)

            # Look for author name with word boundaries
            for author_match in author_pattern.finditer(text):
                author_pos = author_match.start()

                # Check ±50 characters around author name for year in citation context
                search_start = max(0, author_pos - 50)
                search_end = min(len(text), author_pos + len(first_author) + 50)
                nearby_text = text[search_start:search_end]

                if citation_year_pattern.search(nearby_text):
                    return True

        # Pass 2: Cross-element search
        # "Al-Manasir" at end of one element, "(2015)" at start of the next.
        # Build per-page concatenated text from all Main_Text elements,
        # sorted in reading order (column then y-position) so text at the
        # end of one column is adjacent to text at the start of the next.
        from collections import defaultdict
        pages_elems: Dict[int, list] = defaultdict(list)
        for element in main_text_elements:
            if not hasattr(element, 'text') or not hasattr(element, 'page'):
                continue
            pages_elems[element.page].append(element)

        for page_num, elems in pages_elems.items():
            # Sort by reading order: left column top-to-bottom, then right column
            def _reading_order(e):
                if hasattr(e, 'bbox') and e.bbox:
                    x_center = (e.bbox[0] + e.bbox[2]) / 2
                    col = 0 if x_center < 300 else 1  # approximate midpoint
                    return (col, e.bbox[1])
                return (0, 0)
            elems.sort(key=_reading_order)
            texts = [self._fix_line_break_hyphens(e.text) for e in elems]
            concatenated = ' '.join(texts)
            for author_match in author_pattern.finditer(concatenated):
                author_pos = author_match.start()
                search_start = max(0, author_pos - 50)
                search_end = min(len(concatenated), author_pos + len(first_author) + 50)
                nearby_text = concatenated[search_start:search_end]
                if citation_year_pattern.search(nearby_text):
                    return True

        return False

    def search_main_text_for_float(
        self,
        float_type: str,
        float_number: str,
        main_text_elements: List
    ) -> bool:
        """Search Main_Text elements for a figure/table citation.

        Looks for patterns like "Figure 5", "Fig. 5", "Table 3" as well as
        compound references like "Tables 2 and 3", "Figures 1, 2, and 3",
        or "Figs. 1-3" in the body text.

        Args:
            float_type: 'Figure' or 'Table'
            float_number: The number to search for (e.g. '5')
            main_text_elements: List of Main_Text elements with .text attribute

        Returns:
            True if the float is cited in the main text
        """
        num_esc = re.escape(float_number)

        # Direct pattern: "Figure 5", "Fig. 5", "Table 3"
        if float_type == 'Figure':
            direct = re.compile(
                r'\bFig(?:ure)?s?\.?\s*' + num_esc + r'\b', re.IGNORECASE
            )
            # Compound: "Figures 2 and 3", "Figs. 1, 2, and 3", "Figures 1-3"
            # Prefix captures "Figures " or "Figs. " followed by a number list
            compound_prefix = r'\bFig(?:ure)?s?\.?\s*'
        else:
            direct = re.compile(
                r'\bTables?\s*' + num_esc + r'\b', re.IGNORECASE
            )
            compound_prefix = r'\bTables?\s*'

        # Compound pattern: prefix + sequence of digits separated by
        # commas / "and" / "&" / en-dash / hyphen, containing our number
        compound = re.compile(
            compound_prefix + r'(\d+(?:\s*(?:,\s*|\s+and\s+|\s*&\s*|\s*[-–]\s*)\d+)+)',
            re.IGNORECASE
        )

        target = int(float_number)

        for element in main_text_elements:
            if not hasattr(element, 'text'):
                continue
            # Normalize whitespace (newlines/tabs -> spaces) and undo
            # line-break hyphenation. Without this, citations like
            # "Fig-\nure 3" or "Fig- ure 3" — which Document AI does not flag
            # as In_Text_Citations_Figures but are still present in the
            # surrounding Main_Text — would slip past the regex below and
            # the figure would be falsely reported as uncited.
            text = ' '.join(element.text.split())
            text = self._fix_line_break_hyphens(text)

            # Fast path: direct mention
            if direct.search(text):
                return True

            # Compound path: "Tables 2 and 3" → check if target is in the list
            for m in compound.finditer(text):
                nums_text = m.group(1)
                # Extract individual numbers
                found_nums = [int(n) for n in re.findall(r'\d+', nums_text)]
                # Expand ranges: "1-3" → [1, 2, 3]
                if re.search(r'\d+\s*[-–]\s*\d+', nums_text):
                    expanded = set()
                    for rm in re.finditer(r'(\d+)\s*[-–]\s*(\d+)', nums_text):
                        lo, hi = int(rm.group(1)), int(rm.group(2))
                        expanded.update(range(lo, hi + 1))
                    # Also include any standalone numbers
                    expanded.update(found_nums)
                    found_nums = list(expanded)
                if target in found_nums:
                    return True

        return False

    def _is_ref_text_complete(self, text: str) -> bool:
        """
        Check if reference text appears complete (not needing continuation).
        A complete reference typically ends with:
        - Page numbers (45-67, or 123-456)
        - DOI (doi:... or https://doi.org/...)
        - URL (http/https)
        - Year followed by period
        - "pp." indicating page count

        Args:
            text: The reference text to check

        Returns:
            True if text appears to be a complete reference
        """
        text = text.strip()
        if not text:
            return False

        # Check for common reference ending patterns
        ends_with_pages = bool(re.search(r'\d+[-–]\d+\.?\s*$', text))
        ends_with_doi = bool(re.search(r'doi[:\s][^\s]+\.?\s*$', text, re.IGNORECASE))
        ends_with_url = bool(re.search(r'https?://[^\s]+\.?\s*$', text))
        ends_with_year_period = bool(re.search(r'\b(19|20)\d{2}[a-z]?\.\s*$', text))
        ends_with_pp = bool(re.search(r'\d+\s*pp\.?\s*$', text))

        return any([ends_with_pages, ends_with_doi, ends_with_url,
                    ends_with_year_period, ends_with_pp])

    def _combine_adjacent_references(self, ref_elements: List, page_width: float = 595) -> List:
        """
        NOTE: No longer called from validate_citations_and_references().
        Reference merging is handled upstream by main.py:
        - _merge_partial_references() for Reference_Partial elements
        - _merge_column_spanning_references() for References at column/page boundaries
        - _detect_references_in_gaps() for missed references
        Retained for potential future use.

        Original purpose: Combines reference elements that were incorrectly split
        by Document AI within the same column (intra-column fragment merging).

        Document AI sometimes splits a single reference into multiple text blocks.
        This method detects fragments (text that doesn't start with an author-year pattern)
        and merges them with the preceding reference.

        Uses column-aware sorting to handle two-column ISPRS layouts correctly:
        - Sort by (page, column, y_position)
        - Column 0 = left (x_center < mid_page), Column 1 = right
        - This ensures proper reading order: left column top-to-bottom, then right column

        Args:
            ref_elements: List of reference elements with text, page, bbox attributes
            page_width: Page width in points (default 595 for A4)

        Returns:
            List of combined reference elements (as simple objects with text, page, bbox)
        """
        if not ref_elements:
            return []

        # Column-aware sorting: (page, column, y_position)
        # This matches the reading order for two-column layouts
        mid_page = page_width / 2

        def column_sort_key(r):
            page = r.page
            if r.bbox:
                x_center = (r.bbox[0] + r.bbox[2]) / 2
                y_pos = r.bbox[1]
            else:
                x_center = 0
                y_pos = 0
            # Column 0 = left, Column 1 = right
            column = 0 if x_center < mid_page else 1
            return (page, column, y_pos)

        sorted_refs = sorted(ref_elements, key=column_sort_key)

        # Pattern to detect start of a new reference: Author names followed by year
        # Typical format: "Surname, I., Surname2, J. (2020)" or "Surname, I., (2020)"
        # Key: must start with capital letter and have a year within first ~100 chars
        # Includes extended Latin characters for European names (Pöntinen, Müller, etc.)
        new_ref_pattern = re.compile(
            r'^[A-Z\u00C0-\u024F][a-zA-Z\u00C0-\u024F\-\'\u2019]+,?\s',  # Starts with capital letter surname
            re.UNICODE
        )

        # Create a simple class to hold combined references
        class CombinedRef:
            def __init__(self, text, page, bbox):
                self.text = text
                self.page = page
                self.bbox = bbox

        combined = []
        current_ref = None

        for ref_elem in sorted_refs:
            text = ref_elem.text.strip()

            # Skip empty elements
            if not text:
                continue

            # Check if this looks like a new reference (starts with author name)
            starts_new_ref = bool(new_ref_pattern.match(text))

            # Check if text contains a year anywhere (not limited to first N chars,
            # because references with many authors can have the year 150+ chars in)
            has_year = bool(re.search(r'\b(19\d{2}|20\d{2})[a-z]?\b', text))

            if starts_new_ref and has_year:
                # This is a new reference - save current and start fresh
                if current_ref:
                    combined.append(CombinedRef(
                        current_ref['text'],
                        current_ref['page'],
                        current_ref['bbox']
                    ))

                current_ref = {
                    'text': text,
                    'page': ref_elem.page,
                    'bbox': ref_elem.bbox
                }
            else:
                # This might be a fragment - but only merge if previous ref is incomplete
                if current_ref:
                    prev_is_complete = self._is_ref_text_complete(current_ref['text'])
                    if not prev_is_complete:
                        # Previous reference is incomplete - merge this fragment
                        current_ref['text'] = current_ref['text'].rstrip() + ' ' + text.lstrip()

                        # Expand bounding box to include this element
                        if ref_elem.bbox and current_ref['bbox']:
                            x0 = min(current_ref['bbox'][0], ref_elem.bbox[0])
                            y0 = min(current_ref['bbox'][1], ref_elem.bbox[1])
                            x1 = max(current_ref['bbox'][2], ref_elem.bbox[2])
                            y1 = max(current_ref['bbox'][3], ref_elem.bbox[3])
                            current_ref['bbox'] = (x0, y0, x1, y1)
                    else:
                        # Previous reference is complete - start new reference
                        # even if this text doesn't look like a typical reference start
                        combined.append(CombinedRef(
                            current_ref['text'],
                            current_ref['page'],
                            current_ref['bbox']
                        ))
                        current_ref = {
                            'text': text,
                            'page': ref_elem.page,
                            'bbox': ref_elem.bbox
                        }
                else:
                    # No current reference - this fragment is the start (unusual but handle it)
                    current_ref = {
                        'text': text,
                        'page': ref_elem.page,
                        'bbox': ref_elem.bbox
                    }

        # Don't forget the last reference
        if current_ref:
            combined.append(CombinedRef(
                current_ref['text'],
                current_ref['page'],
                current_ref['bbox']
            ))

        return combined

    def _split_merged_references(self, ref_elements, page_width=595):
        """Split reference elements that contain multiple references
        merged by Document AI into a single text block.

        Document AI sometimes returns two or more bibliography entries as a
        single text entity.  ``_combine_adjacent_references`` only *merges*
        fragments — it never *splits* them.  This method detects the boundary
        between merged entries (sentence-ending period followed by a new
        ``Surname, I.`` pattern) and splits them into separate elements so
        that ``parse_reference`` can extract each author-year independently.
        """
        # Pattern: sentence-ending period followed by "Surname, I."
        split_boundary = re.compile(
            r'(?<=\.)\s+'
            r'(?=[A-Z\u00C0-\u024F][a-zA-Z\u00C0-\u024F\-\'\u2019]+,\s+[A-Z]\.)'
        )

        class SplitRef:
            def __init__(self, text, page, bbox):
                self.text = text
                self.page = page
                self.bbox = bbox

        result = []
        for elem in ref_elements:
            cleaned = self._clean_text(elem.text)
            cleaned = self._fix_line_break_hyphens(cleaned)

            # Quick check: multiple years?
            years = list(re.finditer(r'\b(19\d{2}|20\d{2})[a-z]?\b', cleaned))
            if len(years) <= 1:
                result.append(elem)
                continue

            # Find split points
            splits = list(split_boundary.finditer(cleaned))
            if not splits:
                result.append(elem)
                continue

            # Validate each candidate split:
            #   (a) the chunk BEFORE the split must contain a year (otherwise
            #       we are splitting mid-author-list before the year of the
            #       *current* reference — e.g. "..., Eglin, T.. Hedlund, K.,"
            #       where the typo'd "T.." is followed by a co-author),
            #   (b) the chunk AFTER the split must contain a year within
            #       300 chars,
            #   (c) the chunk AFTER must not look like a co-editor
            #       continuation ('.. (Eds.), Title ..'). A book-chapter ref
            #       like 'in: Awe, O.O., A. Vance, E. (Eds.), Practical ...'
            #       has the comma+initial+period split pattern internally,
            #       but the text after the split is just the next editor's
            #       name followed by '(Eds.)' / '(Ed.)' / 'editors,'.
            valid_positions = []
            last_valid_start = 0
            for m in splits:
                before = cleaned[last_valid_start:m.start()]
                if not re.search(r'\b(19\d{2}|20\d{2})\b', before):
                    continue
                after = cleaned[m.end():m.end() + 300]
                if not re.search(r'\b(19\d{2}|20\d{2})\b', after):
                    continue
                if self._EDITOR_CONTINUATION_RE.match(after):
                    continue
                valid_positions.append(m.end())
                last_valid_start = m.end()

            if not valid_positions:
                result.append(elem)
                continue

            # Split and create new elements
            parts = []
            start = 0
            for pos in valid_positions:
                parts.append(cleaned[start:pos].rstrip())
                start = pos
            parts.append(cleaned[start:].strip())

            page = elem.page if hasattr(elem, 'page') else None
            bbox = elem.bbox if hasattr(elem, 'bbox') else None
            for part in parts:
                if part:
                    result.append(SplitRef(part, page, bbox))

        return result

    def validate_citations_and_references(
        self,
        extracted_elements: Dict[str, List]
    ) -> Dict[str, Any]:
        """
        Main entry point. Validates all references and citations.
        """
        results = {
            'references_parsed': [],
            'citations_parsed': [],
            'citation_matches': [],
            'orphan_citations': [],
            'uncited_references': [],
            'invalid_references': [],
        }

        # 1. Parse References - first combine split references
        ref_elements = extracted_elements.get('References', [])

        # Reference merging is handled upstream by main.py:
        # - _merge_partial_references() for Reference_Partial elements
        # - _merge_column_spanning_references() for References at column/page boundaries
        combined_refs = ref_elements
        # Filter out elements Document AI mis-classified as References — most
        # commonly figure captions, chart data blocks, panel labels, or body
        # prose (conclusion paragraphs, etc.) that got swept into the
        # References stream. These would otherwise be parsed as bogus
        # references and surface as "uncited".
        def _is_misclassified(r) -> bool:
            t = getattr(r, 'text', '') or ''
            return self._looks_like_figure_content(t) or self._looks_like_prose_paragraph(t)
        combined_refs = [r for r in combined_refs if not _is_misclassified(r)]
        # Split references that Document AI merged into a single text block
        combined_refs = self._split_merged_references(combined_refs)

        for ref_elem in combined_refs:
            # Pass page and bbox for highlighting out-of-order references
            ref_page = ref_elem.page if hasattr(ref_elem, 'page') else None
            ref_bbox = ref_elem.bbox if hasattr(ref_elem, 'bbox') else None
            parsed_ref = self.parse_reference(ref_elem.text, page=ref_page, bbox=ref_bbox)
            results['references_parsed'].append(parsed_ref)

            if not parsed_ref.is_valid_format:
                results['invalid_references'].append({
                    'text': parsed_ref.original_text,
                    'issues': parsed_ref.format_issues,
                    'page': parsed_ref.page,
                    'bbox': parsed_ref.bbox
                })

        # 2. Parse Citations — split multi-citation entities first
        cit_elements = extracted_elements.get('In_Text_Citations_References', [])
        for cit_elem in cit_elements:
            # Split entities that contain multiple citations
            # e.g., "(Griffin et al., 2019, Van Geffen et al., 2020)" -> 2 citations
            individual_citations = self._split_multi_citations(cit_elem.text)

            for cit_text in individual_citations:
                parsed_cit = self.parse_citation(
                    cit_text,
                    cit_elem.page,
                    cit_elem.bbox,
                    'reference'
                )
                results['citations_parsed'].append(parsed_cit)

                # 3. Match
                match = self.match_citation_to_reference(parsed_cit, results['references_parsed'])
                results['citation_matches'].append(match)

                if not match.matched:
                    results['orphan_citations'].append({
                        'text': parsed_cit.text,
                        'page': parsed_cit.page,
                        'reason': match.reason
                    })

        # 2b. Scan Main_Text for citation patterns Document AI may have
        # missed. The extractor sometimes drops citations whose spacing/
        # punctuation is malformed (e.g. "(Sun et al.2024)"). We pick those
        # up here so they appear in the report instead of vanishing.
        existing_cit_texts: set = set()
        for parsed_cit in results['citations_parsed']:
            existing_cit_texts.add(self._normalize_citation_key(parsed_cit.text))

        for elem in extracted_elements.get('Main_Text', []):
            elem_text = getattr(elem, 'text', '') or ''
            elem_page = getattr(elem, 'page', None)
            elem_bbox = getattr(elem, 'bbox', None)
            if not elem_text or elem_page is None:
                continue
            for m in self.SCAN_PARENTHETICAL_PATTERN.finditer(elem_text):
                full_text = m.group(0)
                key = self._normalize_citation_key(full_text)
                if key in existing_cit_texts:
                    continue
                existing_cit_texts.add(key)

                parsed_cit = self.parse_citation(
                    full_text, elem_page, elem_bbox, 'reference',
                )
                results['citations_parsed'].append(parsed_cit)
                match = self.match_citation_to_reference(
                    parsed_cit, results['references_parsed'],
                )
                results['citation_matches'].append(match)
                if not match.matched:
                    results['orphan_citations'].append({
                        'text': parsed_cit.text,
                        'page': parsed_cit.page,
                        'reason': match.reason,
                    })

        # 4. Find Uncited References
        # Create a set of original texts of references that were matched
        cited_ref_texts = set()
        for match in results['citation_matches']:
            if match.matched and match.reference:
                cited_ref_texts.add(match.reference.original_text)

        # Check main text as fallback for uncited references. Also scan
        # Table content — comparison/literature tables often cite references
        # inline (e.g. "(Chen et al., 2018)") and those don't appear in
        # In_Text_Citations_References extraction.
        fallback_elements = (
            extracted_elements.get('Main_Text', [])
            + extracted_elements.get('Table', [])
        )

        for parsed_ref in results['references_parsed']:
            if parsed_ref.original_text not in cited_ref_texts:
                # Fallback: search in main text and tables
                found_in_main_text = False
                if fallback_elements:
                    found_in_main_text = self.search_main_text_for_citation(
                        parsed_ref,
                        fallback_elements
                    )

                if not found_in_main_text:
                    results['uncited_references'].append({
                        'authors': parsed_ref.primary_surname,
                        'year': parsed_ref.year,
                        'text_preview': parsed_ref.original_text[:100] + "..." if len(parsed_ref.original_text) > 100 else parsed_ref.original_text,
                        'page': parsed_ref.page,
                        'bbox': parsed_ref.bbox
                    })

        return results

    def validate_figure_table_citations(
        self,
        extracted_elements: Dict[str, List]
    ) -> Dict[str, Any]:
        """
        Validate figure and table citations.
        """
        results = {
            'figure_validation': {
                'figures': [],
                'citations': [],
                'uncited_figures': [],
                'orphan_citations': [],
                'out_of_proximity': []
            },
            'table_validation': {
                'tables': [],
                'citations': [],
                'uncited_tables': [],
                'orphan_citations': [],
                'out_of_proximity': []
            }
        }

        # Fix cross-type misclassifications before validation
        self._fix_cross_type_float_citations(extracted_elements)

        # Validate figures
        self._validate_floats(
            extracted_elements,
            'Figure',
            results['figure_validation']
        )

        # Validate tables
        self._validate_floats(
            extracted_elements,
            'Table',
            results['table_validation']
        )

        return results

    def _fix_cross_type_float_citations(self, extracted_elements: Dict[str, List]):
        """Move misclassified float citations to the correct list.

        Document AI sometimes places "Figure 8" text under
        In_Text_Citations_Tables (or vice-versa). This method inspects the
        text of each citation element and moves it to the correct list when
        the text clearly belongs to the other type.
        """
        fig_key = 'In_Text_Citations_Figures'
        tab_key = 'In_Text_Citations_Tables'

        fig_list = extracted_elements.get(fig_key, [])
        tab_list = extracted_elements.get(tab_key, [])

        fig_re = re.compile(r'\bfig(?:ure)?\.?\b', re.IGNORECASE)
        tab_re = re.compile(r'\btable\b', re.IGNORECASE)

        # Check figure citations for table text
        move_to_tab = []
        keep_in_fig = []
        for elem in fig_list:
            text = elem.text if hasattr(elem, 'text') else ''
            has_fig = bool(fig_re.search(text))
            has_tab = bool(tab_re.search(text))
            if has_tab and not has_fig:
                move_to_tab.append(elem)
            else:
                keep_in_fig.append(elem)

        # Check table citations for figure text
        move_to_fig = []
        keep_in_tab = []
        for elem in tab_list:
            text = elem.text if hasattr(elem, 'text') else ''
            has_fig = bool(fig_re.search(text))
            has_tab = bool(tab_re.search(text))
            if has_fig and not has_tab:
                move_to_fig.append(elem)
            else:
                keep_in_tab.append(elem)

        # Apply moves
        if move_to_tab or move_to_fig:
            extracted_elements[fig_key] = keep_in_fig + move_to_fig
            extracted_elements[tab_key] = keep_in_tab + move_to_tab

    def _validate_floats(
        self,
        extracted_elements: Dict[str, List],
        float_type: str,  # 'Figure' or 'Table'
        results: Dict
    ):
        """Helper method to validate figures or tables."""
        # Get float elements
        number_key = f"{float_type}_Number"
        citation_key = f"In_Text_Citations_{float_type}s"

        float_numbers = extracted_elements.get(number_key, [])
        float_citations = extracted_elements.get(citation_key, [])

        # Extract float numbers and pages
        floats_by_number = {}
        for elem in float_numbers:
            # Convert BoundingBox object to tuple if needed
            elem_bbox = elem.bbox
            if hasattr(elem_bbox, 'x0'):
                elem_bbox = (elem_bbox.x0, elem_bbox.y0, elem_bbox.x1, elem_bbox.y1)
            # Extract number from text
            match = re.search(r'\d+', elem.text)
            if match:
                num = match.group(0)
                floats_by_number[num] = {
                    'number': num,
                    'page': elem.page,
                    'text': elem.text,
                    'bbox': elem_bbox,
                    'title_text': None,
                    'title_bbox': None,
                    'cited': False,
                    'citations': []
                }

                # Note: We don't try to match Figure_Title to Figure_Number because
                # multiple figures on the same page would get the wrong title.
                # The validator uses elem.bbox (Figure_Number location) directly.

                results['figures' if float_type == 'Figure' else 'tables'].append({
                    'number': num,
                    'page': elem.page
                })

        # Fallback: if Document AI extracted more {Figure,Table}_Title
        # elements than {Figure,Table}_Number elements, infer the missing
        # numbers from the titles' reading-order position. ISPRS papers
        # number figures/tables sequentially (1..N) in the order they
        # appear, so the i-th title (1-indexed) corresponds to item #i.
        title_key = f"{float_type}_Title"
        title_elements = extracted_elements.get(title_key, []) or []

        def _title_reading_order(e):
            page = getattr(e, 'page', 0) or 0
            bb = getattr(e, 'bbox', None)
            y = bb[1] if bb else 0
            return (page, y)

        sorted_titles = sorted(title_elements, key=_title_reading_order)

        # Each title that's already paired with an extracted Figure/Table_Number
        # (same page, overlapping y-row) shouldn't drive an inference — the
        # number is already in floats_by_number. We mark those titles as
        # consumed and only walk the remaining unpaired titles.
        def _title_paired_with_existing(title) -> bool:
            tb = getattr(title, 'bbox', None)
            if hasattr(tb, 'x0'):
                tb = (tb.x0, tb.y0, tb.x1, tb.y1)
            tp = getattr(title, 'page', None)
            if not tb or tp is None:
                return False
            ty0, ty1 = tb[1], tb[3]
            for entry in floats_by_number.values():
                if entry.get('page') != tp:
                    continue
                eb = entry.get('bbox')
                if not eb:
                    continue
                ey0, ey1 = eb[1], eb[3]
                # Same caption row: title overlaps or sits just below the number
                if ty0 <= ey1 + 15 and ty1 >= ey0 - 15:
                    return True
            return False

        unpaired_titles = [t for t in sorted_titles if not _title_paired_with_existing(t)]

        # Fill missing numbers (lowest-first) using unpaired titles only.
        for title in unpaired_titles:
            # Pick the smallest sequential number not already present
            i = 1
            while str(i) in floats_by_number:
                i += 1
            str_i = str(i)
            title_bbox = getattr(title, 'bbox', None)
            if hasattr(title_bbox, 'x0'):
                title_bbox = (title_bbox.x0, title_bbox.y0, title_bbox.x1, title_bbox.y1)
            title_page = getattr(title, 'page', 0)
            floats_by_number[str_i] = {
                'number': str_i,
                'page': title_page,
                'text': f'(inferred from {title_key})',
                'bbox': title_bbox,
                'title_text': getattr(title, 'text', '') or '',
                'title_bbox': title_bbox,
                'cited': False,
                'citations': []
            }
            results['figures' if float_type == 'Figure' else 'tables'].append({
                'number': str_i,
                'page': title_page
            })

        # Parse citations — extract ALL numbers from compound references
        # e.g. "Tables 2 and 3" or "Figures 1, 2, and 3" should cite every number
        cited_numbers = set()
        for cit_elem in float_citations:
            # Extract all numbers from citation text
            all_nums = re.findall(r'\d+', cit_elem.text)
            if not all_nums:
                continue

            for num in all_nums:
                cited_numbers.add(num)

                citation_info = {
                    'number': num,
                    'page': cit_elem.page,
                    'text': cit_elem.text
                }
                results['citations'].append(citation_info)

                # Check if float exists
                if num in floats_by_number:
                    floats_by_number[num]['cited'] = True
                    floats_by_number[num]['citations'].append(cit_elem.page)

                    # Check proximity (±2 pages)
                    float_page = floats_by_number[num]['page']
                    citation_page = cit_elem.page

                    if abs(float_page - citation_page) > 2:
                        results['out_of_proximity'].append({
                            'number': num,
                            'float_page': float_page,
                            'citation_page': citation_page,
                            'distance': abs(float_page - citation_page)
                        })
                else:
                    # Orphan citation - include bbox for highlighting
                    results['orphan_citations'].append({
                        'number': num,
                        'page': cit_elem.page,
                        'text': cit_elem.text,
                        'bbox': cit_elem.bbox
                    })

        # Find uncited floats — try main-text fallback before reporting
        main_text_elements = extracted_elements.get('Main_Text', [])
        for num, float_info in floats_by_number.items():
            if not float_info['cited']:
                # Fallback: search Main_Text for "Figure N" / "Table N"
                if main_text_elements and self.search_main_text_for_float(
                    float_type, num, main_text_elements
                ):
                    float_info['cited'] = True
                    continue

                # Include title_bbox for highlighting the title element
                # Also include bbox (Figure_Number/Table_Number bbox) as fallback
                results['uncited_figures' if float_type == 'Figure' else 'uncited_tables'].append({
                    'number': num,
                    'page': float_info['page'],
                    'title_text': float_info.get('title_text', ''),
                    'title_bbox': float_info.get('title_bbox'),
                    'bbox': float_info.get('bbox')  # Fallback if title_bbox not found
                })
