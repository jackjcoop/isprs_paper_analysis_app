"""
Citation and Reference Validator Module
Validates reference format and matches citations to references.
Optimized for ISPRS formatting and PDF extraction artifacts (newlines/hyphenation).
"""

import re
import difflib
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
except ImportError:
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

    # 1. (Author, Year) or (Author & Author, Year) or (Author et al., Year)
    # Handles newlines (\s includes \n)
    PARENTHETICAL_PATTERN = re.compile(
        r'\('                                       # Open paren
        r'(?P<author>[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ\-\s]+)'           # Author name (Starts with Cap, Unicode support)
        r'(?:et al\.?|& [A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ\-]+|and [A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ\-]+)?' # Optional: et al, & Author
        r',?\s+'                                    # Comma/Space
        r'(?P<year>19\d{2}|20\d{2})'                # Year
        r'(?:[a-z])?'                               # Optional year suffix (2020a)
        r'\)',                                      # Close paren
        re.MULTILINE | re.DOTALL
    )

    # 2. Author (Year) - Narrative citation
    NARRATIVE_PATTERN = re.compile(
        r'\b(?P<author>[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ\-]+)'           # Author name (Unicode support)
        r'(?:\s+et al\.?|\s+and\s+[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ\-]+|\s*&\s*[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ\-]+)?' # et al modifiers
        r'\s*\('                                    # Open paren
        r'(?P<year>19\d{2}|20\d{2})'                # Year
        r'(?:[a-z])?'                               # Optional year suffix
        r'\)',                                      # Close paren
        re.MULTILINE | re.DOTALL
    )

    # 3. "Author et al., Year" - Simple format without parentheses
    SIMPLE_ET_AL_PATTERN = re.compile(
        r'\b(?P<author>[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ\-]+)'           # Author name (Unicode support)
        r'\s+et al\.?,\s*'                          # et al., or et al, (period optional)
        r'(?P<year>19\d{2}|20\d{2})'                # Year
        r'(?:[a-z])?'                               # Optional year suffix
    )

    # 3b. "Author et al., (Year)" - et al. with parentheses around year
    SIMPLE_ET_AL_PAREN_PATTERN = re.compile(
        r'\b(?P<author>[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ\-]+)'           # Author name (Unicode support)
        r'\s+et al\.?,\s*\('                        # et al., ( or et al, ( (period optional)
        r'(?P<year>19\d{2}|20\d{2})'                # Year
        r'(?:[a-z])?\)'                             # Optional year suffix and )
    )

    # 4. "Author, Year" - Simple comma-separated format
    SIMPLE_COMMA_PATTERN = re.compile(
        r'\b(?P<author>[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ\-]+)'           # Author name (Unicode support)
        r',\s*'                                     # Comma
        r'(?P<year>19\d{2}|20\d{2})'                # Year
        r'(?:[a-z])?'                               # Optional year suffix
    )

    # 5. "Author and Author, Year" or "Author & Author, Year" - Two authors without parentheses
    SIMPLE_TWO_AUTHORS_PATTERN = re.compile(
        r'\b(?P<author>[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ\-]+)'           # First author (Unicode support)
        r'\s+(?:and|&)\s+[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ\-]+,\s*'      # and/& Second author,
        r'(?P<year>19\d{2}|20\d{2})'                # Year
        r'(?:[a-z])?'                               # Optional year suffix
    )

    def __init__(self):
        self.figure_patterns = [re.compile(r'\bFig(ure)?\.?\s*(\d+)', re.IGNORECASE)]
        self.table_patterns = [re.compile(r'\bTable\s*(\d+)', re.IGNORECASE)]

    def _clean_text(self, text: str) -> str:
        """Normalizes text by removing extra whitespace and newlines."""
        return ' '.join(text.split())

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

        # Extract PERSON entities (author names)
        person_entities = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']

        # Filter out common false positives from "et al."
        filtered_entities = []
        for ent in person_entities:
            # Extract the surname part (last word)
            surname = ent.split()[-1].strip('.,;()')

            # Skip common false positives and very short tokens
            # "al" and "et" are from "et al."
            if surname.lower() not in ['al', 'et'] and len(surname) > 2:
                filtered_entities.append(ent)

        person_entities = filtered_entities

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

        # Use first PERSON entity as primary surname
        primary_surname = ""
        if person_entities:
            # Extract surname (last word of entity, cleaned)
            primary_surname = person_entities[0].split()[-1].strip('.,;()')

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
            surname_pattern = re.compile(r'\b([A-Z][a-zA-Z\'\-]+)\b')
            all_authors = surname_pattern.findall(author_segment)
            # Filter out single-letter initials (e.g., "G", "R", "S")
            all_authors = [a for a in all_authors if len(a) > 1]
            # Remove common words that might be captured
            all_authors = [a for a in all_authors if a.lower() not in ['et', 'al', 'and', 'the', 'in', 'of']]

            # 4. Extract Title and Additional Info
            # Title is first sentence after year (up to first period)
            # Additional info is everything after the title
            additional_info = None
            title_match = re.search(r'^[\W_]*([^.]+?)\.(.*)$', title_segment, re.DOTALL)
            if title_match:
                title = title_match.group(1).strip()
                # Everything after the title period
                remaining = title_match.group(2).strip()
                if remaining:
                    additional_info = remaining
            else:
                # No period found, use first 50 chars as title
                title = title_segment[:50] + "..." if len(title_segment) > 50 else title_segment
                additional_info = None

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

        # Cleanup surname (remove standard words that might get caught)
        if primary_surname:
            # Take FIRST word (first author convention for multi-author citations)
            # e.g., "Trotman and Faraway" -> "Trotman"
            primary_surname = primary_surname.split()[0]
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
            year_suffix=year_suffix
        )

    def match_citation_to_reference(self, citation: ParsedCitation, references: List[ParsedReference]) -> CitationMatch:
        """
        Matches a citation to a reference list entry using Year + Surname Similarity.
        Supports year suffix matching (e.g., 2019a, 2019b).
        """
        if citation.citation_type != 'reference':
            return CitationMatch(citation, None, True, "Non-reference citation")

        if not citation.primary_surname or not citation.year:
            return CitationMatch(citation, None, False, "Could not parse Author/Year from citation")

        best_match = None
        highest_score = 0.0
        best_suffix_match = False  # Track if we found an exact suffix match

        cit_surname_clean = citation.primary_surname.lower()

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

            # 3. Surname Matching
            ref_surname_clean = ref.primary_surname.lower()

            # Exact Match
            if cit_surname_clean == ref_surname_clean:
                score = 1.0
            # Containment (e.g., "Van der Waal" vs "Waal")
            elif cit_surname_clean in ref_surname_clean or ref_surname_clean in cit_surname_clean:
                score = 0.9
            # Fuzzy Match (SequenceMatcher) - handles OCR typos
            else:
                score = difflib.SequenceMatcher(None, cit_surname_clean, ref_surname_clean).ratio()

            # 4. If primary match is weak, check ALL authors in reference
            if score < 0.85 and ref.all_authors:
                for author in ref.all_authors:
                    author_clean = author.lower()
                    if cit_surname_clean == author_clean:
                        score = max(score, 0.95)  # Slightly lower than exact primary
                        break
                    elif cit_surname_clean in author_clean or author_clean in cit_surname_clean:
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

        return CitationMatch(
            citation=citation,
            reference=None,
            matched=False,
            reason=f"No matching reference for {citation.primary_surname} ({citation.year})",
            confidence=highest_score
        )

    def search_main_text_for_citation(
        self,
        reference: ParsedReference,
        main_text_elements: List
    ) -> bool:
        """
        Search in main text for citation of a reference (fallback method).
        Looks for first author name followed by year within reasonable proximity.
        """
        if not reference.primary_surname or not reference.year:
            return False

        first_author = reference.primary_surname.lower().strip('.,;:')
        year = reference.year

        # Search each main text element
        for element in main_text_elements:
            if not hasattr(element, 'text'):
                continue

            text = element.text.lower()

            # Look for author name in text
            if first_author in text:
                # Find all positions of author name
                author_positions = []
                start = 0
                while True:
                    pos = text.find(first_author, start)
                    if pos == -1:
                        break
                    author_positions.append(pos)
                    start = pos + 1

                # For each author position, check if year appears nearby (within 50 chars)
                for author_pos in author_positions:
                    # Check ±50 characters around author name
                    search_start = max(0, author_pos - 50)
                    search_end = min(len(text), author_pos + len(first_author) + 50)
                    nearby_text = text[search_start:search_end]

                    if year in nearby_text:
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
        Combines reference elements that were incorrectly split by Document AI.

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
            r'^[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ\-\'\u2019]+,?\s',  # Starts with capital letter surname
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

            # Also check if element has a year near the start (within first 100 chars)
            # This helps identify actual reference starts vs. continuations
            early_text = text[:100] if len(text) > 100 else text
            has_early_year = bool(re.search(r'\b(19\d{2}|20\d{2})[a-z]?\b', early_text))

            if starts_new_ref and has_early_year:
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

        # Combine adjacent reference elements that were split by Document AI
        combined_refs = self._combine_adjacent_references(ref_elements)

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

        # 2. Parse Citations
        cit_elements = extracted_elements.get('In_Text_Citations_References', [])
        for cit_elem in cit_elements:
            parsed_cit = self.parse_citation(
                cit_elem.text,
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

        # 4. Find Uncited References
        # Create a set of original texts of references that were matched
        cited_ref_texts = set()
        for match in results['citation_matches']:
            if match.matched and match.reference:
                cited_ref_texts.add(match.reference.original_text)

        # Check main text as fallback for uncited references
        main_text_elements = extracted_elements.get('Main_Text', [])

        for parsed_ref in results['references_parsed']:
            if parsed_ref.original_text not in cited_ref_texts:
                # Fallback: search in main text
                found_in_main_text = False
                if main_text_elements:
                    found_in_main_text = self.search_main_text_for_citation(
                        parsed_ref,
                        main_text_elements
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

        # Parse citations
        cited_numbers = set()
        for cit_elem in float_citations:
            # Extract number from citation
            match = re.search(r'\d+', cit_elem.text)
            if match:
                num = match.group(0)
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

                    # Check proximity (±1 page)
                    float_page = floats_by_number[num]['page']
                    citation_page = cit_elem.page

                    if abs(float_page - citation_page) > 1:
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

        # Find uncited floats
        for num, float_info in floats_by_number.items():
            if not float_info['cited']:
                # Include title_bbox for highlighting the title element
                # Also include bbox (Figure_Number/Table_Number bbox) as fallback
                results['uncited_figures' if float_type == 'Figure' else 'uncited_tables'].append({
                    'number': num,
                    'page': float_info['page'],
                    'title_text': float_info.get('title_text', ''),
                    'title_bbox': float_info.get('title_bbox'),
                    'bbox': float_info.get('bbox')  # Fallback if title_bbox not found
                })
