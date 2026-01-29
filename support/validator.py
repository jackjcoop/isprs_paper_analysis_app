"""
Compliance Validator Module
Validates document structure against schema requirements.
"""

import re
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    """Validation result severity levels."""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    SUCCESS = "SUCCESS"


# ISPRS Requirements Constants
FONT_SIZE_TOLERANCE = 0.5  # Allow ±0.5pt for PDF rendering differences

# Known Times New Roman equivalents (PostScript / embedded font names)
TIMES_FONT_ALIASES = [
    'times',
    'nimbusromno9l',   # URW++ Times Roman clone
    'nimbusroman',     # URW++ alternate naming
    'tinos',           # Google/Croscore Times equivalent
    'thorndale',       # LibreOffice Times equivalent
    'timesnewroman',   # No-space variant
    'timesnewromps',   # PostScript variant
]

FONT_REQUIREMENTS = {
    'Title': {'size': 12, 'bold': True, 'centered': True},
    'Authors': {'size': 9, 'bold': False, 'centered': True},
    'Affiliations': {'size': 9, 'bold': False, 'centered': True},
    'Keywords': {'size': 9, 'bold': False, 'centered': False},  # "Keywords:" label should be bold
    'Abstract': {'size': 9, 'bold': False, 'centered': False},  # "Abstract" label should be bold
    'Main_Text': {'size': 9, 'bold': False, 'centered': False},
    'Headings': {'size': 9, 'bold': True, 'centered': True},
    'Sub_Headings': {'size': 9, 'bold': True, 'centered': False},
    'Sub_sub_Headings': {'size': 9, 'bold': True, 'centered': False},
    'References': {'size': 9, 'bold': False, 'centered': False},
    'Figure_Title': {'size': 9, 'bold': False, 'centered': True},
    'Table_Title': {'size': 9, 'bold': False, 'centered': True},
}

PAGE_REQUIREMENTS = {
    'size': (595, 842),      # A4 in points (210mm × 297mm)
    'margins': {
        'top': 71,           # 25mm
        'bottom': 71,        # 25mm
        'left': 57,          # 20mm
        'right': 57,         # 20mm
    },
    'min_pages': 6,
    'max_pages': 8,
}

ABSTRACT_REQUIREMENTS = {
    'min_words': 100,
    'max_words': 250,
}

# Tolerances for validation
FONT_SIZE_TOLERANCE = 0.5  # ±0.5pt
PAGE_SIZE_TOLERANCE = 5    # ±5pt
MARGIN_TOLERANCE = 10      # ±10pt
CENTER_TOLERANCE = 20      # ±20pt for centering


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    passed: bool
    severity: Severity
    message: str
    details: str = ""
    # Element references for PDF annotation highlighting with per-instance messages:
    # [(page, (x0, y0, x1, y1), instance_message), ...]
    # The instance_message provides specific details for each highlighted element
    element_refs: Optional[List[Tuple[int, Tuple[float, float, float, float], str]]] = None


class ComplianceValidator:
    """Validates document compliance against schema requirements."""

    # Schema requirements
    REQUIRED_ONCE = [
        "Abstract",
        "Affiliations",
        "Authors",
        "Keywords",
        "Title"
    ]

    REQUIRED_MULTIPLE = [
        "Headings",
        "In_Text_Citations_References",
        "References"
    ]

    OPTIONAL_MULTIPLE = [
        "Abstract_title",
        "Equation",
        "Equation_Number",
        "Figure_Number",
        "Figure_Title",
        "In_Text_Citations_Figures",
        "In_Text_Citations_Tables",
        "Keywords_title",
        "Main_Text",
        "Sub_Headings",
        "Sub_sub_Headings",
        "Table",
        "Table_Number",
        "Table_Title",
        "Reference_Partial"
    ]

    def __init__(self):
        """Initialize validator."""
        self.results: List[ValidationResult] = []

    def validate(
        self,
        extracted_elements: Dict[str, List],
        labels_verified: Dict,
        citation_results: Optional[Dict] = None,
        figure_table_results: Optional[Dict] = None,
        anonymization_result: Optional[Any] = None,
        page_count: Optional[int] = None,
        page_dimensions: Optional[Tuple[float, float]] = None
    ) -> Tuple[bool, List[ValidationResult]]:
        """
        Validate extracted elements against schema requirements.

        Args:
            extracted_elements: Dictionary of element type to list of elements
            labels_verified: Dictionary of label verification results
            citation_results: Optional citation validation results
            figure_table_results: Optional figure/table validation results
            anonymization_result: Optional anonymization check result
            page_count: Optional number of pages in document
            page_dimensions: Optional tuple of (width, height) in points

        Returns:
            Tuple of (overall_pass, list_of_validation_results)
        """
        self.results = []

        # Check required once elements
        self._check_required_once(extracted_elements)

        # Check required multiple elements
        self._check_required_multiple(extracted_elements)

        # Check optional elements (informational)
        self._check_optional_elements(extracted_elements)

        # Check label verification
        self._check_labels(labels_verified)

        # Check citation validation (if provided)
        if citation_results:
            self._check_citations(citation_results)

        # Check figure/table validation (if provided)
        if figure_table_results:
            self._check_figures_tables(figure_table_results)

        # Check anonymization (if provided)
        if anonymization_result:
            self._check_anonymization(anonymization_result, extracted_elements)

        # Check ISPRS formatting (if page info provided)
        if page_count is not None and page_dimensions is not None:
            self._check_isprs_formatting(extracted_elements, page_count, page_dimensions, citation_results)

        # Determine overall pass/fail
        has_errors = any(
            r.severity == Severity.WARNING and not r.passed
            for r in self.results
        )

        overall_pass = not has_errors

        return overall_pass, self.results

    def _check_required_once(self, extracted_elements: Dict[str, List]):
        """Check elements that are required exactly once."""
        for element_type in self.REQUIRED_ONCE:
            elements = extracted_elements.get(element_type, [])
            count = len(elements)

            if count == 0:
                self.results.append(ValidationResult(
                    check_name=f"Required Element: {element_type}",
                    passed=False,
                    severity=Severity.WARNING,
                    message=f"Missing required element: {element_type}",
                    details=f"Expected exactly 1, found 0"
                ))
            elif count == 1:
                self.results.append(ValidationResult(
                    check_name=f"Required Element: {element_type}",
                    passed=True,
                    severity=Severity.SUCCESS,
                    message=f"Found required element: {element_type}",
                    details=f"Found 1 occurrence"
                ))
            else:
                self.results.append(ValidationResult(
                    check_name=f"Required Element: {element_type}",
                    passed=False,
                    severity=Severity.WARNING,
                    message=f"Multiple occurrences of {element_type}",
                    details=f"Expected 1, found {count}"
                ))

    def _check_required_multiple(self, extracted_elements: Dict[str, List]):
        """Check elements that are required at least once."""
        for element_type in self.REQUIRED_MULTIPLE:
            elements = extracted_elements.get(element_type, [])
            count = len(elements)

            if count == 0:
                self.results.append(ValidationResult(
                    check_name=f"Required Multiple: {element_type}",
                    passed=False,
                    severity=Severity.WARNING,
                    message=f"Missing required element: {element_type}",
                    details=f"Expected at least 1, found 0"
                ))
            else:
                self.results.append(ValidationResult(
                    check_name=f"Required Multiple: {element_type}",
                    passed=True,
                    severity=Severity.SUCCESS,
                    message=f"Found required element: {element_type}",
                    details=f"Found {count} occurrence(s)"
                ))

    def _check_optional_elements(self, extracted_elements: Dict[str, List]):
        """Check optional elements (informational only)."""
        for element_type in self.OPTIONAL_MULTIPLE:
            elements = extracted_elements.get(element_type, [])
            count = len(elements)

            if count > 0:
                self.results.append(ValidationResult(
                    check_name=f"Optional Element: {element_type}",
                    passed=True,
                    severity=Severity.INFO,
                    message=f"Found optional element: {element_type}",
                    details=f"Found {count} occurrence(s)"
                ))

    def _check_labels(self, labels_verified: Dict):
        """Check label verification results."""
        # Get element references for highlighting
        keywords_element_ref = labels_verified.get("Keywords_element_ref")
        abstract_element_ref = labels_verified.get("Abstract_element_ref")

        # Check Keywords label format warning (highlight the keywords region)
        keywords_format_warning = labels_verified.get("Keywords_format_warning")
        if keywords_format_warning:
            element_refs = None
            if keywords_element_ref:
                page, bbox = keywords_element_ref
                instance_msg = f"Incorrect label format - {keywords_format_warning}"
                element_refs = [(page, bbox, instance_msg)]

            self.results.append(ValidationResult(
                check_name="Keywords Label Format",
                passed=False,
                severity=Severity.WARNING,
                message="Keywords label uses incorrect formatting",
                details=keywords_format_warning,
                element_refs=element_refs
            ))

        # Check Abstract label format warning (highlight the abstract region)
        abstract_format_warning = labels_verified.get("Abstract_format_warning")
        if abstract_format_warning:
            element_refs = None
            if abstract_element_ref:
                page, bbox = abstract_element_ref
                instance_msg = f"Incorrect label format - {abstract_format_warning}"
                element_refs = [(page, bbox, instance_msg)]

            self.results.append(ValidationResult(
                check_name="Abstract Label Format",
                passed=False,
                severity=Severity.WARNING,
                message="Abstract label uses incorrect formatting",
                details=abstract_format_warning,
                element_refs=element_refs
            ))

        # Check each label (skip internal keys)
        for label, verified in labels_verified.items():
            if label.endswith("_format_warning") or label.endswith("_element_ref"):
                continue  # Skip internal entries

            if verified:
                self.results.append(ValidationResult(
                    check_name=f"Label Verification: {label}",
                    passed=True,
                    severity=Severity.SUCCESS,
                    message=f"Label '{label}' verified",
                    details=f"Label found in expected location"
                ))
            else:
                # For labels not found, highlight the element region if available
                element_refs = None
                if label == "Keywords" and keywords_element_ref:
                    page, bbox = keywords_element_ref
                    instance_msg = "Keywords label not found in expected location"
                    element_refs = [(page, bbox, instance_msg)]
                elif label == "Abstract" and abstract_element_ref:
                    page, bbox = abstract_element_ref
                    instance_msg = "Abstract label not found in expected location"
                    element_refs = [(page, bbox, instance_msg)]

                self.results.append(ValidationResult(
                    check_name=f"Label Verification: {label}",
                    passed=False,
                    severity=Severity.WARNING,
                    message=f"Label '{label}' not verified",
                    details=f"Label not found in expected location",
                    element_refs=element_refs
                ))

    def _check_citations(self, citation_results: Dict):
        """Check citation and reference validation results."""
        # Check for invalid references
        invalid_refs = citation_results.get('invalid_references', [])
        if invalid_refs:
            # Build element_refs for per-instance PDF annotations
            element_refs = []
            for ref in invalid_refs:
                if ref.get('page') is not None and ref.get('bbox'):
                    instance_msg = f"Invalid format: {', '.join(ref['issues'])}"
                    element_refs.append((ref['page'], ref['bbox'], instance_msg))

            # Collect unique issues across all invalid references
            all_issues = set()
            for ref in invalid_refs:
                all_issues.update(ref['issues'])

            self.results.append(ValidationResult(
                check_name="Reference Format Validation",
                passed=False,
                severity=Severity.WARNING,
                message=f"Found {len(invalid_refs)} reference(s) with invalid format",
                details=f"Issues: {'; '.join(sorted(all_issues))}",
                element_refs=element_refs if element_refs else None
            ))

        # Check for orphan citations
        orphan_citations = citation_results.get('orphan_citations', [])
        if orphan_citations:
            # Build element_refs from parsed citations for PDF annotation
            element_refs = []
            citations_parsed = citation_results.get('citations_parsed', [])
            orphan_texts = set(c['text'] for c in orphan_citations)

            for parsed_cit in citations_parsed:
                if parsed_cit.text in orphan_texts:
                    instance_msg = f"Citation '{parsed_cit.text}' has no matching reference in bibliography"
                    element_refs.append((parsed_cit.page, parsed_cit.bbox, instance_msg))

            orphan_list = '; '.join(c['text'] for c in orphan_citations)
            self.results.append(ValidationResult(
                check_name="Orphaned Citations",
                passed=False,
                severity=Severity.WARNING,
                message=f"Found {len(orphan_citations)} citation(s) without matching reference",
                details=f"Unmatched citations: {orphan_list}",
                element_refs=element_refs if element_refs else None
            ))
        else:
            self.results.append(ValidationResult(
                check_name="Orphaned Citations",
                passed=True,
                severity=Severity.SUCCESS,
                message="All citations have matching references",
                details=f"Validated {len(citation_results.get('citations_parsed', []))} citations"
            ))

        # Check for uncited references
        uncited_refs = citation_results.get('uncited_references', [])
        if uncited_refs:
            # Build element_refs for PDF annotation
            element_refs = []
            for ref in uncited_refs:
                if ref.get('bbox') and ref.get('page') is not None:
                    instance_msg = f"Reference '{ref.get('authors', 'Unknown')} ({ref.get('year', '?')})' has no in-text citation"
                    element_refs.append((ref['page'], ref['bbox'], instance_msg))

            def _short_ref(r):
                author = r.get('authors', 'Unknown')
                # Truncate long author fields (e.g., from numbered-format references)
                if len(author) > 25:
                    author = author[:25].rsplit(' ', 1)[0] + '...'
                year = str(r.get('year', '?'))
                return f"{author} ({year})"

            uncited_list = '; '.join(_short_ref(r) for r in uncited_refs)
            self.results.append(ValidationResult(
                check_name="Uncited Sources",
                passed=False,
                severity=Severity.WARNING,
                message=f"Found {len(uncited_refs)} reference(s) without citations",
                details=f"Uncited: {uncited_list}",
                element_refs=element_refs if element_refs else None
            ))
        else:
            self.results.append(ValidationResult(
                check_name="Uncited Sources",
                passed=True,
                severity=Severity.SUCCESS,
                message="All references are cited in text",
                details=f"Validated {len(citation_results.get('references_parsed', []))} references"
            ))

    def _check_figures_tables(self, figure_table_results: Dict):
        """Check figure and table citation validation results."""
        # Check figures
        fig_validation = figure_table_results.get('figure_validation', {})
        self._check_float_type(fig_validation, 'Figure')

        # Check tables
        table_validation = figure_table_results.get('table_validation', {})
        self._check_float_type(table_validation, 'Table')

    def _check_float_type(self, validation: Dict, float_type: str):
        """Helper to check figure or table validation."""
        # Check for uncited floats
        uncited_key = f'uncited_{float_type.lower()}s'
        uncited = validation.get(uncited_key, [])
        if uncited:
            # Build element_refs pointing to the TITLE element for each uncited float
            # Falls back to the Number element bbox if title_bbox not found
            element_refs = []
            for f in uncited:
                highlight_bbox = f.get('title_bbox') or f.get('bbox')
                # Handle case where bbox might be a BoundingBox object instead of tuple
                if highlight_bbox and hasattr(highlight_bbox, 'x0'):
                    highlight_bbox = (highlight_bbox.x0, highlight_bbox.y0, highlight_bbox.x1, highlight_bbox.y1)
                if highlight_bbox:
                    instance_msg = f"{float_type} {f['number']} - no in-text citation found"
                    element_refs.append((f['page'], highlight_bbox, instance_msg))

            uncited_list = ', '.join(f"{float_type} {f['number']}" for f in uncited)
            self.results.append(ValidationResult(
                check_name=f"{float_type} Citation Coverage",
                passed=False,
                severity=Severity.WARNING,
                message=f"Found {len(uncited)} {float_type.lower()}(s) without in-text citation",
                details=f"Uncited: {uncited_list}",
                element_refs=element_refs if element_refs else None
            ))
        elif validation.get(float_type.lower() + 's', []):
            self.results.append(ValidationResult(
                check_name=f"{float_type} Citation Coverage",
                passed=True,
                severity=Severity.SUCCESS,
                message=f"All {float_type.lower()}s are cited",
                details=f"Validated {len(validation.get(float_type.lower() + 's', []))} {float_type.lower()}(s)"
            ))

        # Check for orphan citations
        orphan = validation.get('orphan_citations', [])
        if orphan:
            # Build element_refs pointing to the CITATION in the text
            element_refs = []
            for f in orphan:
                if f.get('bbox'):
                    instance_msg = f"'{f['text']}' - no corresponding {float_type.lower()} found"
                    element_refs.append((f['page'], f['bbox'], instance_msg))

            # Deduplicate by number (same figure/table cited multiple times)
            unique_numbers = sorted(set(f['number'] for f in orphan), key=str)
            orphan_list = ', '.join(f"{float_type} {n}" for n in unique_numbers)
            self.results.append(ValidationResult(
                check_name=f"{float_type} Citation Validity",
                passed=False,
                severity=Severity.WARNING,
                message=f"Found {len(unique_numbers)} citation(s) to non-existent {float_type.lower()}s",
                details=f"No matching {float_type.lower()}: {orphan_list}",
                element_refs=element_refs if element_refs else None
            ))

        # Check for proximity violations
        out_of_proximity = validation.get('out_of_proximity', [])
        if out_of_proximity:
            examples = [
                f"{float_type} {item['number']} on page {item['float_page']}, cited on page {item['citation_page']}"
                for item in out_of_proximity[:2]
            ]
            self.results.append(ValidationResult(
                check_name=f"{float_type} Citation Proximity",
                passed=False,
                severity=Severity.WARNING,
                message=f"Found {len(out_of_proximity)} {float_type.lower()} citation(s) >1 page away",
                details=f"Examples: {', '.join(examples)}"
            ))
        elif validation.get('citations', []):
            self.results.append(ValidationResult(
                check_name=f"{float_type} Citation Proximity",
                passed=True,
                severity=Severity.SUCCESS,
                message=f"All {float_type.lower()} citations within ±1 page",
                details=f"Validated {len(validation.get('citations', []))} citation(s)"
            ))

    def _check_anonymization(self, anonymization_result, extracted_elements: Dict[str, List]):
        """Check anonymization validation results."""
        if anonymization_result.is_anonymous:
            self.results.append(ValidationResult(
                check_name="Anonymization Check",
                passed=True,
                severity=Severity.SUCCESS,
                message="Document appears to be properly anonymized",
                details=f"No PERSON entities found in {', '.join(anonymization_result.sections_checked)}"
            ))
        else:
            # Get PERSON entities for the message
            person_entities = [e for e in anonymization_result.entities_found if e.label == 'PERSON']
            entities_preview = ', '.join(e.text for e in person_entities[:5])
            if len(person_entities) > 5:
                entities_preview += '...'

            # Build element_refs to highlight Authors/Affiliations sections
            element_refs = []
            for section_name in anonymization_result.sections_checked:
                elements = extracted_elements.get(section_name, [])
                for elem in elements:
                    if hasattr(elem, 'page') and hasattr(elem, 'bbox'):
                        bbox = elem.bbox
                        if hasattr(bbox, 'x0'):
                            bbox = (bbox.x0, bbox.y0, bbox.x1, bbox.y1)
                        instance_msg = f"{section_name} section contains identifying information: {entities_preview}"
                        element_refs.append((elem.page, bbox, instance_msg))

            self.results.append(ValidationResult(
                check_name="Anonymization Check",
                passed=False,
                severity=Severity.WARNING,
                message=f"Document contains {anonymization_result.total_person_entities} PERSON entity/entities",
                details=f"Found in {', '.join(anonymization_result.sections_checked)}: {entities_preview}",
                element_refs=element_refs if element_refs else None
            ))

        # Also report ORG and GPE entities as info
        if anonymization_result.total_org_entities > 0:
            self.results.append(ValidationResult(
                check_name="Organization Entities (Info)",
                passed=True,
                severity=Severity.INFO,
                message=f"Found {anonymization_result.total_org_entities} organization mention(s)",
                details="Review if these should be anonymized"
            ))

        if anonymization_result.total_gpe_entities > 0:
            self.results.append(ValidationResult(
                check_name="Location Entities (Info)",
                passed=True,
                severity=Severity.INFO,
                message=f"Found {anonymization_result.total_gpe_entities} location mention(s)",
                details="Review if these should be anonymized"
            ))

    def _check_isprs_formatting(
        self,
        extracted_elements: Dict[str, List],
        page_count: int,
        page_dimensions: Tuple[float, float],
        citation_results: Optional[Dict] = None
    ):
        """Check ISPRS formatting requirements."""
        # Page count validation
        self._check_page_count(page_count)

        # Abstract word count validation
        self._check_abstract_length(extracted_elements)

        # Font and style validation
        self._check_font_requirements(extracted_elements)

        # Font type validation (Times New Roman)
        self._check_font_type(extracted_elements)

        # Heading alignment and sequence validation
        self._check_heading_structure(extracted_elements, page_dimensions[0])

        # Element numbering order validation (headings, figures, tables, equations)
        self._check_element_numbering_order(extracted_elements, page_dimensions[0])

        # Equation number right-justification validation
        self._check_equation_number_alignment(extracted_elements, page_dimensions[0])

        # Reference alphabetical order validation
        self._check_reference_order(extracted_elements, page_dimensions[0], citation_results)

        # Reference justification validation (left-justified within columns)
        self._check_reference_justification(extracted_elements, page_dimensions[0], citation_results)

        # Page layout validation
        self._check_page_layout(page_dimensions, extracted_elements)

        # Column layout validation (single-column header, two-column body)
        self._check_column_layout(extracted_elements, page_dimensions[0])

        # Section spacing validation
        self._check_section_spacing(extracted_elements)

    def _check_page_count(self, page_count: int):
        """Check document is 6-8 pages per ISPRS requirements."""
        min_pages = PAGE_REQUIREMENTS['min_pages']
        max_pages = PAGE_REQUIREMENTS['max_pages']

        if page_count < min_pages:
            self.results.append(ValidationResult(
                check_name="Page Count",
                passed=False,
                severity=Severity.WARNING,
                message=f"Too few pages: {page_count} (minimum {min_pages})",
                details=f"ISPRS requires {min_pages}-{max_pages} pages"
            ))
        elif page_count > max_pages:
            self.results.append(ValidationResult(
                check_name="Page Count",
                passed=False,
                severity=Severity.WARNING,
                message=f"Too many pages: {page_count} (maximum {max_pages})",
                details=f"ISPRS requires {min_pages}-{max_pages} pages"
            ))
        else:
            self.results.append(ValidationResult(
                check_name="Page Count",
                passed=True,
                severity=Severity.SUCCESS,
                message=f"Page count OK: {page_count} pages",
                details=f"Within {min_pages}-{max_pages} page requirement"
            ))

    def _check_abstract_length(self, extracted_elements: Dict[str, List]):
        """Check abstract is 100-250 words per ISPRS requirements."""
        abstracts = extracted_elements.get('Abstract', [])

        if not abstracts:
            self.results.append(ValidationResult(
                check_name="Abstract Length",
                passed=False,
                severity=Severity.WARNING,
                message="No abstract found",
                details="ISPRS requires an abstract of 100-250 words"
            ))
            return

        # Combine all abstract text
        abstract_text = ' '.join(a.text for a in abstracts)
        word_count = len(abstract_text.split())

        min_words = ABSTRACT_REQUIREMENTS['min_words']
        max_words = ABSTRACT_REQUIREMENTS['max_words']

        # Collect element references with per-instance message for highlighting
        element_refs = []
        for a in abstracts:
            if hasattr(a, 'page') and hasattr(a, 'bbox'):
                if word_count < min_words:
                    instance_msg = f"Abstract: Found {word_count} words, expected minimum {min_words} words"
                elif word_count > max_words:
                    instance_msg = f"Abstract: Found {word_count} words, expected maximum {max_words} words"
                else:
                    instance_msg = f"Abstract: {word_count} words"
                element_refs.append((a.page, a.bbox, instance_msg))

        if word_count < min_words:
            self.results.append(ValidationResult(
                check_name="Abstract Length",
                passed=False,
                severity=Severity.WARNING,
                message=f"Abstract too short: {word_count} words (minimum {min_words})",
                details=f"ISPRS requires {min_words}-{max_words} words",
                element_refs=element_refs if element_refs else None
            ))
        elif word_count > max_words:
            self.results.append(ValidationResult(
                check_name="Abstract Length",
                passed=False,
                severity=Severity.WARNING,
                message=f"Abstract too long: {word_count} words (maximum {max_words})",
                details=f"ISPRS requires {min_words}-{max_words} words",
                element_refs=element_refs if element_refs else None
            ))
        else:
            self.results.append(ValidationResult(
                check_name="Abstract Length",
                passed=True,
                severity=Severity.SUCCESS,
                message=f"Abstract length OK: {word_count} words",
                details=f"Within {min_words}-{max_words} word requirement"
            ))

    def _is_clean_font_size(self, size: float) -> bool:
        """
        Check if font size is a 'clean' value (whole number or .5 fraction).
        Clean sizes are trusted as true font sizes.
        Non-clean sizes (e.g., 8.73) may be affected by formatting elements
        like subscripts, superscripts, or other special characters.
        """
        remainder = size % 1
        return remainder == 0 or abs(remainder - 0.5) < 0.01

    @staticmethod
    def _font_is_times(font_name: str) -> bool:
        """Check if a font name is Times New Roman or a known equivalent."""
        normalized = re.sub(r'[\s\-_]', '', font_name).lower()
        return any(alias in normalized for alias in TIMES_FONT_ALIASES)

    def _is_times_font(self, elem) -> bool:
        """
        Check if element uses Times font, accounting for mixed-font paragraphs.
        If element has spans, check if majority of text (by character count) uses Times.
        This allows paragraphs with math symbols (σ, q in CambriaMath) to pass
        if the majority of text is still Times New Roman.
        """
        # Check spans if available (spans are TextSpan objects, not dicts)
        if hasattr(elem, 'spans') and elem.spans:
            times_chars = 0
            total_chars = 0
            for span in elem.spans:
                # Access as object attributes, not dict keys
                span_text = getattr(span, 'text', '') or ''
                span_font = getattr(span, 'font_name', '') or ''
                char_count = len(span_text)
                total_chars += char_count
                if self._font_is_times(span_font):
                    times_chars += char_count

            if total_chars > 0:
                # If >50% of text uses Times, consider it OK
                return times_chars / total_chars > 0.5

        # Fall back to element-level font_name
        if hasattr(elem, 'font_name') and elem.font_name:
            return self._font_is_times(elem.font_name)

        return True  # No font info, assume OK

    def _bboxes_overlap(self, bbox1, bbox2) -> bool:
        """Check if two bounding boxes overlap."""
        x0_1, y0_1, x1_1, y1_1 = bbox1
        x0_2, y0_2, x1_2, y1_2 = bbox2
        return not (x1_1 < x0_2 or x1_2 < x0_1 or y1_1 < y0_2 or y1_2 < y0_1)

    def _bboxes_overlap_significantly(self, bbox1, bbox2, threshold: float = 0.5) -> bool:
        """
        Check if two bboxes overlap by at least threshold (50%) of the smaller bbox.
        Used for deduplication to identify when two elements are essentially the same region.
        """
        x0_1, y0_1, x1_1, y1_1 = bbox1
        x0_2, y0_2, x1_2, y1_2 = bbox2

        # Calculate intersection
        x_left = max(x0_1, x0_2)
        y_top = max(y0_1, y0_2)
        x_right = min(x1_1, x1_2)
        y_bottom = min(y1_1, y1_2)

        if x_right <= x_left or y_bottom <= y_top:
            return False  # No overlap

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        smaller_area = min(area1, area2)

        if smaller_area <= 0:
            return False

        return intersection_area >= threshold * smaller_area

    def _is_in_content_area(self, elem, page_height: float = 842) -> bool:
        """
        Check if element is within the main content area (not header/footer).
        Only checks vertical bounds since header/footer are top/bottom regions.
        """
        if not hasattr(elem, 'bbox'):
            return True

        top_margin = PAGE_REQUIREMENTS['margins']['top']
        bottom_margin = PAGE_REQUIREMENTS['margins']['bottom']

        x0, y0, x1, y1 = elem.bbox

        # Check if element is within vertical margins (with tolerance)
        tolerance = 10
        in_y_bounds = y0 >= (top_margin - tolerance) and y1 <= (page_height - bottom_margin + tolerance)

        return in_y_bounds

    def _check_font_requirements(self, extracted_elements: Dict[str, List]):
        """Check font size and style requirements per ISPRS."""
        font_issues = []

        # Build exclusion zones for Main_Text (figure/equation regions)
        exclude_bboxes = []
        main_text_elements = extracted_elements.get('Main_Text', [])

        # Equation elements - use bbox as-is
        for elem in extracted_elements.get('Equation', []):
            if hasattr(elem, 'bbox') and hasattr(elem, 'page'):
                exclude_bboxes.append((elem.page, elem.bbox))

        # Equation_Number - expand exclusion zone to the LEFT to cover the equation
        # Since Equation_Number has higher detection rate than Equation itself,
        # we create an exclusion zone spanning from the column edge to the equation number
        # Use column-aware expansion: left column from left_margin, right column from midpoint
        left_margin = PAGE_REQUIREMENTS['margins']['left']  # 57pt
        page_width = PAGE_REQUIREMENTS['size'][0]  # 595pt for A4
        midpoint = page_width / 2
        for elem in extracted_elements.get('Equation_Number', []):
            if hasattr(elem, 'bbox') and hasattr(elem, 'page'):
                x0, y0, x1, y1 = elem.bbox
                # Determine which column based on equation number center
                x_center = (x0 + x1) / 2
                if x_center < midpoint:
                    # Left column - expand from left margin
                    column_start = left_margin
                else:
                    # Right column - expand from midpoint (+ gap/2 for column spacing)
                    column_start = midpoint + 3  # 6pt column gap / 2
                expanded_bbox = (column_start, y0, x1, y1)
                exclude_bboxes.append((elem.page, expanded_bbox))

        # Figure/Table titles - expand upward to nearest Main_Text above
        for title_type in ['Figure_Title', 'Table_Title']:
            for title_elem in extracted_elements.get(title_type, []):
                if not (hasattr(title_elem, 'bbox') and hasattr(title_elem, 'page')):
                    continue

                x0, y0, x1, y1 = title_elem.bbox
                page = title_elem.page

                # Find nearest Main_Text bottom edge above this title on same page
                nearest_above_y = 0  # Default to top of page
                for mt in main_text_elements:
                    if hasattr(mt, 'page') and hasattr(mt, 'bbox'):
                        if mt.page == page and mt.bbox[3] < y0:  # mt bottom < title top
                            nearest_above_y = max(nearest_above_y, mt.bbox[3])

                # Expand title bbox up to nearest Main_Text
                expanded_bbox = (x0, nearest_above_y, x1, y1)
                exclude_bboxes.append((page, expanded_bbox))

        # Add header sections to exclusion zones so Main_Text elements overlapping
        # these regions won't be checked for font size (prevents duplicate/conflicting messages)
        header_types = ['Title', 'Authors', 'Affiliations', 'Keywords', 'Abstract']
        for header_type in header_types:
            for elem in extracted_elements.get(header_type, []):
                if hasattr(elem, 'bbox') and hasattr(elem, 'page'):
                    exclude_bboxes.append((elem.page, elem.bbox))

        for elem_type, requirements in FONT_REQUIREMENTS.items():
            elements = extracted_elements.get(elem_type, [])
            if not elements:
                continue

            required_size = requirements['size']
            required_bold = requirements.get('bold', False)

            # Check each element individually (including Main_Text)
            for elem in elements:
                # Skip very short text (likely fragments)
                if hasattr(elem, 'text') and len(elem.text.strip()) < 5:
                    continue

                # For Main_Text: filter header/footer and figure/equation regions
                if elem_type == 'Main_Text' and hasattr(elem, 'bbox') and hasattr(elem, 'page'):
                    # Skip if outside content area (header/footer)
                    if not self._is_in_content_area(elem):
                        continue

                    # Skip if inside figure/equation exclusion zone
                    in_exclusion = False
                    for ex_page, ex_bbox in exclude_bboxes:
                        if elem.page == ex_page and self._bboxes_overlap(elem.bbox, ex_bbox):
                            in_exclusion = True
                            break
                    if in_exclusion:
                        continue

                # Get element reference for highlighting
                elem_ref = None
                if hasattr(elem, 'page') and hasattr(elem, 'bbox'):
                    elem_ref = (elem.page, elem.bbox)

                # Check font size - allow tolerance for PDF rendering differences
                if hasattr(elem, 'font_size') and elem.font_size is not None:
                    if abs(elem.font_size - required_size) > FONT_SIZE_TOLERANCE:
                        is_clean = self._is_clean_font_size(elem.font_size)
                        font_issues.append({
                            'type': elem_type,
                            'issue': 'size',
                            'expected': required_size,
                            'found': elem.font_size,
                            'text': elem.text[:40],
                            'element_ref': elem_ref,
                            'is_clean_size': is_clean
                        })

                # Check bold requirement
                if required_bold and hasattr(elem, 'is_bold'):
                    if not elem.is_bold:
                        font_issues.append({
                            'type': elem_type,
                            'issue': 'bold',
                            'expected': 'bold',
                            'found': 'not bold',
                            'text': elem.text[:40],
                            'element_ref': elem_ref
                        })

        # Group issues by type for cleaner reporting
        size_issues = [i for i in font_issues if i['issue'] == 'size']
        bold_issues = [i for i in font_issues if i['issue'] == 'bold']

        # Collect element refs with per-instance messages for highlighting
        # Deduplicate by bbox OVERLAP, prioritizing non-Main_Text types so specific element types
        # (Keywords, Abstract, etc.) take precedence over Main_Text in annotations
        size_refs = []
        seen_size_bboxes = {}  # (page, bbox) -> issue dict

        for i in size_issues:
            if i.get('element_ref'):
                page, bbox = i['element_ref']

                # Find if any existing entry overlaps significantly
                overlapping_key = None
                for existing_key in seen_size_bboxes:
                    ex_page, ex_bbox = existing_key
                    if ex_page == page and self._bboxes_overlap_significantly(bbox, ex_bbox):
                        overlapping_key = existing_key
                        break

                if overlapping_key:
                    # Prioritize non-Main_Text types over Main_Text
                    if seen_size_bboxes[overlapping_key]['type'] == 'Main_Text' and i['type'] != 'Main_Text':
                        # Replace with more specific type, use its bbox
                        del seen_size_bboxes[overlapping_key]
                        seen_size_bboxes[(page, bbox)] = i
                    # Otherwise keep existing (first specific type wins)
                else:
                    # No overlap, add new entry
                    seen_size_bboxes[(page, bbox)] = i

        # Build refs from deduplicated issues
        for (page, bbox), i in seen_size_bboxes.items():
            is_clean = i.get('is_clean_size', True)
            if is_clean:
                instance_msg = f"{i['type']}: Found {i['found']:.1f}pt, expected {i['expected']}pt"
            else:
                instance_msg = f"{i['type']}: Found {i['found']:.2f}pt, expected {i['expected']}pt (may include formatted text)"
            size_refs.append((page, bbox, instance_msg))

        # Same deduplication for bold issues using overlap detection
        bold_refs = []
        seen_bold_bboxes = {}  # (page, bbox) -> issue dict

        for i in bold_issues:
            if i.get('element_ref'):
                page, bbox = i['element_ref']

                # Find if any existing entry overlaps significantly
                overlapping_key = None
                for existing_key in seen_bold_bboxes:
                    ex_page, ex_bbox = existing_key
                    if ex_page == page and self._bboxes_overlap_significantly(bbox, ex_bbox):
                        overlapping_key = existing_key
                        break

                if overlapping_key:
                    # Prioritize non-Main_Text types
                    if seen_bold_bboxes[overlapping_key]['type'] == 'Main_Text' and i['type'] != 'Main_Text':
                        del seen_bold_bboxes[overlapping_key]
                        seen_bold_bboxes[(page, bbox)] = i
                else:
                    seen_bold_bboxes[(page, bbox)] = i

        # Build refs from deduplicated issues
        for (page, bbox), i in seen_bold_bboxes.items():
            instance_msg = f"{i['type']}: Missing bold formatting (required for this element type)"
            bold_refs.append((page, bbox, instance_msg))

        # Report size issues
        if size_issues:
            # Group by element type
            types_affected = set(i['type'] for i in size_issues)
            # Check if any non-clean sizes were found
            has_non_clean = any(not i.get('is_clean_size', True) for i in size_issues)
            details = f"Affected types: {', '.join(types_affected)}. "
            details += f"Example: {size_issues[0]['type']} expected {size_issues[0]['expected']}pt, "
            details += f"found {size_issues[0]['found']:.1f}pt"
            if has_non_clean:
                details += ". Note: Non-standard sizes (not whole or .5) may be affected by subscripts, superscripts, or other formatting"

            self.results.append(ValidationResult(
                check_name="Font Size",
                passed=False,
                severity=Severity.WARNING,
                message=f"Font size issues in {len(size_issues)} element(s)",
                details=details,
                element_refs=size_refs if size_refs else None
            ))
        else:
            self.results.append(ValidationResult(
                check_name="Font Size",
                passed=True,
                severity=Severity.SUCCESS,
                message="Font sizes conform to ISPRS requirements",
                details="Title=12pt, all other elements=9pt"
            ))

        # Report bold issues
        if bold_issues:
            types_affected = set(i['type'] for i in bold_issues)
            self.results.append(ValidationResult(
                check_name="Bold Style",
                passed=False,
                severity=Severity.WARNING,
                message=f"Missing bold formatting in {len(bold_issues)} element(s)",
                details=f"Affected types: {', '.join(types_affected)}. " +
                       f"ISPRS requires bold for: Title, Headings, Sub_Headings, Sub_sub_Headings",
                element_refs=bold_refs if bold_refs else None
            ))
        else:
            self.results.append(ValidationResult(
                check_name="Bold Style",
                passed=True,
                severity=Severity.SUCCESS,
                message="Bold formatting conforms to ISPRS requirements",
                details="Title, Headings, Subheadings are bold"
            ))

    def _check_font_type(self, extracted_elements: Dict[str, List]):
        """Check font type is Times New Roman per ISPRS requirements."""
        font_issues = []

        # Build equation exclusion zones (elements inside these shouldn't be flagged)
        equation_bboxes = []
        for elem in extracted_elements.get('Equation', []):
            if hasattr(elem, 'bbox') and hasattr(elem, 'page'):
                equation_bboxes.append((elem.page, elem.bbox))
        for elem in extracted_elements.get('Equation_Number', []):
            if hasattr(elem, 'bbox') and hasattr(elem, 'page'):
                equation_bboxes.append((elem.page, elem.bbox))

        # Element types to check (excludes Equation, Equation_Number)
        check_types = ['Title', 'Authors', 'Affiliations', 'Keywords', 'Abstract',
                       'Main_Text', 'Headings', 'Sub_Headings', 'Sub_sub_Headings',
                       'References', 'Figure_Title', 'Table_Title']

        for elem_type in check_types:
            for elem in extracted_elements.get(elem_type, []):
                # Skip if element overlaps with equation bbox
                if hasattr(elem, 'bbox') and hasattr(elem, 'page'):
                    in_equation_zone = any(
                        elem.page == eq_page and self._bboxes_overlap(elem.bbox, eq_bbox)
                        for eq_page, eq_bbox in equation_bboxes
                    )
                    if in_equation_zone:
                        continue

                # Check font using _is_times_font() which handles spans and mixed-font paragraphs
                if not self._is_times_font(elem):
                    elem_ref = None
                    if hasattr(elem, 'page') and hasattr(elem, 'bbox'):
                        elem_ref = (elem.page, elem.bbox)
                    # Get font name for reporting
                    font_name = elem.font_name if hasattr(elem, 'font_name') and elem.font_name else 'Unknown'
                    font_issues.append({
                        'type': elem_type,
                        'font': font_name,
                        'text': elem.text[:40] if hasattr(elem, 'text') else '',
                        'element_ref': elem_ref
                    })

        # Build element_refs with per-instance messages
        element_refs = []
        for issue in font_issues:
            if issue.get('element_ref'):
                page, bbox = issue['element_ref']
                instance_msg = f"{issue['type']}: Found '{issue['font']}', expected Times New Roman"
                element_refs.append((page, bbox, instance_msg))

        if font_issues:
            types_affected = set(i['type'] for i in font_issues)
            self.results.append(ValidationResult(
                check_name="Font Type",
                passed=False,
                severity=Severity.WARNING,
                message=f"Font type issues in {len(font_issues)} element(s)",
                details=f"ISPRS requires Times New Roman. Affected: {', '.join(types_affected)}",
                element_refs=element_refs if element_refs else None
            ))
        else:
            self.results.append(ValidationResult(
                check_name="Font Type",
                passed=True,
                severity=Severity.SUCCESS,
                message="Font type conforms to ISPRS requirements",
                details="Times New Roman used throughout (equations excluded)"
            ))

    def _get_column_center(self, elem_x_center: float, page_width: float) -> float:
        """Determine column center based on element position."""
        mid_page = page_width / 2
        left_margin = PAGE_REQUIREMENTS['margins']['left']
        right_margin = PAGE_REQUIREMENTS['margins']['right']

        # Two-column layout:
        # Left column: left_margin to mid_page - gutter/2 (~8pt)
        # Right column: mid_page + gutter/2 to page_width - right_margin

        if elem_x_center < mid_page:
            # Left column center
            col_left = left_margin
            col_right = mid_page - 8  # Approximate gutter
            return (col_left + col_right) / 2
        else:
            # Right column center
            col_left = mid_page + 8  # Approximate gutter
            col_right = page_width - right_margin
            return (col_left + col_right) / 2

    def _check_heading_structure(self, extracted_elements: Dict[str, List], page_width: float):
        """Check heading alignment and numbering sequence per ISPRS."""
        headings = extracted_elements.get('Headings', [])
        sub_headings = extracted_elements.get('Sub_Headings', [])

        # Check major heading alignment (should be centered within column)
        alignment_issues = []

        for heading in headings:
            if hasattr(heading, 'bbox') and len(heading.bbox) >= 4:
                elem_center = (heading.bbox[0] + heading.bbox[2]) / 2
                # Get the center of the column this heading is in
                column_center = self._get_column_center(elem_center, page_width)
                offset = abs(elem_center - column_center)

                if offset > CENTER_TOLERANCE:
                    elem_ref = None
                    if hasattr(heading, 'page'):
                        elem_ref = (heading.page, heading.bbox)
                    alignment_issues.append({
                        'text': heading.text[:30],
                        'offset': offset,
                        'element_ref': elem_ref
                    })

        # Collect element refs with per-instance messages for highlighting
        alignment_refs = []
        for i in alignment_issues:
            if i.get('element_ref'):
                page, bbox = i['element_ref']
                instance_msg = f"Heading: '{i['text'][:30]}...' is {i['offset']:.0f}pt off-center"
                alignment_refs.append((page, bbox, instance_msg))

        if alignment_issues:
            self.results.append(ValidationResult(
                check_name="Heading Alignment",
                passed=False,
                severity=Severity.WARNING,
                message=f"{len(alignment_issues)} major heading(s) not centered within column",
                details=f"Example: '{alignment_issues[0]['text']}...' " +
                       f"is {alignment_issues[0]['offset']:.0f}pt off-center",
                element_refs=alignment_refs if alignment_refs else None
            ))
        elif headings:
            self.results.append(ValidationResult(
                check_name="Heading Alignment",
                passed=True,
                severity=Severity.SUCCESS,
                message="Major headings are properly centered within columns",
                details=f"Checked {len(headings)} heading(s)"
            ))

        # Check heading numbering sequence
        heading_nums = []
        heading_pattern = re.compile(r'^(\d+)\.\s+')
        for h in headings:
            match = heading_pattern.match(h.text.strip())
            if match:
                heading_nums.append(int(match.group(1)))

        if heading_nums:
            heading_nums_sorted = sorted(set(heading_nums))
            expected = list(range(1, max(heading_nums_sorted) + 1))
            missing = set(expected) - set(heading_nums_sorted)

            if missing:
                self.results.append(ValidationResult(
                    check_name="Heading Numbering",
                    passed=False,
                    severity=Severity.WARNING,
                    message=f"Missing heading number(s): {sorted(missing)}",
                    details=f"Found: {heading_nums_sorted}, Expected: {expected}"
                ))
            else:
                self.results.append(ValidationResult(
                    check_name="Heading Numbering",
                    passed=True,
                    severity=Severity.SUCCESS,
                    message="Heading numbering sequence is correct",
                    details=f"Found headings: {heading_nums_sorted}"
                ))

        # Check sub-heading numbering sequence
        sub_heading_nums = []
        sub_heading_pattern = re.compile(r'^(\d+)\.(\d+)\.?\s+')
        for sh in sub_headings:
            match = sub_heading_pattern.match(sh.text.strip())
            if match:
                sub_heading_nums.append((int(match.group(1)), int(match.group(2))))

        if sub_heading_nums:
            # Group by parent heading number
            by_parent = {}
            for parent, child in sub_heading_nums:
                by_parent.setdefault(parent, []).append(child)

            missing_subs = []
            for parent, children in sorted(by_parent.items()):
                children_sorted = sorted(set(children))
                expected_children = list(range(1, max(children_sorted) + 1))
                for c in expected_children:
                    if c not in children_sorted:
                        missing_subs.append(f"{parent}.{c}")

            if missing_subs:
                self.results.append(ValidationResult(
                    check_name="Sub-heading Numbering",
                    passed=False,
                    severity=Severity.WARNING,
                    message=f"Missing sub-heading number(s): {', '.join(missing_subs)}",
                    details=f"Found sub-headings: {sorted(set(sub_heading_nums))}"
                ))
            else:
                self.results.append(ValidationResult(
                    check_name="Sub-heading Numbering",
                    passed=True,
                    severity=Severity.SUCCESS,
                    message="Sub-heading numbering sequences are correct",
                    details=f"Found sub-headings: {sorted(set(sub_heading_nums))}"
                ))

        # Check heading case (should not be ALL CAPS)
        self._check_heading_case(headings, sub_headings, extracted_elements.get('Sub_sub_Headings', []))

    def _check_heading_case(self, headings: List, sub_headings: List, sub_sub_headings: List):
        """Check that headings use proper case (not ALL CAPS)."""
        all_caps_issues = []

        # Helper to check if text (excluding numbers) is all caps
        def is_all_caps(text: str) -> bool:
            # Remove leading numbers and punctuation (e.g., "1. INTRODUCTION")
            text_only = re.sub(r'^[\d\.\s]+', '', text.strip())
            # Check if remaining text is all uppercase (and has letters)
            alpha_chars = [c for c in text_only if c.isalpha()]
            return len(alpha_chars) > 0 and all(c.isupper() for c in alpha_chars)

        # Check all heading types
        for heading_type, heading_list in [
            ('Heading', headings),
            ('Sub-heading', sub_headings),
            ('Sub-sub-heading', sub_sub_headings)
        ]:
            for heading in heading_list:
                if hasattr(heading, 'text') and is_all_caps(heading.text):
                    elem_ref = None
                    if hasattr(heading, 'page') and hasattr(heading, 'bbox'):
                        elem_ref = (heading.page, heading.bbox)
                    all_caps_issues.append({
                        'type': heading_type,
                        'text': heading.text[:40],
                        'element_ref': elem_ref
                    })

        # Build element refs for highlighting
        element_refs = []
        for issue in all_caps_issues:
            if issue.get('element_ref'):
                page, bbox = issue['element_ref']
                instance_msg = f"{issue['type']}: '{issue['text'][:30]}...' should not be ALL CAPS"
                element_refs.append((page, bbox, instance_msg))

        if all_caps_issues:
            self.results.append(ValidationResult(
                check_name="Heading Case Format",
                passed=False,
                severity=Severity.WARNING,
                message=f"Found {len(all_caps_issues)} heading(s) using ALL CAPS",
                details="Headings should use title case (e.g., 'Introduction' not 'INTRODUCTION')",
                element_refs=element_refs if element_refs else None
            ))
        elif headings or sub_headings or sub_sub_headings:
            total = len(headings) + len(sub_headings) + len(sub_sub_headings)
            self.results.append(ValidationResult(
                check_name="Heading Case Format",
                passed=True,
                severity=Severity.SUCCESS,
                message="Headings use proper case formatting",
                details=f"Checked {total} heading(s)"
            ))

    def _check_element_numbering_order(
        self,
        extracted_elements: Dict[str, List],
        page_width: float
    ):
        """
        Check that numbered elements appear in correct sequential order in document flow.
        Checks: Headings, Sub_Headings, Sub_sub_Headings, Figure_Number, Table_Number, Equation_Number
        """
        # Configuration for each element type
        element_configs = [
            {
                'type': 'Headings',
                'pattern': r'^(\d+)\.\s+',  # "1. Introduction"
                'name': 'Heading',
                'format_num': lambda n: str(n)
            },
            {
                'type': 'Sub_Headings',
                'pattern': r'^(\d+)\.(\d+)\.?\s+',  # "2.1 Methods"
                'name': 'Sub-heading',
                'format_num': lambda n: f"{n[0]}.{n[1]}"
            },
            {
                'type': 'Sub_sub_Headings',
                'pattern': r'^(\d+)\.(\d+)\.(\d+)\.?\s+',  # "2.1.1 Details"
                'name': 'Sub-sub-heading',
                'format_num': lambda n: f"{n[0]}.{n[1]}.{n[2]}"
            },
            {
                'type': 'Figure_Number',
                'pattern': r'(?:Figure|Fig\.?)\s*(\d+)',  # "Figure 1" or "Fig. 1"
                'name': 'Figure',
                'format_num': lambda n: str(n)
            },
            {
                'type': 'Table_Number',
                'pattern': r'Table\s*(\d+)',  # "Table 1"
                'name': 'Table',
                'format_num': lambda n: str(n)
            },
            {
                'type': 'Equation_Number',
                'pattern': r'\((\d+)\)',  # "(1)" or "(13)"
                'name': 'Equation',
                'format_num': lambda n: str(n)
            }
        ]

        mid_page = page_width / 2

        for config in element_configs:
            elements = extracted_elements.get(config['type'], [])
            if len(elements) < 2:
                continue

            # Sort by reading order: (page, column, y_position)
            def reading_order(elem, mid=mid_page):
                page = elem.page if hasattr(elem, 'page') else 0
                if hasattr(elem, 'bbox') and elem.bbox:
                    x_center = (elem.bbox[0] + elem.bbox[2]) / 2
                    y_pos = elem.bbox[1]
                else:
                    x_center, y_pos = 0, 0
                column = 0 if x_center < mid else 1
                return (page, column, y_pos)

            sorted_elements = sorted(elements, key=reading_order)

            # Extract numbers and check order
            pattern = re.compile(config['pattern'], re.IGNORECASE)
            numbered_elements = []

            for elem in sorted_elements:
                text = elem.text.strip() if hasattr(elem, 'text') else ''
                match = pattern.search(text)
                if match:
                    groups = match.groups()
                    if len(groups) == 1:
                        num = int(groups[0])
                    else:
                        num = tuple(int(g) for g in groups)
                    numbered_elements.append({
                        'num': num,
                        'elem': elem,
                        'text': text[:40]
                    })

            # Check for out-of-order elements
            order_issues = []
            for i in range(len(numbered_elements) - 1):
                curr = numbered_elements[i]
                next_item = numbered_elements[i + 1]

                # Compare: curr should be <= next
                if isinstance(curr['num'], tuple):
                    is_out_of_order = curr['num'] > next_item['num']
                else:
                    is_out_of_order = curr['num'] > next_item['num']

                if is_out_of_order:
                    elem_ref = None
                    if hasattr(next_item['elem'], 'page') and hasattr(next_item['elem'], 'bbox'):
                        elem_ref = (next_item['elem'].page, next_item['elem'].bbox)
                    order_issues.append({
                        'curr_num': config['format_num'](curr['num']),
                        'next_num': config['format_num'](next_item['num']),
                        'text': next_item['text'],
                        'element_ref': elem_ref
                    })

            # Report issues
            if order_issues:
                element_refs = []
                for issue in order_issues:
                    if issue.get('element_ref'):
                        page, bbox = issue['element_ref']
                        instance_msg = f"{config['name']} {issue['next_num']} appears after {issue['curr_num']} (out of order)"
                        element_refs.append((page, bbox, instance_msg))

                self.results.append(ValidationResult(
                    check_name=f"{config['name']} Numbering Order",
                    passed=False,
                    severity=Severity.WARNING,
                    message=f"{len(order_issues)} {config['name'].lower()}(s) appear out of order",
                    details=f"Example: {config['name']} {order_issues[0]['next_num']} appears after {order_issues[0]['curr_num']}",
                    element_refs=element_refs if element_refs else None
                ))
            elif len(numbered_elements) >= 2:
                self.results.append(ValidationResult(
                    check_name=f"{config['name']} Numbering Order",
                    passed=True,
                    severity=Severity.SUCCESS,
                    message=f"{config['name']} numbering is in correct order",
                    details=f"Checked {len(numbered_elements)} {config['name'].lower()}(s)"
                ))

    def _sort_by_column_flow(self, elements: List, page_width: float) -> List:
        """Sort elements by column reading order (left column, then right column per page)."""
        mid_page = page_width / 2

        def sort_key(elem):
            page = elem.page if hasattr(elem, 'page') else 0
            x_center = (elem.bbox[0] + elem.bbox[2]) / 2 if hasattr(elem, 'bbox') else 0
            y_pos = elem.bbox[1] if hasattr(elem, 'bbox') else 0

            # Column: 0 for left, 1 for right
            column = 0 if x_center < mid_page else 1

            return (page, column, y_pos)

        return sorted(elements, key=sort_key)

    def _check_equation_number_alignment(
        self,
        extracted_elements: Dict[str, List],
        page_width: float
    ):
        """Check equation numbers are right-justified within their column per ISPRS."""
        equation_numbers = extracted_elements.get('Equation_Number', [])
        if not equation_numbers:
            return

        right_margin = PAGE_REQUIREMENTS['margins']['right']  # 57pt
        midpoint = page_width / 2
        column_gap = 17  # 6mm = ~17pt column gap
        tolerance = 15  # 15pt tolerance for right-justification

        element_refs = []

        for elem in equation_numbers:
            if not (hasattr(elem, 'bbox') and hasattr(elem, 'page')):
                continue

            x0, y0, x1, y1 = elem.bbox
            x_center = (x0 + x1) / 2

            # Determine column and its right edge
            if x_center < midpoint:
                # Left column - right edge is midpoint minus half the column gap
                column_right = midpoint - (column_gap / 2)
            else:
                # Right column - right edge is page width minus right margin
                column_right = page_width - right_margin

            # Check if equation number is right-justified (within tolerance)
            eq_right = x1
            if eq_right < column_right - tolerance:
                # Not right-justified - add warning
                eq_text = elem.text if hasattr(elem, 'text') else 'Unknown'
                instance_msg = f"Equation number '{eq_text}' not right-justified in column"
                # Convert BoundingBox to tuple if needed
                bbox = elem.bbox
                if hasattr(bbox, 'x0'):
                    bbox = (bbox.x0, bbox.y0, bbox.x1, bbox.y1)
                element_refs.append((elem.page, bbox, instance_msg))

        if element_refs:
            self.results.append(ValidationResult(
                check_name="Equation Number Alignment",
                passed=False,
                severity=Severity.WARNING,
                message=f"Equation numbers should be right-justified (flush right) within column",
                details=f"{len(element_refs)} equation number(s) not properly right-justified",
                element_refs=element_refs
            ))

    def _check_reference_order(
        self,
        extracted_elements: Dict[str, List],
        page_width: float,
        citation_results: Optional[Dict] = None
    ):
        """Check references are in alphabetical order per ISPRS."""
        # Use combined references from citation validation if available
        # This ensures fragments that span columns/pages are properly merged
        if citation_results and 'references_parsed' in citation_results:
            # Create wrapper objects for compatibility with existing code
            class RefWrapper:
                def __init__(self, parsed_ref):
                    self.text = parsed_ref.original_text
                    self.page = parsed_ref.page if parsed_ref.page is not None else 0
                    self.bbox = parsed_ref.bbox  # Now includes bbox for highlighting

            references = [RefWrapper(r) for r in citation_results['references_parsed']]
            sorted_refs = self._sort_by_column_flow(references, page_width)
        else:
            references = extracted_elements.get('References', [])
            if len(references) < 2:
                return  # Not enough to check order
            # Sort references by column flow (left col top→bottom, right col top→bottom, next page)
            sorted_refs = self._sort_by_column_flow(references, page_width)

        if len(references) < 2:
            return  # Not enough to check order

        # Extract first author surnames for comparison (in reading order)
        # Also track the reference element for highlighting
        ref_data = []
        author_pattern = re.compile(r'^([A-Z][a-z\u00C0-\u024F]+)')

        for ref in sorted_refs:
            text = ref.text.strip()
            match = author_pattern.match(text)
            if match:
                elem_ref = None
                if hasattr(ref, 'page') and hasattr(ref, 'bbox') and ref.bbox is not None:
                    elem_ref = (ref.page, ref.bbox)
                ref_data.append({
                    'surname': match.group(1).lower(),
                    'text': text[:50],
                    'element_ref': elem_ref
                })

        if len(ref_data) < 2:
            return

        # Check if sorted alphabetically
        surnames_only = [r['surname'] for r in ref_data]
        is_sorted = surnames_only == sorted(surnames_only)

        if not is_sorted:
            # Find out-of-order references and collect their element_refs with per-instance messages
            element_refs = []
            out_of_order = []
            for i in range(len(surnames_only) - 1):
                if surnames_only[i] > surnames_only[i + 1]:
                    out_of_order.append((ref_data[i]['surname'], ref_data[i + 1]['surname']))
                    # Add both the out-of-order reference and the one it should come after
                    if ref_data[i]['element_ref']:
                        page, bbox = ref_data[i]['element_ref']
                        instance_msg = f"Reference '{ref_data[i]['surname']}' should come after '{ref_data[i + 1]['surname']}'"
                        element_refs.append((page, bbox, instance_msg))
                    if ref_data[i + 1]['element_ref']:
                        page, bbox = ref_data[i + 1]['element_ref']
                        instance_msg = f"Reference '{ref_data[i + 1]['surname']}' should come before '{ref_data[i]['surname']}'"
                        element_refs.append((page, bbox, instance_msg))

            if out_of_order:
                self.results.append(ValidationResult(
                    check_name="Reference Order",
                    passed=False,
                    severity=Severity.WARNING,
                    message="References not in alphabetical order",
                    details=f"'{out_of_order[0][0]}' should come after '{out_of_order[0][1]}'",
                    element_refs=element_refs if element_refs else None
                ))
        else:
            self.results.append(ValidationResult(
                check_name="Reference Order",
                passed=True,
                severity=Severity.SUCCESS,
                message="References are in alphabetical order",
                details=f"Checked {len(ref_data)} references"
            ))

    def _check_reference_justification(
        self,
        extracted_elements: Dict[str, List],
        page_width: float,
        citation_results: Optional[Dict] = None
    ):
        """Check references are left-justified within their columns."""
        # Use combined references from citation_results if available
        if citation_results and 'references_parsed' in citation_results:
            class RefWrapper:
                def __init__(self, parsed_ref):
                    self.text = parsed_ref.original_text
                    self.page = parsed_ref.page if parsed_ref.page is not None else 0
                    self.bbox = parsed_ref.bbox
            references = [RefWrapper(r) for r in citation_results['references_parsed'] if r.bbox]
        else:
            references = [r for r in extracted_elements.get('References', []) if hasattr(r, 'bbox') and r.bbox]

        if not references:
            return

        # Column boundaries
        left_margin = PAGE_REQUIREMENTS['margins']['left']  # 57pt
        mid_page = page_width / 2
        column_gap = 17  # 6mm

        # Left column left edge: left_margin (~57pt)
        # Right column left edge: mid_page + column_gap/2 (~306pt)
        left_col_edge = left_margin
        right_col_edge = mid_page + column_gap / 2

        justify_tolerance = 15  # Allow 15pt deviation

        alignment_issues = []
        for ref in references:
            if not ref.bbox:
                continue

            x0 = ref.bbox[0]
            x_center = (ref.bbox[0] + ref.bbox[2]) / 2

            # Determine which column
            if x_center < mid_page:
                expected_x = left_col_edge
            else:
                expected_x = right_col_edge

            offset = x0 - expected_x

            if abs(offset) > justify_tolerance:
                elem_ref = (ref.page, ref.bbox) if hasattr(ref, 'page') else None
                alignment_issues.append({
                    'text': ref.text[:40] if hasattr(ref, 'text') else '',
                    'offset': offset,
                    'element_ref': elem_ref
                })

        # Build element_refs with messages
        element_refs = []
        for issue in alignment_issues:
            if issue.get('element_ref'):
                page, bbox = issue['element_ref']
                direction = "right" if issue['offset'] > 0 else "left"
                instance_msg = f"Reference not left-justified: {abs(issue['offset']):.0f}pt too far {direction}"
                element_refs.append((page, bbox, instance_msg))

        if alignment_issues:
            self.results.append(ValidationResult(
                check_name="Reference Justification",
                passed=False,
                severity=Severity.WARNING,
                message=f"{len(alignment_issues)} reference(s) not properly left-justified",
                details="ISPRS requires left-justified text within columns",
                element_refs=element_refs if element_refs else None
            ))
        else:
            self.results.append(ValidationResult(
                check_name="Reference Justification",
                passed=True,
                severity=Severity.SUCCESS,
                message="References are properly left-justified",
                details=f"Checked {len(references)} reference(s)"
            ))

    def _check_page_layout(
        self,
        page_dimensions: Tuple[float, float],
        extracted_elements: Dict[str, List]
    ):
        """Check page size is A4 per ISPRS requirements."""
        width, height = page_dimensions
        expected_width, expected_height = PAGE_REQUIREMENTS['size']

        # Check page size
        width_ok = abs(width - expected_width) <= PAGE_SIZE_TOLERANCE
        height_ok = abs(height - expected_height) <= PAGE_SIZE_TOLERANCE

        if not width_ok or not height_ok:
            self.results.append(ValidationResult(
                check_name="Page Size",
                passed=False,
                severity=Severity.WARNING,
                message=f"Page size should be A4 (595×842pt), found {width:.0f}×{height:.0f}pt",
                details="ISPRS requires A4 paper size (210mm × 297mm)"
            ))
        else:
            self.results.append(ValidationResult(
                check_name="Page Size",
                passed=True,
                severity=Severity.SUCCESS,
                message="Page size is A4",
                details=f"Dimensions: {width:.0f}×{height:.0f}pt"
            ))

        # Check margins by analyzing element positions
        margin_issues = []
        left_margin = PAGE_REQUIREMENTS['margins']['left']
        right_margin = PAGE_REQUIREMENTS['margins']['right']

        # Check main text margins
        main_text = extracted_elements.get('Main_Text', [])
        for elem in main_text[:10]:  # Sample first 10 elements
            if hasattr(elem, 'bbox') and len(elem.bbox) >= 4:
                left_pos = elem.bbox[0]
                right_pos = elem.bbox[2]

                if left_pos < left_margin - MARGIN_TOLERANCE:
                    margin_issues.append(f"Left margin too small: {left_pos:.0f}pt (expected {left_margin}pt)")
                    break

                if right_pos > width - right_margin + MARGIN_TOLERANCE:
                    margin_issues.append(f"Right margin too small: {width - right_pos:.0f}pt (expected {right_margin}pt)")
                    break

        if margin_issues:
            self.results.append(ValidationResult(
                check_name="Page Margins",
                passed=False,
                severity=Severity.WARNING,
                message="Page margins may not meet ISPRS requirements",
                details=margin_issues[0]
            ))
        elif main_text:
            self.results.append(ValidationResult(
                check_name="Page Margins",
                passed=True,
                severity=Severity.SUCCESS,
                message="Page margins appear correct",
                details=f"Left/Right: 20mm, Top/Bottom: 25mm"
            ))

    def _check_column_layout(
        self,
        extracted_elements: Dict[str, List],
        page_width: float
    ):
        """
        Check column layout requirements per ISPRS:
        - Single-column for header sections (Title, Authors, Affiliations, Keywords, Abstract)
        - Two-column for body sections (Main_Text, Headings, References, etc.)
        """
        layout_issues = []

        # Page width is ~595pt for A4
        # Margins: 20mm (57pt) left/right
        # Content width: 595 - 2*57 = 481pt
        # Two columns: 82mm (233pt) each with 6mm (17pt) gap
        left_margin = PAGE_REQUIREMENTS['margins']['left']  # 57pt
        content_width = page_width - 2 * left_margin  # ~481pt
        column_gap = 17  # 6mm in points
        two_column_width = (content_width - column_gap) / 2  # ~232pt

        # Tolerance for width checks
        width_tolerance = 30  # ~10mm tolerance

        # Note: Header sections (Title, Authors, Affiliations, Keywords, Abstract) can span
        # either single or two columns, so we don't check them.
        # Figures and Tables are also excluded as they may legitimately span two columns.

        # Build header section exclusion zones for page 0
        # Elements that overlap these zones should not be checked for two-column layout
        header_types = ['Title', 'Authors', 'Affiliations', 'Keywords', 'Abstract']
        header_zones = []
        for header_type in header_types:
            for elem in extracted_elements.get(header_type, []):
                if hasattr(elem, 'bbox') and hasattr(elem, 'page'):
                    header_zones.append({
                        'page': elem.page,
                        'bbox': elem.bbox,
                        'type': header_type
                    })

        def overlaps_header_zone(elem_page, elem_bbox):
            """Check if an element overlaps any header zone."""
            for zone in header_zones:
                if zone['page'] != elem_page:
                    continue
                # Calculate overlap
                x0, y0, x1, y1 = elem_bbox
                zx0, zy0, zx1, zy1 = zone['bbox']

                int_x0 = max(x0, zx0)
                int_y0 = max(y0, zy0)
                int_x1 = min(x1, zx1)
                int_y1 = min(y1, zy1)

                if int_x0 < int_x1 and int_y0 < int_y1:
                    # Calculate overlap ratio
                    int_area = (int_x1 - int_x0) * (int_y1 - int_y0)
                    elem_area = (x1 - x0) * (y1 - y0)
                    if elem_area > 0 and int_area / elem_area >= 0.3:
                        return True
            return False

        # Check two-column sections (should be narrow - fit within single column)
        two_column_types = ['Main_Text', 'Headings', 'Sub_Headings', 'Sub_sub_Headings', 'References']
        max_two_column_width = two_column_width + width_tolerance  # ~262pt

        for elem_type in two_column_types:
            for elem in extracted_elements.get(elem_type, []):
                if not hasattr(elem, 'bbox'):
                    continue

                # Skip elements that overlap header zones (they're part of header section)
                elem_page = elem.page if hasattr(elem, 'page') else 0
                if overlaps_header_zone(elem_page, elem.bbox):
                    continue

                # Skip elements outside content area (document header/footer)
                if not self._is_in_content_area(elem):
                    continue

                elem_width = elem.bbox[2] - elem.bbox[0]

                # If element is too wide (spans across column gap), flag it
                if elem_width > max_two_column_width:
                    layout_issues.append({
                        'type': elem_type,
                        'page': elem_page,
                        'bbox': elem.bbox,
                        'text': elem.text[:50] if hasattr(elem, 'text') else '',
                        'issue': f'{elem_type} spans across columns - should be two-column layout'
                    })

        # Report issues with ValidationResult
        if layout_issues:
            element_refs = [(i['page'], i['bbox'], i['issue']) for i in layout_issues]
            self.results.append(ValidationResult(
                check_name="Column Layout",
                passed=False,
                severity=Severity.ERROR,
                message=f"Column layout issues detected in {len(layout_issues)} element(s)",
                details="ISPRS requires single-column for header (Title through Abstract) and two-column for body (Introduction through References)",
                element_refs=element_refs
            ))
        else:
            self.results.append(ValidationResult(
                check_name="Column Layout",
                passed=True,
                severity=Severity.SUCCESS,
                message="Column layout conforms to ISPRS requirements"
            ))

    def _check_section_spacing(self, extracted_elements: Dict[str, List]):
        """Check vertical spacing between sections per ISPRS requirements."""
        # Sort all relevant elements by page and y-position
        section_elements = []
        section_types = ['Title', 'Authors', 'Affiliations', 'Keywords', 'Abstract', 'Headings']

        for elem_type in section_types:
            for elem in extracted_elements.get(elem_type, []):
                if hasattr(elem, 'bbox') and hasattr(elem, 'page'):
                    section_elements.append((elem_type, elem))

        if len(section_elements) < 2:
            return

        # Sort by page, then y0
        section_elements.sort(key=lambda x: (x[1].page, x[1].bbox[1]))

        spacing_issues = []
        # Check gaps between consecutive section types on same page
        for i in range(len(section_elements) - 1):
            curr_type, curr_elem = section_elements[i]
            next_type, next_elem = section_elements[i + 1]

            if curr_elem.page != next_elem.page:
                continue

            gap = next_elem.bbox[1] - curr_elem.bbox[3]  # y0_next - y1_curr

            # Authors → Keywords should have ~2 blank lines (~24pt gap)
            if curr_type in ['Authors', 'Affiliations'] and next_type == 'Keywords':
                if gap < 18:  # Less than ~2 lines
                    spacing_issues.append({
                        'from': curr_type,
                        'to': next_type,
                        'gap': gap,
                        'expected': '2 blank lines (~24pt)'
                    })

            # Keywords → Abstract should have ~2 blank lines
            elif curr_type == 'Keywords' and next_type == 'Abstract':
                if gap < 18:
                    spacing_issues.append({
                        'from': curr_type,
                        'to': next_type,
                        'gap': gap,
                        'expected': '2 blank lines (~24pt)'
                    })

        if spacing_issues:
            issue = spacing_issues[0]
            self.results.append(ValidationResult(
                check_name="Section Spacing",
                passed=False,
                severity=Severity.WARNING,
                message=f"Insufficient spacing between {issue['from']} and {issue['to']}",
                details=f"Found {issue['gap']:.0f}pt gap, expected {issue['expected']}"
            ))
        else:
            self.results.append(ValidationResult(
                check_name="Section Spacing",
                passed=True,
                severity=Severity.SUCCESS,
                message="Section spacing appears correct",
                details="Checked spacing between major sections"
            ))

    @staticmethod
    def format_results_for_console(
        overall_pass: bool,
        results: List[ValidationResult]
    ) -> str:
        """
        Format validation results for console output.

        Args:
            overall_pass: Overall validation result
            results: List of validation results

        Returns:
            Formatted string for console display
        """
        lines = []

        # Header
        lines.append("=" * 70)
        lines.append("COMPLIANCE VALIDATION RESULTS")
        lines.append("=" * 70)
        lines.append("")

        # Overall result
        status = "PASSED" if overall_pass else "FAILED"
        status_symbol = "✓" if overall_pass else "✗"
        lines.append(f"Overall Status: {status_symbol} {status}")
        lines.append("")

        # Group results by severity
        errors = [r for r in results if r.severity == Severity.WARNING and not r.passed]
        warnings = [r for r in results if r.severity == Severity.WARNING and not r.passed]
        successes = [r for r in results if r.passed]
        infos = [r for r in results if r.severity == Severity.INFO]

        # Show errors
        if errors:
            lines.append("ERRORS:")
            lines.append("-" * 70)
            for result in errors:
                lines.append(f"  ✗ {result.check_name}")
                lines.append(f"    {result.message}")
                if result.details:
                    lines.append(f"    Details: {result.details}")
                lines.append("")

        # Show warnings
        if warnings:
            lines.append("WARNINGS:")
            lines.append("-" * 70)
            for result in warnings:
                lines.append(f"  ⚠ {result.check_name}")
                lines.append(f"    {result.message}")
                if result.details:
                    lines.append(f"    Details: {result.details}")
                lines.append("")

        # Show successes (summary)
        if successes:
            lines.append("PASSED CHECKS:")
            lines.append("-" * 70)
            for result in successes:
                lines.append(f"  ✓ {result.check_name}")
            lines.append("")

        # Show informational
        if infos:
            lines.append("ADDITIONAL INFORMATION:")
            lines.append("-" * 70)
            for result in infos:
                lines.append(f"  ℹ {result.check_name}")
                lines.append(f"    {result.message}")
                if result.details:
                    lines.append(f"    Details: {result.details}")
            lines.append("")

        # Summary statistics
        lines.append("=" * 70)
        lines.append("SUMMARY:")
        lines.append(f"  Total Checks: {len(results)}")
        lines.append(f"  Errors: {len(errors)}")
        lines.append(f"  Warnings: {len(warnings)}")
        lines.append(f"  Passed: {len(successes)}")
        lines.append(f"  Info: {len(infos)}")
        lines.append("=" * 70)

        return "\n".join(lines)
