#!/usr/bin/env python3
"""
PDF Compliance Analyzer - app_3
Hybrid approach combining Google Cloud Document AI and PyMuPDF.

Usage:
    python main.py <path_to_document> [--output <json_path>] [--credentials <path>] [--anon] [--report]
"""

import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import asdict

# Import support modules
from support.converter import DocumentConverter
from support.document_ai_client import DocumentAIClient, ExtractedElement
from support.pymupdf_extractor import PyMuPDFExtractor, EnrichedElement
from support.validator import ComplianceValidator
from support.output_generator import OutputGenerator
from support.citation_validator import CitationValidator
from support.anonymization_checker import AnonymizationChecker
from support.report_generator import generate_report


class ProgressIndicator:
    """Simple progress indicator for long operations."""

    def __init__(self, message: str):
        self.message = message

    def __enter__(self):
        print(f"{self.message}...", end=" ", flush=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            print("Done!")
        else:
            print("Failed!")


class PDFComplianceAnalyzer:
    """Main analyzer combining Document AI and PyMuPDF."""

    # Class-level constants for heading detection
    SPECIAL_HEADINGS = ['acknowledgements', 'references']
    MAX_HEADING_LENGTH = 100

    # Element types that should block gap scanning (figures, tables, equations)
    BLOCKING_ELEMENT_TYPES = {
        'Figure', 'Figure_Title', 'Figure_Number',
        'Table', 'Table_Title', 'Table_Number',
        'Equation', 'Equation_Number'
    }

    # Minimum vertical gap between references to trigger PyMuPDF scanning (points)
    MIN_GAP_HEIGHT = 18  # ~1.5 lines of 9pt reference text

    # Minimum Document AI confidence to include an extracted element
    MIN_CONFIDENCE = 0.5

    # Compiled regex patterns for heading classification
    SUB_SUB_HEADING_PATTERN = re.compile(r'^\d+\.\d+\.\d+\.?\s+')
    SUB_HEADING_PATTERN = re.compile(r'^\d+\.\d+\.?\s+')
    HEADING_PATTERN = re.compile(r'^\d+\.\s+')
    # Header section types that should not be checked for two-column layout
    # and should not have Main_Text duplicates
    HEADER_SECTION_TYPES = {'Title', 'Authors', 'Affiliations', 'Keywords', 'Abstract', 'Abstract_title', 'Keywords_title'}

    def __init__(
        self,
        credentials_path: str = None,
        credentials_info: dict = None,
        check_anonymization: bool = False,
        generate_report: bool = False
    ):
        """
        Initialize analyzer.

        Args:
            credentials_path: Path to Google Cloud credentials
            credentials_info: GCP credentials as dict (for Streamlit secrets)
            check_anonymization: Whether to check for anonymization
            generate_report: Whether to generate annotated PDF report
        """
        self.credentials_path = credentials_path
        self.credentials_info = credentials_info
        self.check_anonymization = check_anonymization
        self.generate_report_flag = generate_report

    def analyze(self, document_path: str, output_path: str = None) -> Dict:
        """
        Analyze a document for compliance.

        Args:
            document_path: Path to document (PDF, DOCX, or TEX)
            output_path: Optional path for JSON output

        Returns:
            Analysis results dictionary
        """
        print("\n" + "=" * 70)
        print("PDF COMPLIANCE ANALYZER - Hybrid Approach")
        print("=" * 70 + "\n")

        # Step 1: Convert to PDF if needed
        input_ext = Path(document_path).suffix.lower()
        if input_ext == '.pdf':
            print("Input is PDF - no conversion needed")
            pdf_path = document_path
        else:
            with ProgressIndicator(f"Converting {input_ext} to PDF"):
                pdf_path = DocumentConverter.convert_to_pdf(document_path)

        print(f"Analyzing: {pdf_path}\n")

        # Step 2: Extract using Document AI
        with ProgressIndicator("Extracting text and bounding boxes (Document AI)"):
            doc_ai_client = DocumentAIClient(
                credentials_path=self.credentials_path,
                credentials_info=self.credentials_info
            )
            doc_ai_document = doc_ai_client.process_document(pdf_path)
            ai_elements = doc_ai_client.extract_elements_by_type(doc_ai_document)

        # Step 3: Enrich with PyMuPDF font information
        with ProgressIndicator("Extracting font and style information (PyMuPDF)"):
            enriched_elements = self._enrich_with_pymupdf(ai_elements, pdf_path)

        # Step 3.1: Filter duplicate Main_Text elements that overlap header sections
        with ProgressIndicator("Filtering duplicate Main_Text in header regions"):
            enriched_elements = self._filter_main_text_duplicates(enriched_elements)

        # Step 3.5: Reclassify headings based on numbering
        with ProgressIndicator("Reclassifying headings based on numbering"):
            enriched_elements = self._reclassify_headings(enriched_elements)

        # Step 3.7: Merge partial references (column/page spanning)
        with ProgressIndicator("Merging partial references"):
            enriched_elements = self._merge_partial_references(enriched_elements, pdf_path)

        # Step 3.8: Merge column-spanning references (regular References at boundaries)
        with ProgressIndicator("Merging column-spanning references"):
            enriched_elements = self._merge_column_spanning_references(enriched_elements, pdf_path)

        # Step 3.9: Detect references in gaps via PyMuPDF scanning
        with ProgressIndicator("Scanning for missed references in gaps"):
            enriched_elements = self._detect_references_in_gaps(enriched_elements, pdf_path)

        # Step 4: Verify special labels (Keywords, Abstract)
        with ProgressIndicator("Verifying special labels"):
            labels_verified = self._verify_labels(enriched_elements, pdf_path)

        print(f"  Keywords label verified: {labels_verified.get('Keywords', False)}")
        print(f"  Abstract label verified: {labels_verified.get('Abstract', False)}")

        # Step 5: Validate citations and references
        with ProgressIndicator("Validating citations and references"):
            citation_validator = CitationValidator()
            citation_results = citation_validator.validate_citations_and_references(
                enriched_elements
            )
            figure_table_results = citation_validator.validate_figure_table_citations(
                enriched_elements
            )

        print(f"  Found {len(citation_results.get('references_parsed', []))} references")
        print(f"  Found {len(citation_results.get('citations_parsed', []))} citations")
        print(f"  Orphan citations: {len(citation_results.get('orphan_citations', []))}")
        print(f"  Uncited references: {len(citation_results.get('uncited_references', []))}")

        # Step 6: Check anonymization (if requested)
        anonymization_result = None
        if self.check_anonymization:
            with ProgressIndicator("Checking anonymization"):
                anon_checker = AnonymizationChecker()
                anonymization_result = anon_checker.check_anonymization(
                    enriched_elements,
                    sections_to_check=['Authors', 'Affiliations']
                )

            print(f"  Anonymized: {anonymization_result.is_anonymous}")
            print(f"  PERSON entities: {anonymization_result.total_person_entities}")
            print(f"  ORG entities: {anonymization_result.total_org_entities}")

        # Step 7: Get page info for ISPRS validation
        with ProgressIndicator("Getting page information"):
            with PyMuPDFExtractor(pdf_path) as extractor:
                page_count = len(extractor.doc)
                page_dimensions = extractor.get_page_dimensions(0)  # First page

        print(f"  Page count: {page_count}")
        print(f"  Page dimensions: {page_dimensions[0]:.0f}×{page_dimensions[1]:.0f}pt")

        # Step 8: Validate compliance
        with ProgressIndicator("Validating compliance"):
            validator = ComplianceValidator()
            overall_pass, validation_results = validator.validate(
                enriched_elements,
                labels_verified,
                citation_results=citation_results,
                figure_table_results=figure_table_results,
                anonymization_result=anonymization_result,
                page_count=page_count,
                page_dimensions=page_dimensions
            )

        # Step 9: Display results
        print("\n")
        result_text = validator.format_results_for_console(
            overall_pass,
            validation_results
        )
        print(result_text)

        # Step 10: Display element summary
        summary = OutputGenerator.format_element_summary(enriched_elements)
        print(summary)

        # Step 11: Generate JSON output
        json_output = OutputGenerator.generate_json_output(
            pdf_path=pdf_path,
            extracted_elements=enriched_elements,
            validation_results=validation_results,
            overall_pass=overall_pass,
            labels_verified=labels_verified
        )

        # Add citation and anonymization results to JSON (convert dataclasses to dicts)
        json_output['citation_validation'] = self._serialize_citation_results(citation_results)
        json_output['figure_table_validation'] = figure_table_results
        if anonymization_result:
            json_output['anonymization'] = {
                'is_anonymous': anonymization_result.is_anonymous,
                'total_person_entities': anonymization_result.total_person_entities,
                'total_org_entities': anonymization_result.total_org_entities,
                'total_gpe_entities': anonymization_result.total_gpe_entities,
                'sections_checked': anonymization_result.sections_checked,
                'entities_found': [
                    {
                        'text': e.text,
                        'label': e.label,
                        'start': e.start,
                        'end': e.end
                    }
                    for e in anonymization_result.entities_found
                ]
            }

        # Step 12: Save JSON output
        if output_path is None:
            output_path = OutputGenerator.get_default_output_path(pdf_path)

        with ProgressIndicator(f"\nSaving JSON output to {output_path}"):
            OutputGenerator.save_json(json_output, output_path)

        # Step 13: Generate PDF report (if requested)
        report_path = None
        if self.generate_report_flag:
            with ProgressIndicator("Generating annotated PDF report"):
                report_path = generate_report(
                    pdf_path=pdf_path,
                    validation_results=validation_results,
                    enriched_elements=enriched_elements,
                    overall_pass=overall_pass
                )

        print("\n" + "=" * 70)
        print(f"Analysis complete! JSON output saved to: {output_path}")
        if report_path:
            print(f"PDF report generated: {report_path}")
        print("=" * 70 + "\n")

        return json_output

    def _enrich_with_pymupdf(
        self,
        ai_elements: Dict[str, List[ExtractedElement]],
        pdf_path: str
    ) -> Dict[str, List[EnrichedElement]]:
        """
        Enrich Document AI elements with PyMuPDF font information.

        Args:
            ai_elements: Elements from Document AI
            pdf_path: Path to PDF file

        Returns:
            Dictionary of enriched elements
        """
        enriched = {}

        with PyMuPDFExtractor(pdf_path) as extractor:
            # Enrich each element type
            for element_type, elements in ai_elements.items():
                enriched[element_type] = []

                for element in elements:
                    # Skip low-confidence elements from Document AI
                    if element.confidence < self.MIN_CONFIDENCE:
                        continue

                    enriched_element = extractor.enrich_element(
                        element_type=element_type,
                        text=element.text,
                        bbox=(
                            element.bbox.x0,
                            element.bbox.y0,
                            element.bbox.x1,
                            element.bbox.y1
                        ),
                        page=element.bbox.page
                    )
                    enriched[element_type].append(enriched_element)

        return enriched

    def _verify_labels(
        self,
        enriched_elements: Dict[str, List[EnrichedElement]],
        pdf_path: str
    ) -> Dict[str, Any]:
        """
        Verify that special labels (Keywords, Abstract) exist.

        Primary: use Abstract_title / Keywords_title elements from Document AI.
        Fallback: search with PyMuPDF to the left of Keywords and above Abstract.

        Args:
            enriched_elements: Enriched document elements
            pdf_path: Path to PDF file (for PyMuPDF fallback)

        Returns:
            Dictionary of label verification results with keys:
            - "Keywords": bool (found)
            - "Keywords_format_warning": str or None (warning if incorrect format)
            - "Abstract": bool (found)
        """
        labels_verified = {
            "Keywords": False,
            "Keywords_format_warning": None,
            "Keywords_element_ref": None,  # (page, bbox) for highlighting
            "Abstract": False,
            "Abstract_format_warning": None,
            "Abstract_element_ref": None  # (page, bbox) for highlighting
        }

        # Check Keywords_title element
        keywords_title_elements = enriched_elements.get("Keywords_title", [])
        if keywords_title_elements:
            title_elem = keywords_title_elements[0]
            labels_verified["Keywords"] = True
            labels_verified["Keywords_element_ref"] = (title_elem.page, title_elem.bbox)

            # Format validation: should be "Keywords" (not "KEYWORDS", "key words", etc.)
            title_text = title_elem.text.strip().rstrip(':')
            if title_text != "Keywords":
                labels_verified["Keywords_format_warning"] = (
                    f"Found '{title_text}' - should be 'Keywords'"
                )
        else:
            # Fallback: search to the left of the Keywords content element
            keywords_elements = enriched_elements.get("Keywords", [])
            if keywords_elements:
                first_element = keywords_elements[0]
                labels_verified["Keywords_element_ref"] = (
                    first_element.page,
                    first_element.bbox
                )
                found, actual_text, is_correct = self._search_for_label(
                    pdf_path, "Keywords", first_element, "left"
                )
                if found:
                    labels_verified["Keywords"] = True
                    if not is_correct:
                        labels_verified["Keywords_format_warning"] = (
                            f"Found '{actual_text}' - should be 'Keywords'"
                        )

        # Check Abstract_title element
        abstract_title_elements = enriched_elements.get("Abstract_title", [])
        if abstract_title_elements:
            title_elem = abstract_title_elements[0]
            labels_verified["Abstract"] = True
            labels_verified["Abstract_element_ref"] = (title_elem.page, title_elem.bbox)

            # Format validation: should be "Abstract" (not "ABSTRACT", etc.)
            title_text = title_elem.text.strip().rstrip(':')
            if title_text != "Abstract":
                labels_verified["Abstract_format_warning"] = (
                    f"Found '{title_text}' - should be 'Abstract'"
                )
        else:
            # Fallback: search above the Abstract content element
            abstract_elements = enriched_elements.get("Abstract", [])
            if abstract_elements:
                first_element = abstract_elements[0]
                labels_verified["Abstract_element_ref"] = (
                    first_element.page,
                    first_element.bbox
                )
                found, actual_text, is_correct = self._search_for_label(
                    pdf_path, "Abstract", first_element, "above"
                )
                if found:
                    labels_verified["Abstract"] = True
                    if not is_correct:
                        labels_verified["Abstract_format_warning"] = (
                            f"Found '{actual_text}' - should be 'Abstract'"
                        )

        return labels_verified

    def _search_for_label(
        self,
        pdf_path: str,
        label_text: str,
        content_element: EnrichedElement,
        direction: str
    ) -> tuple:
        """
        Search for a label near a content element using PyMuPDF.

        Args:
            pdf_path: Path to PDF file
            label_text: Expected label (e.g., "Keywords", "Abstract")
            content_element: The content element whose label to find
            direction: "left" or "above"

        Returns:
            Tuple of (found, actual_text, is_correct_format)
        """
        search_margin = 100  # points

        # content_element.bbox is already in absolute coordinates
        x0, y0, x1, y1 = content_element.bbox
        page = content_element.page

        with PyMuPDFExtractor(pdf_path) as extractor:
            page_width, page_height = extractor.get_page_dimensions(page)
            if page_width == 0:
                return (False, "", True)

            # First check inside the content element's own bbox.
            # Document AI sometimes returns a bbox that already covers the
            # label (e.g. "Keywords" header included in the Keywords bbox).
            # In that case searching outside the bbox would miss it.
            regions_to_check = [
                (x0, y0, x1, y1),  # inside the content bbox itself
            ]

            # Then check the directional region outside the bbox
            if direction == "left":
                regions_to_check.append(
                    (max(0, x0 - search_margin), y0, x0, y1)
                )
            elif direction == "above":
                regions_to_check.append(
                    (x0, max(0, y0 - search_margin), x1, y0)
                )

            label_lower = label_text.lower()
            label_upper = label_text.upper()

            for search_bbox in regions_to_check:
                spans = extractor.extract_text_from_bbox(page, search_bbox)
                text = ' '.join(s.text for s in spans).strip()

                if not text:
                    continue

                text_lower = text.lower()

                # Check for correct format (e.g., "Keywords", "Abstract")
                if label_text in text or f"{label_text}:" in text:
                    return (True, label_text, True)

                # Check for ALL CAPS (e.g., "KEYWORDS", "ABSTRACT")
                if label_upper in text or f"{label_upper}:" in text:
                    return (True, label_upper, False)

                # Check for any case variation
                if label_lower in text_lower or f"{label_lower}:" in text_lower:
                    return (True, text_lower, False)

                # Check for "key words" / "key-words" variations
                if label_lower == "keywords":
                    for pattern in ["key word", "key words", "key-word",
                                    "key-words"]:
                        if pattern in text_lower:
                            return (True, pattern, False)

            return (False, "", True)

    def _reclassify_headings(
        self,
        extracted_elements: Dict[str, List]
    ) -> Dict[str, List]:
        """
        Reclassify heading elements based on numbering patterns ONLY.

        Patterns:
        - Headings: "N. Text" (e.g., "1. Introduction")
        - Sub_Headings: "N.N Text" (e.g., "2.1 Methods")
        - Sub_sub_Headings: "N.N.N Text" (e.g., "2.1.1 Details")
        - Special: "Acknowledgements", "References" (no numbers)

        Trusts Document AI for all non-heading categories.

        Args:
            extracted_elements: Elements from Document AI

        Returns:
            Reclassified elements
        """
        # Collect all heading-related elements
        all_heading_elements = []
        original_categories = {}

        for key in ['Headings', 'Sub_Headings', 'Sub_sub_Headings']:
            for elem in extracted_elements.get(key, []):
                all_heading_elements.append(elem)
                original_categories[id(elem)] = key

        # Initialize reclassified buckets
        reclassified = {
            'Headings': [],
            'Sub_Headings': [],
            'Sub_sub_Headings': []
        }

        # Reclassify based on patterns only (using class-level patterns)
        for elem in all_heading_elements:
            text = elem.text.strip()
            text_lower = text.lower()

            # Check patterns in order (most specific first)
            # Add length check to filter out body text that starts with numbers
            if self.SUB_SUB_HEADING_PATTERN.match(text) and len(text) <= self.MAX_HEADING_LENGTH:
                reclassified['Sub_sub_Headings'].append(elem)
                elem.element_type = 'Sub_sub_Headings'
            elif self.SUB_HEADING_PATTERN.match(text) and len(text) <= self.MAX_HEADING_LENGTH:
                reclassified['Sub_Headings'].append(elem)
                elem.element_type = 'Sub_Headings'
            elif self.HEADING_PATTERN.match(text) and len(text) <= self.MAX_HEADING_LENGTH:
                reclassified['Headings'].append(elem)
                elem.element_type = 'Headings'
            elif len(text) <= 50 and any(text_lower.strip() == special or text_lower.strip().startswith(special) for special in self.SPECIAL_HEADINGS):
                # Only match short text that IS or starts with special heading word
                # "References" (11 chars) = heading, "References Aynechi..." (500+ chars) = not heading
                reclassified['Headings'].append(elem)
                elem.element_type = 'Headings'
            else:
                # No pattern matched - this element doesn't belong in heading categories
                # Drop it (Document AI misclassified it)
                orig = original_categories.get(id(elem), '?')
                print(f"  [Heading Reclassify] Dropped (was {orig}, len={len(text)}): '{text[:80]}'")
                continue

        # Update extracted_elements
        extracted_elements['Headings'] = reclassified['Headings']
        extracted_elements['Sub_Headings'] = reclassified['Sub_Headings']
        extracted_elements['Sub_sub_Headings'] = reclassified['Sub_sub_Headings']

        # Print summary
        print(f"  [Reclassify] Summary: {len(reclassified['Headings'])} Headings, "
              f"{len(reclassified['Sub_Headings'])} Sub_Headings, "
              f"{len(reclassified['Sub_sub_Headings'])} Sub_sub_Headings")

        return extracted_elements

    def _filter_main_text_duplicates(
        self,
        enriched_elements: Dict[str, List]
    ) -> Dict[str, List]:
        """
        Remove Main_Text elements that:
        1. Significantly overlap with header section elements
        2. Are outside the page content area (document header/footer regions)

        This prevents duplicate validation warnings and excludes document chrome.

        Args:
            enriched_elements: Dictionary of enriched elements

        Returns:
            Updated enriched_elements with filtered Main_Text
        """
        main_text = enriched_elements.get('Main_Text', [])
        if not main_text:
            return enriched_elements

        # Page margin constants (matching validator.py PAGE_REQUIREMENTS)
        top_margin = 71  # 25mm in points
        bottom_margin = 71  # 25mm in points
        page_height = 842  # A4 height in points
        margin_tolerance = 10  # Tolerance for margin checks

        # Collect header section bboxes
        header_bboxes = []
        for elem_type in self.HEADER_SECTION_TYPES:
            for elem in enriched_elements.get(elem_type, []):
                if hasattr(elem, 'bbox') and hasattr(elem, 'page'):
                    header_bboxes.append({
                        'page': elem.page,
                        'bbox': elem.bbox,
                        'type': elem_type
                    })

        # Collect blocking element bboxes (equations, figures, tables)
        # Main_Text overlapping these should be filtered as duplicates
        blocking_bboxes = []
        for elem_type in self.BLOCKING_ELEMENT_TYPES:
            for elem in enriched_elements.get(elem_type, []):
                if hasattr(elem, 'bbox') and hasattr(elem, 'page'):
                    blocking_bboxes.append({
                        'page': elem.page,
                        'bbox': elem.bbox,
                        'type': elem_type
                    })

        # Filter Main_Text elements
        filtered_main_text = []
        for mt in main_text:
            if not hasattr(mt, 'bbox') or not hasattr(mt, 'page'):
                filtered_main_text.append(mt)
                continue

            x0, y0, x1, y1 = mt.bbox

            # Filter 1: Check if element is outside content area (header/footer region)
            # Skip elements above top margin or below bottom margin
            if y0 < (top_margin - margin_tolerance) or y1 > (page_height - bottom_margin + margin_tolerance):
                print(f"  [Main_Text Filter] Removed outside content area: '{mt.text[:40]}...'")
                continue

            # Filter 2: Check overlap with header sections
            overlaps_header = False
            for header in header_bboxes:
                if header['page'] != mt.page:
                    continue

                # Calculate overlap
                hx0, hy0, hx1, hy1 = header['bbox']

                int_x0 = max(x0, hx0)
                int_y0 = max(y0, hy0)
                int_x1 = min(x1, hx1)
                int_y1 = min(y1, hy1)

                if int_x0 < int_x1 and int_y0 < int_y1:
                    # Calculate overlap ratio based on smaller area
                    int_area = (int_x1 - int_x0) * (int_y1 - int_y0)
                    mt_area = (x1 - x0) * (y1 - y0)
                    h_area = (hx1 - hx0) * (hy1 - hy0)
                    smaller_area = min(mt_area, h_area)

                    if smaller_area > 0 and int_area / smaller_area >= 0.5:
                        overlaps_header = True
                        print(f"  [Main_Text Filter] Removed duplicate overlapping {header['type']}: '{mt.text[:40]}...'")
                        break

            if overlaps_header:
                continue

            # Filter 3: Check overlap with blocking elements (equations, figures, tables)
            overlaps_blocking = False
            for blocker in blocking_bboxes:
                if blocker['page'] != mt.page:
                    continue

                # Calculate overlap
                bx0, by0, bx1, by1 = blocker['bbox']

                int_x0 = max(x0, bx0)
                int_y0 = max(y0, by0)
                int_x1 = min(x1, bx1)
                int_y1 = min(y1, by1)

                if int_x0 < int_x1 and int_y0 < int_y1:
                    # Calculate overlap ratio based on smaller area
                    int_area = (int_x1 - int_x0) * (int_y1 - int_y0)
                    mt_area = (x1 - x0) * (y1 - y0)
                    b_area = (bx1 - bx0) * (by1 - by0)
                    smaller_area = min(mt_area, b_area)

                    if smaller_area > 0 and int_area / smaller_area >= 0.3:  # Lower threshold for blocking elements
                        overlaps_blocking = True
                        print(f"  [Main_Text Filter] Removed duplicate overlapping {blocker['type']}: '{mt.text[:40]}...'")
                        break

            if not overlaps_blocking:
                filtered_main_text.append(mt)

        enriched_elements['Main_Text'] = filtered_main_text
        return enriched_elements

    def _merge_partial_references(
        self,
        enriched_elements: Dict[str, List],
        pdf_path: str
    ) -> Dict[str, List]:
        """
        Merge Reference_Partial elements that span columns or pages.

        Partials are classified by position (page, column, vertical region) and
        paired with their expected partner. Unpaired partials go through fallback
        strategies before being added as solo References.

        Args:
            enriched_elements: Dictionary of enriched elements
            pdf_path: Path to PDF file

        Returns:
            Updated enriched_elements with merged references added
        """
        partials = enriched_elements.get('Reference_Partial', [])
        if not partials:
            return enriched_elements

        with PyMuPDFExtractor(pdf_path) as extractor:
            # Get page dimensions
            page_dims = {}
            for p in partials:
                if p.page not in page_dims:
                    page_dims[p.page] = extractor.get_page_dimensions(p.page)
            # Also gather dims for Reference elements (needed for fallback)
            for ref in enriched_elements.get('References', []):
                if ref.page not in page_dims:
                    page_dims[ref.page] = extractor.get_page_dimensions(ref.page)

            # Classify each partial: (page, column, position)
            # column: 0=left, 1=right
            # position: determined by TEXT CONTENT, not Y coordinate.
            #   A reference start (has author+year) is at the 'bottom' of a column
            #   (last entry before the column boundary). A continuation fragment
            #   (no author+year) is at the 'top' of the next column.
            #   This is robust for pages where columns don't span the full height
            #   (e.g., last page with references ending at the page midpoint).
            def classify(elem):
                page_width, _ = page_dims.get(elem.page, (595, 842))
                mid_x = page_width / 2
                x_center = (elem.bbox[0] + elem.bbox[2]) / 2
                column = 0 if x_center < mid_x else 1
                position = 'top' if self._is_reference_fragment(elem.text) else 'bottom'
                return (elem.page, column, position)

            classified = [(p, classify(p)) for p in partials]

            # Sort by reading order: (page, column, y)
            def reading_key(item):
                elem, (pg, col, _) = item
                return (pg, col, elem.bbox[1])

            classified.sort(key=reading_key)

            # Build a lookup for quick partner search
            # Key: (page, column, position) -> list of (elem, classification)
            by_location = {}
            for item in classified:
                _, loc = item
                by_location.setdefault(loc, []).append(item)

            merged_indices = set()  # Track indices of partials already merged
            new_references = []

            for idx, (partial, (pg, col, pos)) in enumerate(classified):
                if idx in merged_indices:
                    continue

                # Determine expected partner location
                partner_loc = None
                if pos == 'bottom' and col == 0:
                    partner_loc = (pg, 1, 'top')  # Top-right same page
                elif pos == 'bottom' and col == 1:
                    partner_loc = (pg + 1, 0, 'top')  # Top-left next page
                elif pos == 'top' and col == 1:
                    partner_loc = (pg, 0, 'bottom')  # Bottom-left same page (reverse)
                elif pos == 'top' and col == 0:
                    partner_loc = (pg - 1, 1, 'bottom')  # Bottom-right prev page (reverse)

                # Search for partner in other partials
                partner_found = False
                if partner_loc and partner_loc in by_location:
                    for partner_idx_offset, (candidate, _) in enumerate(by_location[partner_loc]):
                        # Find the global index of this candidate
                        cand_global_idx = None
                        for gi, (ce, _) in enumerate(classified):
                            if ce is candidate:
                                cand_global_idx = gi
                                break
                        if cand_global_idx is not None and cand_global_idx not in merged_indices:
                            # Found a partner — merge (bottom fragment text first)
                            if pos == 'bottom':
                                merged_text = partial.text.rstrip() + ' ' + candidate.text.lstrip()
                                base_elem = partial
                            else:
                                merged_text = candidate.text.rstrip() + ' ' + partial.text.lstrip()
                                base_elem = candidate

                            page_width, page_height = page_dims.get(base_elem.page, (595, 842))
                            norm_bbox = (
                                base_elem.bbox[0] / page_width,
                                base_elem.bbox[1] / page_height,
                                base_elem.bbox[2] / page_width,
                                base_elem.bbox[3] / page_height
                            )
                            new_elem = extractor.enrich_element(
                                'References', merged_text, norm_bbox, base_elem.page
                            )
                            new_references.append(new_elem)
                            merged_indices.add(idx)
                            merged_indices.add(cand_global_idx)
                            partner_found = True
                            print(f"  [Partial Merge] Paired partials: '{partial.text[:40]}...' + '{candidate.text[:40]}...'")
                            break

                if not partner_found:
                    # Try fallback strategies
                    fallback_elem = self._find_partial_partner_fallback(
                        partial, pg, col, pos, partner_loc,
                        enriched_elements, extractor, page_dims
                    )
                    if fallback_elem:
                        new_references.append(fallback_elem)
                    else:
                        # Last resort: add solo partial as Reference
                        page_width, page_height = page_dims.get(pg, (595, 842))
                        norm_bbox = (
                            partial.bbox[0] / page_width,
                            partial.bbox[1] / page_height,
                            partial.bbox[2] / page_width,
                            partial.bbox[3] / page_height
                        )
                        solo_elem = extractor.enrich_element(
                            'References', partial.text, norm_bbox, pg
                        )
                        new_references.append(solo_elem)
                        print(f"  [Partial Merge] Solo partial added as Reference: '{partial.text[:60]}...'")
                    merged_indices.add(idx)

            # Add merged references to References list
            enriched_elements['References'].extend(new_references)

            # Clear processed partials
            enriched_elements['Reference_Partial'] = []

        return enriched_elements

    def _find_partial_partner_fallback(
        self,
        partial,
        page: int,
        column: int,
        position: str,
        partner_loc,
        enriched_elements: Dict[str, List],
        extractor,
        page_dims: Dict[int, tuple]
    ):
        """
        Fallback strategies when a partial has no matching pair among other partials.

        Document AI explicitly flagged this element as Reference_Partial, so we
        know the other half exists. Two strategies to find it:

        Strategy 1: The other half was misclassified as a regular Reference.
                    Search existing References at the expected position.
        Strategy 2: The other half has no bbox from Document AI at all.
                    Scan the expected region with PyMuPDF to extract the text.

        Args:
            partial: The unmatched partial element
            page: Partial's page number
            column: Partial's column (0=left, 1=right)
            position: Partial's vertical position ('top' or 'bottom')
            partner_loc: Expected partner (page, column, position) tuple
            enriched_elements: Full element dictionary
            extractor: Open PyMuPDFExtractor instance
            page_dims: Page dimensions lookup

        Returns:
            Merged EnrichedElement if a partner was found, None otherwise
        """
        if not partner_loc:
            return None

        target_page, target_col, target_pos = partner_loc
        if target_page not in page_dims:
            # Target page doesn't exist (e.g., partial on first page looking backward)
            return None

        target_page_width, target_page_height = page_dims[target_page]
        mid_x = target_page_width / 2
        mid_y = target_page_height / 2
        left_margin = 57  # 20mm in points

        # Define the expected region bbox for the partner
        if target_col == 0:
            region_x0 = left_margin
            region_x1 = mid_x - 3  # Column gap / 2
        else:
            region_x0 = mid_x + 3
            region_x1 = target_page_width - left_margin

        if target_pos == 'top':
            region_y0 = 71  # Top margin (25mm)
            region_y1 = mid_y
        else:
            region_y0 = mid_y
            region_y1 = target_page_height - 71  # Bottom margin

        # Strategy 1: The other half was misclassified as a regular Reference.
        # Find References in the target column on the target page, then pick the
        # topmost (if target_pos='top') or bottommost (if target_pos='bottom').
        # This avoids the fragile 50% midpoint classification — a Reference near
        # the top of a column is the partner regardless of its exact Y coordinate.
        references = enriched_elements.get('References', [])
        candidates = []
        for ref_idx, ref in enumerate(references):
            if ref.page != target_page:
                continue
            ref_x_center = (ref.bbox[0] + ref.bbox[2]) / 2
            ref_col = 0 if ref_x_center < mid_x else 1
            if ref_col == target_col:
                candidates.append((ref_idx, ref))

        if candidates:
            # Sort by Y position
            candidates.sort(key=lambda item: item[1].bbox[1])

            if target_pos == 'top':
                # Partner should be at the top of the column — pick topmost Reference
                ref_idx, ref = candidates[0]
            else:
                # Partner should be at the bottom — pick bottommost Reference
                ref_idx, ref = candidates[-1]

            # Only merge if the partial and this Reference are complementary halves.
            # At least one must be a fragment (continuation without author+year).
            # If both look like reference starts, they can't be two halves of the
            # same reference — skip to Strategy 2.
            partial_is_fragment = self._is_reference_fragment(partial.text)
            ref_is_fragment = self._is_reference_fragment(ref.text)
            if partial_is_fragment or ref_is_fragment:
                # Merge
                if position == 'bottom':
                    merged_text = partial.text.rstrip() + ' ' + ref.text.lstrip()
                    base_elem = partial
                else:
                    merged_text = ref.text.rstrip() + ' ' + partial.text.lstrip()
                    base_elem = ref

                base_page_width, base_page_height = page_dims.get(base_elem.page, (595, 842))
                norm_bbox = (
                    base_elem.bbox[0] / base_page_width,
                    base_elem.bbox[1] / base_page_height,
                    base_elem.bbox[2] / base_page_width,
                    base_elem.bbox[3] / base_page_height
                )
                new_elem = extractor.enrich_element(
                    'References', merged_text, norm_bbox, base_elem.page
                )
                # Remove the original Reference to avoid duplication
                enriched_elements['References'].pop(ref_idx)
                print(f"  [Partial Merge] Fallback: merged with existing Reference in target column")
                return new_elem

        # Strategy 2: No matching Reference found — scan the gap region.
        # Narrow the scan to just the gap (top of column to first Reference, or
        # last Reference to bottom of column) to avoid capturing other references.
        # Find the nearest Reference boundary in the target column on the target page.
        all_refs_on_page = []
        for ref in references:
            if ref.page != target_page:
                continue
            ref_x_center = (ref.bbox[0] + ref.bbox[2]) / 2
            ref_col = 0 if ref_x_center < mid_x else 1
            if ref_col == target_col:
                all_refs_on_page.append(ref)

        # Refine scan region to only the gap (avoid capturing existing references)
        scan_y0 = region_y0
        scan_y1 = region_y1
        if all_refs_on_page:
            all_refs_on_page.sort(key=lambda r: r.bbox[1])
            if target_pos == 'top':
                # Scan from top margin to the top of the first Reference
                scan_y1 = min(region_y1, all_refs_on_page[0].bbox[1])
            else:
                # Scan from the bottom of the last Reference to bottom margin
                scan_y0 = max(region_y0, all_refs_on_page[-1].bbox[3])

        # Must have meaningful gap height (at least ~1 line of text)
        if scan_y1 - scan_y0 < 10:
            return None

        abs_bbox = (region_x0, scan_y0, region_x1, scan_y1)
        spans = extractor.extract_text_from_bbox(target_page, abs_bbox)
        if spans:
            gap_text = ' '.join(s.text for s in spans).strip()
            if len(gap_text) >= 10:
                if position == 'bottom':
                    merged_text = partial.text.rstrip() + ' ' + gap_text.lstrip()
                else:
                    merged_text = gap_text.rstrip() + ' ' + partial.text.lstrip()

                partial_page_width, partial_page_height = page_dims.get(partial.page, (595, 842))
                norm_bbox = (
                    partial.bbox[0] / partial_page_width,
                    partial.bbox[1] / partial_page_height,
                    partial.bbox[2] / partial_page_width,
                    partial.bbox[3] / partial_page_height
                )
                new_elem = extractor.enrich_element(
                    'References', merged_text, norm_bbox, partial.page
                )
                print(f"  [Partial Merge] Fallback: merged with text scanned from gap region")
                return new_elem

        return None

    def _is_reference_complete(self, text: str) -> bool:
        """
        Check if reference text ends with a pattern indicating a complete reference.

        Used by _merge_column_spanning_references() to decide whether the last
        reference at a column/page boundary needs continuation.

        Args:
            text: The reference text to check

        Returns:
            True if text appears to end with a complete-reference pattern
        """
        text = text.strip()
        if not text:
            return False

        # Page range: "45-67" or "123–456"
        if re.search(r'\d+[-\u2013]\d+\.?\s*$', text):
            return True
        # DOI
        if re.search(r'doi[:\s][^\s]+\.?\s*$', text, re.IGNORECASE):
            return True
        # URL
        if re.search(r'https?://[^\s]+\.?\s*$', text):
            return True
        # Year with period: "2020."
        if re.search(r'\b(19|20)\d{2}[a-z]?\.\s*$', text):
            return True
        # Page count: "123 pp."
        if re.search(r'\d+\s*pp\.?\s*$', text):
            return True

        return False

    def _is_reference_fragment(self, text: str) -> bool:
        """
        Check if text is a reference continuation fragment rather than a new reference.

        A fragment lacks the author-year pattern that would indicate a new reference start.
        Returns True if text does NOT start with a surname pattern AND/OR does NOT
        contain a year in the first ~100 characters.

        Used by _merge_column_spanning_references() to identify the second half
        of a split reference at column/page boundaries.

        Args:
            text: The text to check

        Returns:
            True if text appears to be a continuation fragment (not a new reference)
        """
        text = text.strip()
        if not text:
            return True

        # Reference start pattern: capital letter surname followed by comma or space
        ref_start = re.match(
            r'^[A-Z\u00C0-\u024F][a-zA-Z\u00C0-\u024F\-\'\u2019]+,?\s',
            text
        )

        # Year in the first ~100 characters — but ignore years inside DOIs/URLs
        # (e.g., "https://doi.org/10.1016/j.isprsjprs.2009.09.002" contains "2009"
        # but that's not a publication year)
        early_text = text[:100]
        # Strip out DOI/URL substrings before checking for years
        cleaned_for_year = re.sub(r'https?://\S+', '', early_text)
        cleaned_for_year = re.sub(r'doi[:\s]\S+', '', cleaned_for_year, flags=re.IGNORECASE)
        has_early_year = bool(re.search(r'\b(19\d{2}|20\d{2})[a-z]?\b', cleaned_for_year))

        # It's a new reference only if it has BOTH a surname start AND an early year
        return not (ref_start and has_early_year)

    def _merge_column_spanning_references(
        self,
        enriched_elements: Dict[str, List],
        pdf_path: str
    ) -> Dict[str, List]:
        """
        Merge regular Reference elements at column/page boundaries.

        Handles the case where Document AI classifies both halves of a split
        reference as regular 'References' rather than 'Reference_Partial'.
        _merge_partial_references() only handles Reference_Partial elements,
        and _combine_adjacent_references() only handles intra-column fragments.
        This method fills the gap for cross-column/cross-page splits.

        Args:
            enriched_elements: Dictionary of enriched elements
            pdf_path: Path to PDF file (for page dimensions)

        Returns:
            Updated enriched_elements with merged references
        """
        references = enriched_elements.get('References', [])
        if len(references) < 2:
            return enriched_elements

        with PyMuPDFExtractor(pdf_path) as extractor:
            # Get page dimensions for column classification
            page_dims = {}
            for ref in references:
                if ref.page not in page_dims:
                    page_dims[ref.page] = extractor.get_page_dimensions(ref.page)

            def get_column(elem):
                page_width, _ = page_dims.get(elem.page, (595, 842))
                x_center = (elem.bbox[0] + elem.bbox[2]) / 2
                return 0 if x_center < page_width / 2 else 1

            def reading_order_key(elem):
                return (elem.page, get_column(elem), elem.bbox[1])

            sorted_refs = sorted(references, key=reading_order_key)

            # Walk consecutive pairs and merge at column/page boundaries
            merged = []
            i = 0
            merge_count = 0

            while i < len(sorted_refs):
                current = sorted_refs[i]

                if i + 1 < len(sorted_refs):
                    next_ref = sorted_refs[i + 1]
                    cur_col = get_column(current)
                    next_col = get_column(next_ref)

                    # Boundary = different page or different column
                    is_boundary = (current.page != next_ref.page) or (cur_col != next_col)

                    if is_boundary:
                        cur_incomplete = not self._is_reference_complete(current.text)
                        next_is_fragment = self._is_reference_fragment(next_ref.text)

                        if cur_incomplete and next_is_fragment:
                            # Merge: concatenate texts
                            merged_text = current.text.rstrip() + ' ' + next_ref.text.lstrip()

                            # Create new enriched element using current's bbox
                            page_width, page_height = page_dims.get(current.page, (595, 842))
                            norm_bbox = (
                                current.bbox[0] / page_width,
                                current.bbox[1] / page_height,
                                current.bbox[2] / page_width,
                                current.bbox[3] / page_height
                            )
                            new_elem = extractor.enrich_element(
                                'References', merged_text, norm_bbox, current.page
                            )
                            merged.append(new_elem)
                            merge_count += 1
                            print(f"  [Column Merge] Merged boundary refs: "
                                  f"'{current.text[:40]}...' + '{next_ref.text[:40]}...'")
                            i += 2  # Skip both elements
                            continue

                merged.append(current)
                i += 1

            if merge_count > 0:
                enriched_elements['References'] = merged
                print(f"  [Column Merge] Total merges: {merge_count}")

        return enriched_elements

    def _detect_references_in_gaps(
        self,
        enriched_elements: Dict[str, List],
        pdf_path: str
    ) -> Dict[str, List]:
        """
        Scan gaps between consecutive Reference elements for missed references.

        After all merging steps, sort References by reading order and check for
        vertical gaps between consecutive pairs within the same column. If a gap
        exceeds MIN_GAP_HEIGHT, scan the region with PyMuPDF. If the extracted
        text looks like a reference (starts with author pattern, contains year),
        add it as a new Reference element.

        Args:
            enriched_elements: Dictionary of enriched elements
            pdf_path: Path to PDF file

        Returns:
            Updated enriched_elements with any newly discovered references
        """
        references = enriched_elements.get('References', [])
        if len(references) < 2:
            return enriched_elements

        # Reference start pattern for validation
        ref_start_pattern = re.compile(
            r'^[A-Z\u00C0-\u024F][a-zA-Z\u00C0-\u024F\-\'\u2019]+,?\s',
            re.UNICODE
        )

        with PyMuPDFExtractor(pdf_path) as extractor:
            # Get page dimensions
            page_dims = {}
            for ref in references:
                if ref.page not in page_dims:
                    page_dims[ref.page] = extractor.get_page_dimensions(ref.page)

            # Layout constants (matching _find_partial_partner_fallback)
            left_margin = 57   # 20mm in points
            col_gap_margin = 3

            # Build blocking element bboxes for overlap checking
            blocking_bboxes = []
            for elem_type in self.BLOCKING_ELEMENT_TYPES:
                for elem in enriched_elements.get(elem_type, []):
                    if hasattr(elem, 'bbox') and hasattr(elem, 'page'):
                        blocking_bboxes.append({
                            'page': elem.page,
                            'bbox': elem.bbox
                        })

            def get_column(elem):
                page_width, _ = page_dims.get(elem.page, (595, 842))
                x_center = (elem.bbox[0] + elem.bbox[2]) / 2
                return 0 if x_center < page_width / 2 else 1

            def reading_order_key(elem):
                return (elem.page, get_column(elem), elem.bbox[1])

            sorted_refs = sorted(references, key=reading_order_key)

            new_references = []

            for i in range(len(sorted_refs) - 1):
                ref_a = sorted_refs[i]
                ref_b = sorted_refs[i + 1]

                # Only check gaps within the same page and column
                col_a = get_column(ref_a)
                col_b = get_column(ref_b)
                if ref_a.page != ref_b.page or col_a != col_b:
                    continue

                # Calculate gap height
                gap_y0 = ref_a.bbox[3]  # Bottom of ref A
                gap_y1 = ref_b.bbox[1]  # Top of ref B
                gap_height = gap_y1 - gap_y0

                if gap_height < self.MIN_GAP_HEIGHT:
                    continue

                # Define the gap scan region using column boundaries
                page_width, page_height = page_dims.get(ref_a.page, (595, 842))
                mid_x = page_width / 2

                if col_a == 0:  # Left column
                    gap_x0 = left_margin
                    gap_x1 = mid_x - col_gap_margin
                else:  # Right column
                    gap_x0 = mid_x + col_gap_margin
                    gap_x1 = page_width - left_margin

                gap_bbox = (gap_x0, gap_y0, gap_x1, gap_y1)

                # Check if the gap region overlaps with a blocking element
                overlaps_blocker = False
                for blocker in blocking_bboxes:
                    if blocker['page'] != ref_a.page:
                        continue
                    bx0, by0, bx1, by1 = blocker['bbox']
                    if by0 < gap_y1 and by1 > gap_y0 and bx0 < gap_x1 and bx1 > gap_x0:
                        overlaps_blocker = True
                        break

                if overlaps_blocker:
                    continue

                # Scan the gap region with PyMuPDF
                spans = extractor.extract_text_from_bbox(ref_a.page, gap_bbox)
                if not spans:
                    continue

                gap_text = ' '.join(s.text for s in spans).strip()

                # Validate: must be non-trivial and look like a reference
                if len(gap_text) < 20:
                    continue
                if not ref_start_pattern.match(gap_text):
                    continue
                if not re.search(r'\b(19\d{2}|20\d{2})\b', gap_text):
                    continue

                # Create a new Reference element
                norm_bbox = (
                    gap_x0 / page_width,
                    gap_y0 / page_height,
                    gap_x1 / page_width,
                    gap_y1 / page_height
                )
                new_elem = extractor.enrich_element(
                    'References', gap_text, norm_bbox, ref_a.page
                )
                new_references.append(new_elem)
                print(f"  [Gap Detect] Found missed reference: '{gap_text[:60]}...'")

            if new_references:
                enriched_elements['References'].extend(new_references)
                print(f"  [Gap Detect] Total missed references found: {len(new_references)}")

        return enriched_elements

    def _serialize_citation_results(self, citation_results: Dict) -> Dict:
        """
        Convert citation results to JSON-serializable format.

        Args:
            citation_results: Citation validation results with dataclass objects

        Returns:
            Dictionary with serialized results
        """
        serialized = {}

        # Serialize parsed references
        if 'references_parsed' in citation_results:
            serialized['references_parsed'] = [
                {
                    'text': ref.original_text,
                    'primary_surname': ref.primary_surname,
                    'all_authors': ref.all_authors,
                    'year': ref.year,
                    'year_suffix': ref.year_suffix,
                    'title': ref.title,
                    'additional_info': ref.additional_info,
                    'is_valid_format': ref.is_valid_format,
                    'format_issues': ref.format_issues
                }
                for ref in citation_results['references_parsed']
            ]

        # Serialize parsed citations
        if 'citations_parsed' in citation_results:
            serialized['citations_parsed'] = [
                {
                    'text': cit.text,
                    'primary_surname': cit.primary_surname,
                    'year': cit.year,
                    'year_suffix': cit.year_suffix,
                    'page': cit.page,
                    'bbox': cit.bbox,
                    'citation_type': cit.citation_type
                }
                for cit in citation_results['citations_parsed']
            ]

        # Serialize citation matches
        if 'citation_matches' in citation_results:
            serialized['citation_matches'] = [
                {
                    'citation_text': match.citation.text,
                    'matched': match.matched,
                    'reason': match.reason,
                    'confidence': match.confidence,
                    'reference_text': match.reference.original_text if match.reference else None
                }
                for match in citation_results['citation_matches']
            ]

        # Copy simple lists (already serializable)
        for key in ['orphan_citations', 'uncited_references', 'invalid_references', 'reference_format_issues']:
            if key in citation_results:
                serialized[key] = citation_results[key]

        return serialized


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze PDF documents for compliance using hybrid AI approach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py paper.pdf
  python main.py paper.docx --output analysis.json
  python main.py paper.tex --credentials /path/to/credentials.json
  python main.py paper.pdf --anon
  python main.py paper.pdf --anon --output results.json
  python main.py paper.pdf --report
  python main.py paper.pdf --anon --report
        """
    )

    parser.add_argument(
        "document",
        help="Path to document (PDF, DOCX, or TEX)"
    )

    parser.add_argument(
        "--output", "-o",
        help="Path for JSON output file (default: <document>_analysis.json)"
    )

    parser.add_argument(
        "--credentials", "-c",
        help="Path to Google Cloud credentials JSON file"
    )

    parser.add_argument(
        "--anon",
        action="store_true",
        help="Check for anonymization (detects named entities in Authors/Affiliations)"
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate annotated PDF report with validation results and summary"
    )

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.document).exists():
        print(f"Error: File not found: {args.document}", file=sys.stderr)
        sys.exit(1)

    try:
        # Run analysis
        analyzer = PDFComplianceAnalyzer(
            credentials_path=args.credentials,
            check_anonymization=args.anon,
            generate_report=args.report
        )
        analyzer.analyze(
            document_path=args.document,
            output_path=args.output
        )

        sys.exit(0)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
