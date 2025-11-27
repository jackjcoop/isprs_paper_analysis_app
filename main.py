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

    # Compiled regex patterns for heading classification
    SUB_SUB_HEADING_PATTERN = re.compile(r'^\d+\.\d+\.\d+\.?\s+')
    SUB_HEADING_PATTERN = re.compile(r'^\d+\.\d+\.?\s+(?!\d)')
    HEADING_PATTERN = re.compile(r'^\d+\.\s+(?!\d)')
    HEADING_NUM = re.compile(r'^(\d+)\.\s+')
    SUB_HEADING_NUM = re.compile(r'^(\d+)\.(\d+)\.?\s+')

    # Header section types that should not be checked for two-column layout
    # and should not have Main_Text duplicates
    HEADER_SECTION_TYPES = {'Title', 'Authors', 'Affiliations', 'Keywords', 'Abstract'}

    def __init__(
        self,
        credentials_path: str = None,
        check_anonymization: bool = False,
        generate_report: bool = False
    ):
        """
        Initialize analyzer.

        Args:
            credentials_path: Path to Google Cloud credentials
            check_anonymization: Whether to check for anonymization
            generate_report: Whether to generate annotated PDF report
        """
        self.credentials_path = credentials_path
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
                credentials_path=self.credentials_path
            )
            doc_ai_document = doc_ai_client.process_document(pdf_path)
            ai_elements = doc_ai_client.extract_elements_by_type(doc_ai_document)
            all_blocks = doc_ai_client.get_all_blocks(doc_ai_document)

        print(f"  Found {len(all_blocks)} text blocks from Document AI")

        # Step 3: Enrich with PyMuPDF font information
        with ProgressIndicator("Extracting font and style information (PyMuPDF)"):
            enriched_elements = self._enrich_with_pymupdf(
                ai_elements,
                all_blocks,
                pdf_path
            )

        # Step 3.1: Filter duplicate Main_Text elements that overlap header sections
        with ProgressIndicator("Filtering duplicate Main_Text in header regions"):
            enriched_elements = self._filter_main_text_duplicates(enriched_elements)

        # Step 3.5: Reclassify headings based on numbering
        with ProgressIndicator("Reclassifying headings based on numbering"):
            enriched_elements = self._reclassify_headings(enriched_elements)

        # Step 3.6: Detect missed headings in gaps between Main_Text
        with ProgressIndicator("Detecting missed headings in gaps"):
            enriched_elements = self._detect_headings_in_gaps(enriched_elements, pdf_path)

        # Step 3.7: Detect missed references in gaps
        with ProgressIndicator("Detecting missed references in gaps"):
            enriched_elements = self._detect_references_in_gaps(enriched_elements, pdf_path)

        # Step 3.8: Merge column-spanning references
        with ProgressIndicator("Merging column-spanning references"):
            enriched_elements = self._merge_column_spanning_references(enriched_elements, pdf_path)

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
        all_blocks: List[ExtractedElement],
        pdf_path: str
    ) -> Dict[str, List[EnrichedElement]]:
        """
        Enrich Document AI elements with PyMuPDF font information.

        Args:
            ai_elements: Elements from Document AI
            all_blocks: All text blocks from Document AI
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

            # Also enrich generic blocks if no specific elements found
            for element_type in ai_elements.keys():
                if not enriched[element_type] and all_blocks:
                    # Use first few blocks as fallback
                    for block in all_blocks[:5]:
                        enriched_element = extractor.enrich_element(
                            element_type=element_type,
                            text=block.text,
                            bbox=(
                                block.bbox.x0,
                                block.bbox.y0,
                                block.bbox.x1,
                                block.bbox.y1
                            ),
                            page=block.bbox.page
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

        Args:
            enriched_elements: Enriched document elements
            pdf_path: Path to PDF file

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

        with PyMuPDFExtractor(pdf_path) as extractor:
            # Check for Keywords label (to the left of keywords box)
            keywords_elements = enriched_elements.get("Keywords", [])
            if keywords_elements:
                # Store element ref for potential highlighting (use first element initially)
                first_element = keywords_elements[0]
                labels_verified["Keywords_element_ref"] = (
                    first_element.page,
                    first_element.bbox
                )

                for element in keywords_elements:
                    found, actual_text, is_correct = extractor.verify_label(
                        label_text="Keywords",
                        bbox=(
                            element.bbox[0] / extractor.get_page_dimensions(element.page)[0],
                            element.bbox[1] / extractor.get_page_dimensions(element.page)[1],
                            element.bbox[2] / extractor.get_page_dimensions(element.page)[0],
                            element.bbox[3] / extractor.get_page_dimensions(element.page)[1]
                        ),
                        page=element.page,
                        direction="left"
                    )
                    if found:
                        labels_verified["Keywords"] = True
                        if not is_correct:
                            labels_verified["Keywords_format_warning"] = (
                                f"Found '{actual_text}' - should be 'Keywords'"
                            )
                            # Update element_ref to match the element where format issue was found
                            labels_verified["Keywords_element_ref"] = (
                                element.page,
                                element.bbox
                            )
                        break

            # Check for Abstract label (above abstract section)
            abstract_elements = enriched_elements.get("Abstract", [])
            if abstract_elements:
                # Store element ref for potential highlighting (use first element initially)
                first_element = abstract_elements[0]
                labels_verified["Abstract_element_ref"] = (
                    first_element.page,
                    first_element.bbox
                )

                for element in abstract_elements:
                    found, actual_text, is_correct = extractor.verify_label(
                        label_text="Abstract",
                        bbox=(
                            element.bbox[0] / extractor.get_page_dimensions(element.page)[0],
                            element.bbox[1] / extractor.get_page_dimensions(element.page)[1],
                            element.bbox[2] / extractor.get_page_dimensions(element.page)[0],
                            element.bbox[3] / extractor.get_page_dimensions(element.page)[1]
                        ),
                        page=element.page,
                        direction="above"
                    )
                    if found:
                        labels_verified["Abstract"] = True
                        if not is_correct:
                            labels_verified["Abstract_format_warning"] = (
                                f"Found '{actual_text}' - should be 'Abstract'"
                            )
                            # Update element_ref to match the element where format issue was found
                            labels_verified["Abstract_element_ref"] = (
                                element.page,
                                element.bbox
                            )
                        break

        return labels_verified

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
                print(f"  [Heading Reclassify] Dropped non-heading: '{text[:60]}...'")
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

    def _extract_heading_numbers(self, enriched_elements: Dict[str, List]) -> Dict[str, List]:
        """
        Extract heading numbers from Document AI results.

        Returns:
            Dict with 'headings' and 'sub_headings' lists of numbers found
            e.g., {'headings': [1, 2, 4], 'sub_headings': [(2,1), (2,2), (4,1)]}
        """
        heading_nums = []
        sub_heading_nums = []

        for elem in enriched_elements.get('Headings', []):
            match = self.HEADING_NUM.match(elem.text.strip())
            if match:
                heading_nums.append(int(match.group(1)))

        for elem in enriched_elements.get('Sub_Headings', []):
            match = self.SUB_HEADING_NUM.match(elem.text.strip())
            if match:
                sub_heading_nums.append((int(match.group(1)), int(match.group(2))))

        return {
            'headings': sorted(set(heading_nums)),
            'sub_headings': sorted(set(sub_heading_nums))
        }

    def _find_missing_heading_numbers(self, found_nums: Dict[str, List]) -> Dict[str, List]:
        """
        Find missing numbers in heading sequences + next expected (end-of-series).

        For headings [1, 2, 4]: missing = [3, 5]  (3 is gap, 5 is next expected)
        For sub_headings under 2 [(2,1), (2,3)]: missing = [(2,2), (2,4)]
        """
        missing = {'headings': [], 'sub_headings': []}

        # Check main headings
        headings = found_nums['headings']
        if headings:
            # Find gaps in sequence
            for i in range(headings[0], headings[-1]):
                if i not in headings:
                    missing['headings'].append(i)
            # Add N+1 for end-of-series
            missing['headings'].append(headings[-1] + 1)

        # Check sub-headings grouped by parent
        sub_headings = found_nums['sub_headings']
        if sub_headings:
            # Group by parent heading number
            by_parent = {}
            for parent, child in sub_headings:
                by_parent.setdefault(parent, []).append(child)

            for parent, children in by_parent.items():
                children = sorted(set(children))
                # Find gaps
                for i in range(children[0], children[-1]):
                    if i not in children:
                        missing['sub_headings'].append((parent, i))
                # Add N+1 for end-of-series
                missing['sub_headings'].append((parent, children[-1] + 1))

        return missing

    def _get_column(self, bbox: tuple, page_width: float) -> int:
        """
        Determine which column an element is in based on its x-position.

        Args:
            bbox: Bounding box tuple (x0, y0, x1, y1) in absolute coordinates
            page_width: Width of the page in pixels

        Returns:
            0 for left column, 1 for right column
        """
        center_x = (bbox[0] + bbox[2]) / 2
        midpoint = page_width / 2
        return 0 if center_x < midpoint else 1

    def _build_exclusion_zones(
        self,
        enriched_elements: Dict[str, List],
        page_dims: Dict[int, tuple],
        include_header_sections: bool = True
    ) -> List[Dict]:
        """
        Build exclusion zones for figure/table/equation regions and optionally header sections.

        Text within these zones should be skipped during gap detection to prevent
        capturing figure axis labels, equation text, or header section content.

        Args:
            enriched_elements: Dictionary of enriched elements
            page_dims: Dictionary mapping page number to (width, height) tuples
            include_header_sections: Whether to include Title/Authors/Affiliations/Keywords/Abstract

        Returns:
            List of exclusion zone dictionaries with keys: page, x0, y0, x1, y1
        """
        exclusion_zones = []

        # Left margin for equation expansion
        left_margin = 57  # 20mm in points

        # Add figure/table/equation exclusion zones
        for elem_type in self.BLOCKING_ELEMENT_TYPES:
            for elem in enriched_elements.get(elem_type, []):
                if not hasattr(elem, 'bbox') or not hasattr(elem, 'page'):
                    continue

                page = elem.page
                bbox = elem.bbox

                if elem_type == 'Equation_Number':
                    # Expand equation number bbox to the left to cover the equation itself
                    # Use column-aware expansion: left column from left_margin, right column from midpoint
                    page_width, _ = page_dims.get(page, (595, 842))
                    midpoint = page_width / 2
                    x_center = (bbox[0] + bbox[2]) / 2
                    if x_center < midpoint:
                        # Left column - expand from left margin
                        column_start = left_margin
                    else:
                        # Right column - expand from midpoint (+ gap/2 for column spacing)
                        column_start = midpoint + 3  # 6pt column gap / 2
                    expanded_bbox = (column_start, bbox[1], bbox[2], bbox[3])
                    exclusion_zones.append({
                        'page': page,
                        'x0': expanded_bbox[0],
                        'y0': expanded_bbox[1],
                        'x1': expanded_bbox[2],
                        'y1': expanded_bbox[3],
                        'type': elem_type
                    })
                elif elem_type in ('Figure_Title', 'Table_Title'):
                    # For titles, expand upward to cover the figure/table above
                    # Use a minimum fixed expansion (150pt) to capture figure labels like a), b), c)
                    # Also check nearest Main_Text to avoid extending too far
                    page_width, page_height = page_dims.get(page, (595, 842))

                    # Minimum expansion above title (150pt covers most figure labels)
                    min_expansion = 150
                    min_y = max(0, bbox[1] - min_expansion)

                    # Find nearest Main_Text element above - don't expand past it
                    nearest_above_y = 0  # Default to top of page
                    for mt in enriched_elements.get('Main_Text', []):
                        if hasattr(mt, 'page') and hasattr(mt, 'bbox'):
                            if mt.page == page and mt.bbox[3] < bbox[1]:  # mt bottom < title top
                                nearest_above_y = max(nearest_above_y, mt.bbox[3])

                    # Use the lower of: nearest Main_Text bottom OR minimum expansion limit
                    # This ensures we cover figure content but don't extend into Main_Text
                    expanded_y = max(nearest_above_y, min_y)

                    # Also expand horizontally to full column width to catch all labels
                    column_center = (bbox[0] + bbox[2]) / 2
                    column_width = (page_width - 2 * left_margin - 17) / 2  # Two columns with gap
                    if column_center < page_width / 2:  # Left column
                        col_x0 = left_margin
                        col_x1 = left_margin + column_width
                    else:  # Right column
                        col_x0 = page_width - left_margin - column_width
                        col_x1 = page_width - left_margin

                    expanded_bbox = (col_x0, expanded_y, col_x1, bbox[3])
                    exclusion_zones.append({
                        'page': page,
                        'x0': expanded_bbox[0],
                        'y0': expanded_bbox[1],
                        'x1': expanded_bbox[2],
                        'y1': expanded_bbox[3],
                        'type': elem_type
                    })
                else:
                    # Use bbox as-is for other blocking elements
                    exclusion_zones.append({
                        'page': page,
                        'x0': bbox[0],
                        'y0': bbox[1],
                        'x1': bbox[2],
                        'y1': bbox[3],
                        'type': elem_type
                    })

        # Add header section exclusion zones if requested
        if include_header_sections:
            for elem_type in self.HEADER_SECTION_TYPES:
                for elem in enriched_elements.get(elem_type, []):
                    if not hasattr(elem, 'bbox') or not hasattr(elem, 'page'):
                        continue
                    exclusion_zones.append({
                        'page': elem.page,
                        'x0': elem.bbox[0],
                        'y0': elem.bbox[1],
                        'x1': elem.bbox[2],
                        'y1': elem.bbox[3],
                        'type': elem_type
                    })

        return exclusion_zones

    def _is_in_exclusion_zone(
        self,
        bbox: tuple,
        page: int,
        exclusion_zones: List[Dict],
        overlap_threshold: float = 0.3
    ) -> bool:
        """
        Check if a bounding box overlaps significantly with any exclusion zone.

        Args:
            bbox: Bounding box tuple (x0, y0, x1, y1) in absolute coordinates
            page: Page number
            exclusion_zones: List of exclusion zone dictionaries
            overlap_threshold: Minimum overlap ratio to consider as "inside" (0.0-1.0)

        Returns:
            True if bbox overlaps significantly with any exclusion zone
        """
        x0, y0, x1, y1 = bbox
        bbox_area = (x1 - x0) * (y1 - y0)

        if bbox_area <= 0:
            return False

        for zone in exclusion_zones:
            if zone['page'] != page:
                continue

            # Calculate intersection
            int_x0 = max(x0, zone['x0'])
            int_y0 = max(y0, zone['y0'])
            int_x1 = min(x1, zone['x1'])
            int_y1 = min(y1, zone['y1'])

            if int_x0 >= int_x1 or int_y0 >= int_y1:
                continue  # No overlap

            int_area = (int_x1 - int_x0) * (int_y1 - int_y0)
            overlap_ratio = int_area / bbox_area

            if overlap_ratio >= overlap_threshold:
                return True

        return False

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

    def _detect_headings_in_gaps(
        self,
        enriched_elements: Dict[str, List],
        pdf_path: str
    ) -> Dict[str, List]:
        """
        Detect ONLY missing headings in gaps between Main_Text elements.
        Uses sequence-based detection to avoid false positives.

        Args:
            enriched_elements: Dictionary of enriched elements
            pdf_path: Path to PDF file

        Returns:
            Updated enriched_elements with detected headings added
        """
        # Extract what Document AI found and determine what's missing
        found_nums = self._extract_heading_numbers(enriched_elements)
        missing_nums = self._find_missing_heading_numbers(found_nums)

        print(f"  [Heading Gap Detection] Found heading numbers: {found_nums['headings']}")
        print(f"  [Heading Gap Detection] Found sub-heading numbers: {found_nums['sub_headings']}")
        print(f"  [Heading Gap Detection] Missing/expected headings: {missing_nums['headings']}")
        print(f"  [Heading Gap Detection] Missing/expected sub-headings: {missing_nums['sub_headings']}")

        # Build patterns for ONLY the missing/expected numbers
        missing_patterns = []
        for num in missing_nums['headings']:
            pattern = re.compile(rf'^{num}\.\s+(?!\d)')
            missing_patterns.append(('Headings', pattern, str(num)))

        for parent, child in missing_nums['sub_headings']:
            pattern = re.compile(rf'^{parent}\.{child}\.?\s+(?!\d)')
            missing_patterns.append(('Sub_Headings', pattern, f"{parent}.{child}"))

        # If no missing patterns and no special headings to search, skip
        if not missing_patterns:
            print("  [Heading Gap Detection] No missing heading numbers, skipping numbered heading scan")

        main_text = enriched_elements.get('Main_Text', [])
        if len(main_text) < 2:
            return enriched_elements

        # Collect blocking elements (figures, tables, equations) that should prevent gap scanning
        blocking_elements = []
        for elem_type in self.BLOCKING_ELEMENT_TYPES:
            for elem in enriched_elements.get(elem_type, []):
                blocking_elements.append(elem)

        # Open extractor early to get page dimensions for column detection
        with PyMuPDFExtractor(pdf_path) as extractor:
            # Get page dimensions for all pages with main text
            page_dims = {}
            for elem in main_text:
                if elem.page not in page_dims:
                    page_dims[elem.page] = extractor.get_page_dimensions(elem.page)

            # Build exclusion zones for figure/table/equation content
            # This prevents capturing text inside figures (axis labels, etc.)
            exclusion_zones = self._build_exclusion_zones(
                enriched_elements, page_dims, include_header_sections=True
            )

            # Sort by page, column, then y0 for column-aware gap detection
            def sort_key(e):
                page_width = page_dims.get(e.page, (1, 1))[0]
                # Convert normalized bbox to absolute for column detection
                abs_bbox = (
                    e.bbox[0] * page_width,
                    e.bbox[1],
                    e.bbox[2] * page_width,
                    e.bbox[3]
                )
                return (e.page, self._get_column(abs_bbox, page_width), e.bbox[1])

            sorted_main = sorted(main_text, key=sort_key)

            # Calculate gaps between consecutive elements in same page AND column
            gaps = []
            for i in range(len(sorted_main) - 1):
                curr = sorted_main[i]
                next_elem = sorted_main[i + 1]

                if curr.page != next_elem.page:
                    continue

                page_width, page_height = page_dims.get(curr.page, (1, 1))

                # Convert to absolute coords for column check
                curr_abs_bbox = (
                    curr.bbox[0] * page_width,
                    curr.bbox[1] * page_height,
                    curr.bbox[2] * page_width,
                    curr.bbox[3] * page_height
                )
                next_abs_bbox = (
                    next_elem.bbox[0] * page_width,
                    next_elem.bbox[1] * page_height,
                    next_elem.bbox[2] * page_width,
                    next_elem.bbox[3] * page_height
                )

                # Must be in same column
                if self._get_column(curr_abs_bbox, page_width) != self._get_column(next_abs_bbox, page_width):
                    continue

                gap_height = next_abs_bbox[1] - curr_abs_bbox[3]  # y0_next - y1_curr (absolute)
                if gap_height <= 0:
                    continue

                # Define gap region for blocking check
                gap_x0 = min(curr_abs_bbox[0], next_abs_bbox[0])
                gap_x1 = max(curr_abs_bbox[2], next_abs_bbox[2])
                gap_y_start = curr_abs_bbox[3]
                gap_y_end = next_abs_bbox[1]

                # Check if any blocking element falls within this gap
                gap_blocked = False
                for blocker in blocking_elements:
                    if blocker.page != curr.page:
                        continue

                    # Convert blocker bbox to absolute coords
                    blocker_abs_bbox = (
                        blocker.bbox[0] * page_width,
                        blocker.bbox[1] * page_height,
                        blocker.bbox[2] * page_width,
                        blocker.bbox[3] * page_height
                    )

                    # Check if blocker overlaps with the gap vertically
                    if blocker_abs_bbox[1] < gap_y_end and blocker_abs_bbox[3] > gap_y_start:
                        # Check horizontal overlap (same column region)
                        if blocker_abs_bbox[2] > gap_x0 and blocker_abs_bbox[0] < gap_x1:
                            gap_blocked = True
                            break

                if not gap_blocked:
                    gaps.append({
                        'height': gap_height,
                        'page': curr.page,
                        'y_start': gap_y_start,
                        'y_end': gap_y_end,
                        'x0': gap_x0,
                        'x1': gap_x1
                    })

            if not gaps:
                return enriched_elements

            # Calculate median gap height
            gap_heights = [g['height'] for g in gaps]
            median_gap = sorted(gap_heights)[len(gap_heights) // 2]

            # Threshold: gaps > 2x median are suspicious
            threshold = median_gap * 2
            large_gaps = [g for g in gaps if g['height'] > threshold]

            print(f"  [Heading Gap Detection] Median gap: {median_gap:.1f}px, threshold: {threshold:.1f}px")
            print(f"  [Heading Gap Detection] Found {len(large_gaps)} large gaps to scan (blocked gaps excluded)")

            # Scan large gaps for headings
            for gap in large_gaps:
                page_width, page_height = page_dims.get(gap['page'], (1, 1))
                if page_width == 0 or page_height == 0:
                    continue

                # Gap bbox is already in absolute coords
                abs_bbox = (gap['x0'], gap['y_start'], gap['x1'], gap['y_end'])
                spans = extractor.extract_text_from_bbox(gap['page'], abs_bbox)

                if not spans:
                    continue

                # Filter out spans that fall within exclusion zones (figure/equation content)
                filtered_spans = []
                for span in spans:
                    span_bbox = span.bbox  # Already in absolute coords
                    if not self._is_in_exclusion_zone(span_bbox, gap['page'], exclusion_zones):
                        filtered_spans.append(span)

                if not filtered_spans:
                    continue

                # Combine span text
                gap_text = ' '.join(s.text for s in filtered_spans).strip()
                if not gap_text:
                    continue

                # Convert absolute coords to normalized for enrich_element()
                norm_bbox = (
                    gap['x0'] / page_width,
                    gap['y_start'] / page_height,
                    gap['x1'] / page_width,
                    gap['y_end'] / page_height
                )

                # Check for ONLY missing/expected heading patterns (with length filter)
                matched = False
                if len(gap_text) <= self.MAX_HEADING_LENGTH:
                    for elem_type, pattern, num_str in missing_patterns:
                        if pattern.match(gap_text):
                            print(f"  [Heading Gap Detection] Found missing {elem_type} {num_str}: '{gap_text[:60]}'")
                            new_elem = extractor.enrich_element(
                                elem_type, gap_text, norm_bbox, gap['page']
                            )
                            enriched_elements[elem_type].append(new_elem)
                            matched = True
                            break

                # Check for special headings (stricter: short text that IS or starts with the word)
                if not matched and len(gap_text) <= 50:
                    gap_text_lower = gap_text.lower().strip()
                    for special in self.SPECIAL_HEADINGS:
                        if gap_text_lower == special or gap_text_lower.startswith(special + ' '):
                            print(f"  [Heading Gap Detection] Found Special Heading: '{gap_text[:60]}'")
                            new_elem = extractor.enrich_element(
                                'Headings', gap_text, norm_bbox, gap['page']
                            )
                            enriched_elements['Headings'].append(new_elem)
                            break

        return enriched_elements

    def _detect_references_in_gaps(
        self,
        enriched_elements: Dict[str, List],
        pdf_path: str
    ) -> Dict[str, List]:
        """
        Detect missed references in gaps between Reference elements using PyMuPDF.

        Args:
            enriched_elements: Dictionary of enriched elements
            pdf_path: Path to PDF file

        Returns:
            Updated enriched_elements with detected references added
        """
        references = enriched_elements.get('References', [])
        if len(references) < 2:
            return enriched_elements

        # Sort by page, then y0
        sorted_refs = sorted(references, key=lambda e: (e.page, e.bbox[1]))

        # Calculate gaps between consecutive elements on same page
        gaps = []
        for i in range(len(sorted_refs) - 1):
            curr = sorted_refs[i]
            next_elem = sorted_refs[i + 1]

            if curr.page != next_elem.page:
                continue

            gap_height = next_elem.bbox[1] - curr.bbox[3]  # y0_next - y1_curr
            if gap_height > 0:
                gaps.append({
                    'height': gap_height,
                    'page': curr.page,
                    'y_start': curr.bbox[3],
                    'y_end': next_elem.bbox[1],
                    'x0': min(curr.bbox[0], next_elem.bbox[0]),
                    'x1': max(curr.bbox[2], next_elem.bbox[2])
                })

        if not gaps:
            return enriched_elements

        # Calculate median gap height
        gap_heights = [g['height'] for g in gaps]
        median_gap = sorted(gap_heights)[len(gap_heights) // 2]

        # Threshold: gaps > 2x median are suspicious
        threshold = median_gap * 2
        large_gaps = [g for g in gaps if g['height'] > threshold]

        print(f"  [Reference Gap Detection] Median gap: {median_gap:.1f}px, threshold: {threshold:.1f}px")
        print(f"  [Reference Gap Detection] Found {len(large_gaps)} large gaps to scan")

        # Scan large gaps for references
        with PyMuPDFExtractor(pdf_path) as extractor:
            # Get page dimensions for exclusion zone building
            page_dims = {}
            for ref in references:
                if ref.page not in page_dims:
                    page_dims[ref.page] = extractor.get_page_dimensions(ref.page)

            # Build exclusion zones for figure/table/equation content
            exclusion_zones = self._build_exclusion_zones(
                enriched_elements, page_dims, include_header_sections=False
            )

            for gap in large_gaps:
                page_width, page_height = extractor.get_page_dimensions(gap['page'])
                if page_width == 0 or page_height == 0:
                    continue

                abs_bbox = (gap['x0'], gap['y_start'], gap['x1'], gap['y_end'])
                spans = extractor.extract_text_from_bbox(gap['page'], abs_bbox)

                if not spans:
                    continue

                # Filter out spans that fall within exclusion zones (figure/equation content)
                filtered_spans = []
                for span in spans:
                    span_bbox = span.bbox  # Already in absolute coords
                    if not self._is_in_exclusion_zone(span_bbox, gap['page'], exclusion_zones):
                        filtered_spans.append(span)

                if not filtered_spans:
                    continue

                gap_text = ' '.join(s.text for s in filtered_spans).strip()
                if not gap_text or len(gap_text) < 10:  # Skip tiny fragments
                    continue

                print(f"  [Reference Gap Detection] Found reference: '{gap_text[:60]}...'")

                norm_bbox = (
                    gap['x0'] / page_width,
                    gap['y_start'] / page_height,
                    gap['x1'] / page_width,
                    gap['y_end'] / page_height
                )

                new_elem = extractor.enrich_element(
                    'References', gap_text, norm_bbox, gap['page']
                )
                enriched_elements['References'].append(new_elem)

        return enriched_elements

    def _is_reference_fragment(self, text: str) -> bool:
        """
        Determine if text is a fragment (continuation) rather than a new reference.
        A fragment lacks both: author surname pattern at start AND year in first 100 chars.

        Args:
            text: The reference text to check

        Returns:
            True if text appears to be a fragment, False if it looks like a new reference
        """
        text = text.strip()
        if not text:
            return True

        # Pattern: starts with capitalized surname followed by comma or space
        # Includes Latin-1 extended characters for European names (Pöntinen, Müller, etc.)
        starts_with_author = bool(re.match(r'^[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ\-\'\u2019]+,?\s', text))

        # Check for year within first 100 characters
        early_text = text[:100] if len(text) > 100 else text
        has_early_year = bool(re.search(r'\b(19\d{2}|20\d{2})[a-z]?\b', early_text))

        # It's a fragment if it doesn't have BOTH author pattern AND early year
        return not (starts_with_author and has_early_year)

    def _is_reference_complete(self, text: str) -> bool:
        """
        Check if a reference appears complete (not needing continuation).
        A complete reference typically ends with:
        - Page numbers (pp. 123, 45-67, or just 45-67)
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

    def _merge_column_spanning_references(
        self,
        enriched_elements: Dict[str, List],
        pdf_path: str
    ) -> Dict[str, List]:
        """
        Merge references that span columns/pages.
        Only merges at column/page boundaries where text looks like a fragment.

        Merge conditions:
        1. Position: Previous ref at bottom of column, current at top of next column
        2. Content: Current text lacks author+year pattern (is a fragment)

        Valid transitions:
        - Left column bottom → Right column top (same page)
        - Right column bottom → Left column top (next page)

        Args:
            enriched_elements: Dictionary of enriched elements
            pdf_path: Path to PDF file

        Returns:
            Updated enriched_elements with merged references
        """
        references = enriched_elements.get('References', [])
        if len(references) < 2:
            return enriched_elements

        with PyMuPDFExtractor(pdf_path) as extractor:
            # Get page dimensions for column detection
            page_dims = {}
            for ref in references:
                if ref.page not in page_dims:
                    page_dims[ref.page] = extractor.get_page_dimensions(ref.page)

            # Sort by reading order: (page, column, y_position)
            def reading_order_key(r):
                page = r.page
                page_width = page_dims.get(page, (595, 842))[0]
                mid_x = page_width / 2
                x_center = (r.bbox[0] + r.bbox[2]) / 2
                column = 0 if x_center < mid_x else 1
                y_pos = r.bbox[1]
                return (page, column, y_pos)

            sorted_refs = sorted(references, key=reading_order_key)

            # Helper to get column info for a reference
            def get_column_info(ref):
                page = ref.page
                page_width, page_height = page_dims.get(page, (595, 842))
                mid_x = page_width / 2
                x_center = (ref.bbox[0] + ref.bbox[2]) / 2
                column = 0 if x_center < mid_x else 1
                # Normalize y position (0=top, 1=bottom)
                y_normalized = ref.bbox[3] / page_height  # Use bottom of bbox
                return page, column, y_normalized, page_height

            # Merge fragments at column/page boundaries
            merged_refs = []
            current_ref = None
            current_page, current_col, current_y_norm = None, None, None

            for ref in sorted_refs:
                text = ref.text.strip()
                if not text:
                    continue

                ref_page, ref_col, ref_y_norm, page_height = get_column_info(ref)
                is_fragment = self._is_reference_fragment(text)

                # Determine if this is a valid merge position
                # Require BOTH: current text is a fragment AND previous reference is incomplete
                can_merge = False
                prev_is_complete = self._is_reference_complete(current_ref.text) if current_ref else True
                if current_ref is not None and is_fragment and not prev_is_complete:
                    # Check for column/page boundary transition
                    prev_at_bottom = current_y_norm > 0.70  # Previous ref at bottom 30%
                    curr_at_top = (ref.bbox[1] / page_height) < 0.30  # Current ref at top 30%

                    # Case 1: Same page, left column → right column
                    same_page_col_transition = (
                        ref_page == current_page and
                        current_col == 0 and ref_col == 1 and
                        prev_at_bottom and curr_at_top
                    )

                    # Case 2: Cross-page, right column → left column of next page
                    cross_page_transition = (
                        ref_page == current_page + 1 and
                        current_col == 1 and ref_col == 0 and
                        prev_at_bottom and curr_at_top
                    )

                    can_merge = same_page_col_transition or cross_page_transition

                if can_merge:
                    # Merge with current reference
                    merged_text = current_ref.text.rstrip() + ' ' + text.lstrip()
                    print(f"  [Reference Merge] Merging fragment at col/page boundary: '{text[:40]}...'")

                    # Always use first reference's bbox
                    merged_bbox = current_ref.bbox
                    page_width, page_height = page_dims.get(current_ref.page, (595, 842))
                    norm_bbox = (
                        merged_bbox[0] / page_width,
                        merged_bbox[1] / page_height,
                        merged_bbox[2] / page_width,
                        merged_bbox[3] / page_height
                    )

                    current_ref = extractor.enrich_element(
                        'References', merged_text, norm_bbox, current_ref.page
                    )
                    # Update position tracking after merge
                    current_page, current_col, current_y_norm, _ = get_column_info(current_ref)
                else:
                    # Save previous reference and start new one
                    if current_ref is not None:
                        merged_refs.append(current_ref)
                    current_ref = ref
                    current_page, current_col, current_y_norm = ref_page, ref_col, ref_y_norm

            # Don't forget the last reference
            if current_ref is not None:
                merged_refs.append(current_ref)

            enriched_elements['References'] = merged_refs

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
