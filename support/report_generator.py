"""
PDF Report Generator Module
Generates annotated PDF reports with validation results and summary pages.
"""

import fitz  # PyMuPDF
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from .validator import ValidationResult, Severity


# Layout constants
PAGE_WIDTH = 595   # A4 width in points
PAGE_HEIGHT = 842  # A4 height in points
MARGIN = 40
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN  # 515pt

# Table settings
TABLE_ROW_HEIGHT = 18
TABLE_HEADER_HEIGHT = 22
TABLE_BORDER_COLOR = (0.7, 0.7, 0.7)
TABLE_HEADER_BG = (0.92, 0.92, 0.92)
TABLE_ALT_ROW_BG = (0.97, 0.97, 0.97)

# Colors
COLOR_PASS = (0, 0.6, 0)       # Green
COLOR_WARN = (1, 0.5, 0)       # Orange
COLOR_INFO = (0, 0.4, 0.8)     # Blue
COLOR_GRAY = (0.5, 0.5, 0.5)
COLOR_DARK = (0.2, 0.2, 0.2)

# Annotation color - all warnings use orange
WARNING_COLOR = COLOR_WARN

# Subjective reminder items from ISPRS checklist (cannot be auto-checked)
SUBJECTIVE_REMINDERS = [
    {
        'category': 'Content Quality',
        'items': [
            'Abstract is understandable to non-specialists',
            'Clearly states the scientific contribution and results',
            'Logical organization of content throughout',
        ]
    },
    {
        'category': 'Units & Formatting',
        'items': [
            'SI Units (Système International) used throughout',
            'Large figures/tables rotated 90° with top on left side (if needed)',
            'Copyright statements included for third-party imagery',
        ]
    },
    {
        'category': 'Final Review',
        'items': [
            'Spell check completed',
            'All figures and tables render correctly',
            'Document opens correctly in Word/PDF',
        ]
    },
    {
        'category': 'Review vs Camera-Ready',
        'items': [
            'No self-identifying citations (review version)',
            'Full author information included (camera-ready version)',
        ]
    },
]


class ReportGenerator:
    """Generates annotated PDF reports with validation results."""

    def __init__(
        self,
        pdf_path: str,
        validation_results: List[ValidationResult],
        enriched_elements: Dict[str, List],
        overall_pass: bool
    ):
        """
        Initialize report generator.

        Args:
            pdf_path: Path to original PDF document
            validation_results: List of validation results
            enriched_elements: Dictionary of enriched document elements
            overall_pass: Overall validation pass/fail status
        """
        self.pdf_path = pdf_path
        self.validation_results = validation_results
        self.enriched_elements = enriched_elements
        self.overall_pass = overall_pass

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate complete PDF report with summary pages followed by annotated PDF.

        Args:
            output_path: Optional output path (default: <original>_report.pdf)

        Returns:
            Path to generated report
        """
        if output_path is None:
            base_path = Path(self.pdf_path)
            output_path = str(base_path.parent / f"{base_path.stem}_report.pdf")

        # Generate summary pages FIRST
        summary_doc = self._generate_summary_pages()

        # Open the original PDF and add annotations
        original_doc = fitz.open(self.pdf_path)
        self._add_annotations(original_doc)

        # Append annotated original to summary (summary comes first)
        summary_doc.insert_pdf(original_doc)
        original_doc.close()

        # Save the final report
        summary_doc.save(output_path)
        summary_doc.close()

        return output_path

    def _add_annotations(self, doc: fitz.Document):
        """Add bounding box annotations for failed validation checks."""
        # Collect all annotations by location (page, bbox) to combine duplicates
        annotations_by_location = {}

        for result in self.validation_results:
            if result.passed:
                continue

            if not result.element_refs:
                continue

            for ref in result.element_refs:
                # Handle new format (page, bbox, instance_msg) with per-instance messages
                if len(ref) == 3:
                    page_num, bbox, instance_msg = ref
                else:
                    # Fallback for old format (page, bbox) - use summary message
                    page_num, bbox = ref
                    instance_msg = result.message

                if page_num >= len(doc):
                    continue

                # Create a hashable key for the location (round bbox coords for matching)
                bbox_key = (page_num, round(bbox[0], 1), round(bbox[1], 1), round(bbox[2], 1), round(bbox[3], 1))

                if bbox_key not in annotations_by_location:
                    annotations_by_location[bbox_key] = {
                        'page_num': page_num,
                        'bbox': bbox,
                        'check_names': [],
                        'messages': [],
                        'details': []
                    }

                # Add this issue to the annotation
                if result.check_name not in annotations_by_location[bbox_key]['check_names']:
                    annotations_by_location[bbox_key]['check_names'].append(result.check_name)
                annotations_by_location[bbox_key]['messages'].append(instance_msg)
                if result.details and result.details not in annotations_by_location[bbox_key]['details']:
                    annotations_by_location[bbox_key]['details'].append(result.details)

        # Now add combined annotations
        color = WARNING_COLOR
        for bbox_key, annot_data in annotations_by_location.items():
            page = doc[annot_data['page_num']]

            # Combine check names
            combined_check_name = " + ".join(annot_data['check_names'])

            # Combine messages (remove duplicates while preserving order)
            seen_messages = set()
            unique_messages = []
            for msg in annot_data['messages']:
                if msg not in seen_messages:
                    seen_messages.add(msg)
                    unique_messages.append(msg)
            combined_message = "\n".join(unique_messages)

            # Combine details
            combined_details = "\n".join(annot_data['details']) if annot_data['details'] else ""

            self._add_annotation(
                page=page,
                bbox=annot_data['bbox'],
                color=color,
                check_name=combined_check_name,
                message=combined_message,
                details=combined_details
            )

    def _add_annotation(
        self,
        page: fitz.Page,
        bbox: Tuple[float, float, float, float],
        color: Tuple[float, float, float],
        check_name: str,
        message: str,
        details: str = ""
    ):
        """
        Add a visible rectangle with clickable comment annotation.

        Args:
            page: PyMuPDF page object
            bbox: Bounding box (x0, y0, x1, y1) in absolute coordinates
            color: RGB color tuple (0-1 scale)
            check_name: Name of the validation check
            message: Main error/warning message
            details: Additional details
        """
        rect = fitz.Rect(bbox)

        # 1. Draw visible rectangle border
        shape = page.new_shape()
        shape.draw_rect(rect)
        shape.finish(color=color, width=2, fill=None)
        shape.commit()

        # 2. Add clickable text annotation (comment popup)
        annotation_text = f"{check_name}\n\n{message}"
        if details:
            annotation_text += f"\n\nDetails: {details}"

        # Position comment icon at top-right corner, outside the box
        comment_pos = fitz.Point(rect.x1 + 2, rect.y0)
        annot = page.add_text_annot(
            comment_pos,
            annotation_text,
            icon="Comment"
        )
        annot.set_colors(stroke=color)
        annot.update()

    def _generate_summary_pages(self) -> fitz.Document:
        """Generate summary report pages with improved formatting."""
        summary_doc = fitz.open()

        # Create first summary page
        page = summary_doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        y_pos = MARGIN

        # Collect results
        warnings = [r for r in self.validation_results if not r.passed]
        passed = [r for r in self.validation_results if r.passed]

        # 1. Header box with status
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subtitle = f"Document: {Path(self.pdf_path).name}    Generated: {timestamp}"
        status_text = "PASSED" if self.overall_pass else "WARNINGS FOUND"

        y_pos = self._draw_header_box(
            page=page,
            y=y_pos,
            title="ISPRS Paper Formatting Analysis",
            subtitle=subtitle,
            status_text=status_text,
            status_passed=self.overall_pass,
            stats={"Warnings": len(warnings), "Passed": len(passed)}
        )

        # 2. Extracted Elements table
        elem_rows = []

        # Required Once elements
        for elem_type in ["Title", "Abstract", "Authors", "Affiliations", "Keywords"]:
            count = len(self.enriched_elements.get(elem_type, []))
            status = "✓" if count == 1 else ("✗" if count == 0 else "⚠")
            color = COLOR_PASS if count == 1 else (COLOR_WARN if count > 1 else (0.8, 0, 0))
            elem_rows.append([
                (elem_type, COLOR_DARK),
                (str(count), COLOR_DARK),
                (status, color)
            ])

        # Required Multiple elements
        for elem_type in ["Headings", "References", "Citations"]:
            key = "In_Text_Citations_References" if elem_type == "Citations" else elem_type
            count = len(self.enriched_elements.get(key, []))
            status = "✓" if count > 0 else "✗"
            color = COLOR_PASS if count > 0 else (0.8, 0, 0)
            elem_rows.append([
                (elem_type, COLOR_DARK),
                (str(count), COLOR_DARK),
                (status, color)
            ])

        # Optional elements (only show if present)
        optional_map = [
            ("Sub_Headings", "Sub-Headings"),
            ("Sub_sub_Headings", "Sub-sub-Headings"),
            ("Figures", "Figures"),
            ("Tables", "Tables"),
            ("Main_Text", "Main Text"),
        ]
        for key, label in optional_map:
            count = len(self.enriched_elements.get(key, []))
            if count > 0:
                elem_rows.append([
                    (label, COLOR_GRAY),
                    (str(count), COLOR_GRAY),
                    ("○", COLOR_GRAY)
                ])

        page, y_pos = self._draw_table(
            doc=summary_doc,
            page=page,
            y=y_pos,
            headers=["Element", "Count", "Status"],
            rows=elem_rows,
            col_widths=[200, 60, 60],
            title="Extracted Elements"
        )

        y_pos += 10

        # 3. Validation Results table
        validation_rows = []

        # Add all validation results
        for result in self.validation_results:
            status = "✓" if result.passed else "⚠"
            status_color = COLOR_PASS if result.passed else COLOR_WARN

            # Get detail text (message or truncated details)
            detail = result.message if not result.passed else ""
            if len(detail) > 50:
                detail = detail[:47] + "..."

            validation_rows.append([
                (result.check_name, COLOR_DARK),
                (status, status_color),
                (detail, COLOR_GRAY if result.passed else COLOR_DARK)
            ])

        page, y_pos = self._draw_table(
            doc=summary_doc,
            page=page,
            y=y_pos,
            headers=["Check", "Status", "Details"],
            rows=validation_rows,
            col_widths=[180, 50, 285],
            title="Validation Results"
        )

        # Check if we need a new page for manual reminders
        y_pos = self._check_page_break(summary_doc, page, y_pos + 10, PAGE_HEIGHT, MARGIN)
        if y_pos == MARGIN + 30:
            page = summary_doc[-1]

        y_pos += 10

        # 4. Manual Review Reminders (2-column checklist grid)
        y_pos = self._draw_checklist_grid(
            page=page,
            y=y_pos,
            categories=SUBJECTIVE_REMINDERS,
            title="Manual Review Reminders"
        )

        return summary_doc

    def _draw_text(
        self,
        page: fitz.Page,
        text: str,
        x: float,
        y: float,
        fontsize: int = 10,
        bold: bool = False,
        color: Tuple[float, float, float] = (0, 0, 0)
    ) -> float:
        """
        Draw text on page and return new y position.

        Args:
            page: PyMuPDF page object
            text: Text to draw
            x: X position
            y: Y position
            fontsize: Font size in points
            bold: Whether to use bold font
            color: RGB color tuple

        Returns:
            New y position after text
        """
        fontname = "helv" if not bold else "hebo"
        page.insert_text(
            (x, y + fontsize),
            text,
            fontsize=fontsize,
            fontname=fontname,
            color=color
        )
        return y + fontsize + 4

    def _check_page_break(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        y_pos: float,
        page_height: float,
        margin: float
    ) -> float:
        """Check if new page is needed and create one if necessary."""
        if y_pos > page_height - margin - 50:
            new_page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
            return margin + 30
        return y_pos

    def _draw_header_box(
        self,
        page: fitz.Page,
        y: float,
        title: str,
        subtitle: str,
        status_text: str,
        status_passed: bool,
        stats: Dict[str, int]
    ) -> float:
        """
        Draw a header box with title, subtitle, status, and stats.

        Returns:
            New y position after the box
        """
        box_height = 97
        rect = fitz.Rect(MARGIN, y, PAGE_WIDTH - MARGIN, y + box_height)

        # Draw box background
        page.draw_rect(rect, color=TABLE_BORDER_COLOR, fill=(0.98, 0.98, 0.98), width=1)

        # Draw title
        page.insert_text(
            (MARGIN + 15, y + 22),
            title,
            fontsize=16,
            fontname="hebo",
            color=COLOR_DARK
        )

        # Draw subtitle (document name and timestamp)
        page.insert_text(
            (MARGIN + 15, y + 40),
            subtitle,
            fontsize=9,
            fontname="helv",
            color=COLOR_GRAY
        )

        # Draw ISPRS guidelines link
        page.insert_text(
            (MARGIN + 15, y + 52),
            "Guidelines: https://www.isprs.org/documents/orangebook/app5.aspx",
            fontsize=8,
            fontname="helv",
            color=COLOR_GRAY
        )

        # Draw separator line
        page.draw_line(
            fitz.Point(MARGIN + 10, y + 60),
            fitz.Point(PAGE_WIDTH - MARGIN - 10, y + 60),
            color=TABLE_BORDER_COLOR,
            width=0.5
        )

        # Draw status
        status_color = COLOR_PASS if status_passed else COLOR_WARN
        status_symbol = "✓" if status_passed else "⚠"
        page.insert_text(
            (MARGIN + 15, y + 80),
            f"Status: {status_symbol} {status_text}",
            fontsize=12,
            fontname="hebo",
            color=status_color
        )

        # Draw stats on right side
        stats_text = "    ".join(f"{k}: {v}" for k, v in stats.items())
        page.insert_text(
            (PAGE_WIDTH - MARGIN - 150, y + 80),
            stats_text,
            fontsize=10,
            fontname="helv",
            color=COLOR_DARK
        )

        return y + box_height + 15

    def _draw_table(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        y: float,
        headers: List[str],
        rows: List[List[Tuple[str, Tuple[float, float, float]]]],
        col_widths: List[float],
        title: str = None
    ) -> Tuple[fitz.Page, float]:
        """
        Draw a table with headers and rows, handling page breaks.

        Args:
            doc: PyMuPDF document object (needed to create new pages)
            page: PyMuPDF page object
            y: Starting y position
            headers: List of header strings
            rows: List of rows, each row is list of (text, color) tuples
            col_widths: List of column widths in points
            title: Optional section title above table

        Returns:
            Tuple of (current page, new y position after the table)
        """
        x = MARGIN
        table_width = sum(col_widths)

        # Draw title if provided
        if title:
            page.insert_text(
                (x, y + 12),
                title,
                fontsize=11,
                fontname="hebo",
                color=COLOR_DARK
            )
            y += 20

        # Helper function to draw table headers
        def draw_headers(p: fitz.Page, y_pos: float) -> float:
            header_rect = fitz.Rect(x, y_pos, x + table_width, y_pos + TABLE_HEADER_HEIGHT)
            p.draw_rect(header_rect, color=TABLE_BORDER_COLOR, fill=TABLE_HEADER_BG, width=0.5)
            curr_x = x
            for i, header in enumerate(headers):
                p.insert_text(
                    (curr_x + 8, y_pos + 15),
                    header,
                    fontsize=9,
                    fontname="hebo",
                    color=COLOR_DARK
                )
                curr_x += col_widths[i]
            return y_pos + TABLE_HEADER_HEIGHT

        # Draw initial headers
        header_start_y = y
        y = draw_headers(page, y)

        # Track where headers start on current page (for column separators)
        current_header_y = header_start_y

        # Draw data rows
        for row_idx, row in enumerate(rows):
            # Check if row would exceed page bounds
            if y + TABLE_ROW_HEIGHT > PAGE_HEIGHT - MARGIN:
                # Draw column separators for current page before moving to new page
                curr_x = x
                for width in col_widths[:-1]:
                    curr_x += width
                    page.draw_line(
                        fitz.Point(curr_x, current_header_y),
                        fitz.Point(curr_x, y),
                        color=TABLE_BORDER_COLOR,
                        width=0.5
                    )

                # Create new page and redraw headers
                page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
                y = MARGIN
                current_header_y = y
                y = draw_headers(page, y)

            row_rect = fitz.Rect(x, y, x + table_width, y + TABLE_ROW_HEIGHT)

            # Alternating row background
            if row_idx % 2 == 1:
                page.draw_rect(row_rect, color=None, fill=TABLE_ALT_ROW_BG)

            # Draw row border
            page.draw_rect(row_rect, color=TABLE_BORDER_COLOR, fill=None, width=0.5)

            # Draw cell text
            curr_x = x
            for i, (text, color) in enumerate(row):
                # Truncate text if too long
                max_chars = int(col_widths[i] / 5)  # Approximate chars that fit
                display_text = text[:max_chars] + "..." if len(text) > max_chars else text

                page.insert_text(
                    (curr_x + 8, y + 13),
                    display_text,
                    fontsize=9,
                    fontname="helv",
                    color=color
                )
                curr_x += col_widths[i]

            y += TABLE_ROW_HEIGHT

        # Draw vertical column separators for final page
        curr_x = x
        for width in col_widths[:-1]:
            curr_x += width
            page.draw_line(
                fitz.Point(curr_x, current_header_y),
                fitz.Point(curr_x, y),
                color=TABLE_BORDER_COLOR,
                width=0.5
            )

        return page, y + 10

    def _draw_checklist_grid(
        self,
        page: fitz.Page,
        y: float,
        categories: List[Dict],
        title: str = None
    ) -> float:
        """
        Draw a 2-column checklist grid.

        Args:
            page: PyMuPDF page object
            y: Starting y position
            categories: List of category dicts with 'category' and 'items' keys
            title: Optional section title

        Returns:
            New y position after the grid
        """
        x = MARGIN
        col_width = CONTENT_WIDTH / 2 - 5

        # Draw title if provided
        if title:
            page.insert_text(
                (x, y + 12),
                title,
                fontsize=11,
                fontname="hebo",
                color=COLOR_INFO
            )
            y += 8
            page.insert_text(
                (x, y + 12),
                "(Items requiring manual verification)",
                fontsize=8,
                fontname="helv",
                color=COLOR_GRAY
            )
            y += 20

        # Pair up categories for 2-column layout
        for i in range(0, len(categories), 2):
            left_cat = categories[i]
            right_cat = categories[i + 1] if i + 1 < len(categories) else None

            # Calculate row heights
            left_items = len(left_cat['items'])
            right_items = len(right_cat['items']) if right_cat else 0
            max_items = max(left_items, right_items)
            box_height = 22 + max_items * 14

            # Draw left box
            left_rect = fitz.Rect(x, y, x + col_width, y + box_height)
            page.draw_rect(left_rect, color=TABLE_BORDER_COLOR, fill=(0.98, 0.98, 1.0), width=0.5)

            # Left category title
            page.insert_text(
                (x + 8, y + 14),
                left_cat['category'],
                fontsize=9,
                fontname="hebo",
                color=COLOR_INFO
            )

            # Left items
            item_y = y + 28
            for item in left_cat['items']:
                page.insert_text(
                    (x + 12, item_y),
                    f"☐ {item[:45]}{'...' if len(item) > 45 else ''}",
                    fontsize=8,
                    fontname="helv",
                    color=COLOR_DARK
                )
                item_y += 14

            # Draw right box if exists
            if right_cat:
                right_x = x + col_width + 10
                right_rect = fitz.Rect(right_x, y, right_x + col_width, y + box_height)
                page.draw_rect(right_rect, color=TABLE_BORDER_COLOR, fill=(0.98, 0.98, 1.0), width=0.5)

                # Right category title
                page.insert_text(
                    (right_x + 8, y + 14),
                    right_cat['category'],
                    fontsize=9,
                    fontname="hebo",
                    color=COLOR_INFO
                )

                # Right items
                item_y = y + 28
                for item in right_cat['items']:
                    page.insert_text(
                        (right_x + 12, item_y),
                        f"☐ {item[:45]}{'...' if len(item) > 45 else ''}",
                        fontsize=8,
                        fontname="helv",
                        color=COLOR_DARK
                    )
                    item_y += 14

            y += box_height + 5

        return y + 5


def generate_report(
    pdf_path: str,
    validation_results: List[ValidationResult],
    enriched_elements: Dict[str, List],
    overall_pass: bool,
    output_path: Optional[str] = None
) -> str:
    """
    Convenience function to generate a PDF report.

    Args:
        pdf_path: Path to original PDF
        validation_results: List of validation results
        enriched_elements: Dictionary of enriched elements
        overall_pass: Overall pass/fail status
        output_path: Optional output path

    Returns:
        Path to generated report
    """
    generator = ReportGenerator(
        pdf_path=pdf_path,
        validation_results=validation_results,
        enriched_elements=enriched_elements,
        overall_pass=overall_pass
    )
    return generator.generate_report(output_path)
