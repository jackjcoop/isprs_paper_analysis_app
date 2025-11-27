"""
PyMuPDF Font and Style Extractor Module
Extracts text with font metadata using bounding boxes from Document AI.
"""

import fitz  # PyMuPDF
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class TextSpan:
    """Represents a text span with styling information."""
    text: str
    font_name: str
    font_size: float
    is_bold: bool
    is_italic: bool
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)


@dataclass
class EnrichedElement:
    """Document element with text, bounding box, and font metadata."""
    element_type: str
    text: str
    bbox: Tuple[float, float, float, float]
    page: int
    font_size: Optional[float] = None
    font_name: Optional[str] = None
    is_bold: Optional[bool] = None
    is_italic: Optional[bool] = None
    spans: Optional[List[TextSpan]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        return result


class PyMuPDFExtractor:
    """Extracts font and style information from PDFs using PyMuPDF."""

    def __init__(self, pdf_path: str):
        """
        Initialize extractor with a PDF document.

        Args:
            pdf_path: Path to PDF file
        """
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

    def close(self):
        """Close the PDF document."""
        if self.doc:
            self.doc.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def extract_text_from_bbox(
        self,
        page_num: int,
        bbox: Tuple[float, float, float, float]
    ) -> List[TextSpan]:
        """
        Extract text spans with font information from a bounding box.

        Args:
            page_num: Page number (0-indexed)
            bbox: Bounding box coordinates (x0, y0, x1, y1) in absolute units

        Returns:
            List of text spans with font metadata
        """
        if page_num >= len(self.doc):
            return []

        page = self.doc[page_num]
        rect = fitz.Rect(bbox)

        # Get text as dictionary with detailed information
        text_dict = page.get_text("dict", clip=rect)

        spans = []

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # Not a text block
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text.strip():
                        continue

                    font_name = span.get("font", "")
                    font_size = span.get("size", 0.0)
                    flags = span.get("flags", 0)
                    span_bbox = span.get("bbox", (0, 0, 0, 0))

                    # Decode font flags
                    is_bold = bool(flags & 16)  # Bit 4
                    is_italic = bool(flags & 2)  # Bit 1

                    text_span = TextSpan(
                        text=text,
                        font_name=font_name,
                        font_size=round(font_size, 2),
                        is_bold=is_bold,
                        is_italic=is_italic,
                        bbox=span_bbox
                    )
                    spans.append(text_span)

        return spans

    def get_dominant_font_info(
        self,
        spans: List[TextSpan]
    ) -> Tuple[Optional[str], Optional[float], Optional[bool], Optional[bool]]:
        """
        Get dominant font characteristics from a list of spans.

        Args:
            spans: List of text spans

        Returns:
            Tuple of (font_name, font_size, is_bold, is_italic)
        """
        if not spans:
            return None, None, None, None

        # Count occurrences of each font characteristic
        font_names = {}
        font_sizes = {}
        bold_count = 0
        italic_count = 0

        for span in spans:
            # Count font names
            font_names[span.font_name] = font_names.get(span.font_name, 0) + 1

            # Count font sizes
            font_sizes[span.font_size] = font_sizes.get(span.font_size, 0) + 1

            # Count bold/italic
            if span.is_bold:
                bold_count += 1
            if span.is_italic:
                italic_count += 1

        # Get dominant characteristics
        dominant_font_name = max(font_names, key=font_names.get) if font_names else None
        dominant_font_size = max(font_sizes, key=font_sizes.get) if font_sizes else None
        is_bold = bold_count > len(spans) / 2
        is_italic = italic_count > len(spans) / 2

        return dominant_font_name, dominant_font_size, is_bold, is_italic

    def enrich_element(
        self,
        element_type: str,
        text: str,
        bbox: Tuple[float, float, float, float],
        page: int
    ) -> EnrichedElement:
        """
        Enrich an element with font and style information.

        Args:
            element_type: Type of element
            text: Element text
            bbox: Bounding box (normalized coordinates)
            page: Page number (0-indexed)

        Returns:
            Enriched element with font metadata
        """
        # Convert normalized coordinates to absolute
        if page < len(self.doc):
            page_obj = self.doc[page]
            page_width = page_obj.rect.width
            page_height = page_obj.rect.height

            abs_bbox = (
                bbox[0] * page_width,
                bbox[1] * page_height,
                bbox[2] * page_width,
                bbox[3] * page_height
            )
        else:
            abs_bbox = bbox

        # Extract text spans from bounding box
        spans = self.extract_text_from_bbox(page, abs_bbox)

        # Get dominant font characteristics
        font_name, font_size, is_bold, is_italic = self.get_dominant_font_info(spans)

        return EnrichedElement(
            element_type=element_type,
            text=text,
            bbox=abs_bbox,
            page=page,
            font_size=font_size,
            font_name=font_name,
            is_bold=is_bold,
            is_italic=is_italic,
            spans=spans
        )

    def verify_label(
        self,
        label_text: str,
        bbox: Tuple[float, float, float, float],
        page: int,
        direction: str = "left"
    ) -> Tuple[bool, str, bool]:
        """
        Verify if a label (e.g., "Keywords:", "Abstract") exists near a bounding box.

        Args:
            label_text: Text to search for (e.g., "Keywords", "Abstract")
            bbox: Bounding box of the content (normalized)
            page: Page number (0-indexed)
            direction: Direction to search ("left", "above", "below", "right")

        Returns:
            Tuple of (found, actual_text, is_correct_format)
            - found: True if any variant of the label was found
            - actual_text: The actual text found (e.g., "Key words:")
            - is_correct_format: True if properly formatted (no space)
        """
        if page >= len(self.doc):
            return (False, "", True)

        page_obj = self.doc[page]
        page_width = page_obj.rect.width
        page_height = page_obj.rect.height

        # Convert to absolute coordinates
        x0, y0, x1, y1 = (
            bbox[0] * page_width,
            bbox[1] * page_height,
            bbox[2] * page_width,
            bbox[3] * page_height
        )

        # Define search area based on direction
        search_margin = 100  # pixels

        if direction == "left":
            # Search to the left of the bounding box
            search_bbox = (
                max(0, x0 - search_margin),
                y0,
                x0,
                y1
            )
        elif direction == "above":
            # Search above the bounding box
            search_bbox = (
                x0,
                max(0, y0 - search_margin),
                x1,
                y0
            )
        elif direction == "below":
            # Search below the bounding box
            search_bbox = (
                x0,
                y1,
                x1,
                min(page_height, y1 + search_margin)
            )
        elif direction == "right":
            # Search to the right of the bounding box
            search_bbox = (
                x1,
                y0,
                min(page_width, x1 + search_margin),
                y1
            )
        else:
            return (False, "", True)

        # Extract text from search area
        rect = fitz.Rect(search_bbox)
        text = page_obj.get_text("text", clip=rect)

        # Check if label text is present (case-insensitive)
        label_lower = label_text.lower().strip()
        label_upper = label_text.upper().strip()
        text_lower = text.lower().strip()
        text_stripped = text.strip()

        # Check for correct format first (proper case: "Keywords", "Abstract")
        if label_text in text_stripped or f"{label_text}:" in text_stripped:
            return (True, label_text, True)

        # Check for ALL CAPS variation (incorrect format: "KEYWORDS", "ABSTRACT")
        if label_upper in text_stripped or f"{label_upper}:" in text_stripped:
            return (True, label_upper, False)

        # Check for lowercase or other case variations (found but incorrect case)
        if label_lower in text_lower or f"{label_lower}:" in text_lower:
            # Found in some form but not proper case - extract actual text
            return (True, label_lower, False)

        # Check for "key word" or "key words" variations (incorrect format)
        if label_lower == "keywords":
            incorrect_patterns = ["key word", "key words", "key-word", "key-words"]
            for pattern in incorrect_patterns:
                if pattern in text_lower or f"{pattern}:" in text_lower:
                    return (True, pattern, False)

        return (False, "", True)

    def expand_bbox_and_extract(
        self,
        bbox: Tuple[float, float, float, float],
        page: int,
        direction: str,
        expansion: float = 100
    ) -> Tuple[str, List[TextSpan]]:
        """
        Expand bounding box in a direction and extract text.

        Args:
            bbox: Original bounding box (normalized)
            page: Page number
            direction: Direction to expand ("left", "above", etc.)
            expansion: Pixels to expand

        Returns:
            Tuple of (extracted_text, text_spans)
        """
        if page >= len(self.doc):
            return "", []

        page_obj = self.doc[page]
        page_width = page_obj.rect.width
        page_height = page_obj.rect.height

        # Convert to absolute
        x0, y0, x1, y1 = (
            bbox[0] * page_width,
            bbox[1] * page_height,
            bbox[2] * page_width,
            bbox[3] * page_height
        )

        # Expand based on direction
        if direction == "left":
            x0 = max(0, x0 - expansion)
        elif direction == "above":
            y0 = max(0, y0 - expansion)
        elif direction == "below":
            y1 = min(page_height, y1 + expansion)
        elif direction == "right":
            x1 = min(page_width, x1 + expansion)

        expanded_bbox = (x0, y0, x1, y1)

        # Extract text spans
        spans = self.extract_text_from_bbox(page, expanded_bbox)

        # Combine text
        text = " ".join(span.text for span in spans)

        return text, spans

    def get_page_dimensions(self, page_num: int) -> Tuple[float, float]:
        """
        Get page dimensions.

        Args:
            page_num: Page number (0-indexed)

        Returns:
            Tuple of (width, height)
        """
        if page_num < len(self.doc):
            page = self.doc[page_num]
            return page.rect.width, page.rect.height
        return 0, 0
