"""
Google Cloud Document AI Client Module
Extracts text and bounding boxes from PDF documents using Document AI.
"""

import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account


@dataclass
class BoundingBox:
    """Represents a bounding box with normalized coordinates."""
    x0: float  # Left
    y0: float  # Top
    x1: float  # Right
    y1: float  # Bottom
    page: int  # Page number (0-indexed)

    def to_absolute(self, page_width: float, page_height: float) -> Tuple[float, float, float, float]:
        """Convert normalized coordinates to absolute pixel coordinates."""
        return (
            self.x0 * page_width,
            self.y0 * page_height,
            self.x1 * page_width,
            self.y1 * page_height
        )


@dataclass
class ExtractedElement:
    """Represents an extracted document element with text and bounding box."""
    element_type: str  # e.g., "Title", "Abstract", "Headings"
    text: str
    bbox: BoundingBox
    confidence: float = 1.0


class DocumentAIClient:
    """Client for Google Cloud Document AI API."""

    def __init__(
        self,
        project_id: str = "582708116407",
        location: str = "us",
        processor_id: str = "bbfe5b3adc33b351",
        credentials_path: Optional[str] = None,
        credentials_info: Optional[Dict] = None
    ):
        """
        Initialize Document AI client.

        Args:
            project_id: Google Cloud project ID
            location: Processor location
            processor_id: Document AI processor ID
            credentials_path: Path to service account credentials JSON
            credentials_info: Service account credentials as dict (for Streamlit secrets)
        """
        self.project_id = project_id
        self.location = location
        self.processor_id = processor_id

        # Set up credentials (priority: credentials_info > credentials_path > default)
        if credentials_info:
            credentials = service_account.Credentials.from_service_account_info(
                credentials_info
            )
            self.client = documentai.DocumentProcessorServiceClient(
                credentials=credentials
            )
        elif credentials_path and os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            self.client = documentai.DocumentProcessorServiceClient(
                credentials=credentials
            )
        else:
            # Use default credentials (GOOGLE_APPLICATION_CREDENTIALS env var)
            self.client = documentai.DocumentProcessorServiceClient()

        self.processor_name = self.client.processor_path(
            self.project_id, self.location, self.processor_id
        )

    def process_document(self, pdf_path: str) -> documentai.Document:
        """
        Process a PDF document using Document AI.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Processed document object from Document AI
        """
        # Read the file
        with open(pdf_path, "rb") as pdf_file:
            pdf_content = pdf_file.read()

        # Create the document
        raw_document = documentai.RawDocument(
            content=pdf_content,
            mime_type="application/pdf"
        )

        # Configure the process request
        request = documentai.ProcessRequest(
            name=self.processor_name,
            raw_document=raw_document
        )

        # Process the document
        result = self.client.process_document(request=request)

        return result.document

    def extract_elements_by_type(
        self,
        document: documentai.Document
    ) -> Dict[str, List[ExtractedElement]]:
        """
        Extract document elements organized by type.

        Args:
            document: Processed document from Document AI

        Returns:
            Dictionary mapping element types to lists of extracted elements
        """
        elements_by_type = {
            "Abstract": [],
            "Abstract_title": [],
            "Affiliations": [],
            "Authors": [],
            "Equation": [],
            "Equation_Number": [],
            "Figure_Number": [],
            "Figure_Title": [],
            "Headings": [],
            "In_Text_Citations_Figures": [],
            "In_Text_Citations_References": [],
            "In_Text_Citations_Tables": [],
            "Keywords": [],
            "Keywords_title": [],
            "Main_Text": [],
            "References": [],
            "Sub_Headings": [],
            "Sub_sub_Headings": [],
            "Table": [],
            "Table_Number": [],
            "Table_Title": [],
            "Title": [],
            "Reference_Partial": []
        }

        # Extract from document entities
        for entity in document.entities:
            entity_type = entity.type_

            # Map Document AI types to our schema
            # Note: Document AI may use different naming conventions
            if entity_type in elements_by_type:
                bbox = self._extract_bounding_box(entity, document)
                if bbox:
                    element = ExtractedElement(
                        element_type=entity_type,
                        text=entity.mention_text,
                        bbox=bbox,
                        confidence=entity.confidence
                    )
                    elements_by_type[entity_type].append(element)

        return elements_by_type

    def _extract_bounding_box(
        self,
        entity: documentai.Document.Entity,
        document: documentai.Document
    ) -> Optional[BoundingBox]:
        """Extract bounding box from entity."""
        if not entity.page_anchor or not entity.page_anchor.page_refs:
            return None

        page_ref = entity.page_anchor.page_refs[0]
        page_num = page_ref.page

        if not page_ref.bounding_poly:
            return None

        vertices = page_ref.bounding_poly.normalized_vertices

        if not vertices:
            return None

        # Get min/max coordinates
        x_coords = [v.x for v in vertices]
        y_coords = [v.y for v in vertices]

        return BoundingBox(
            x0=min(x_coords),
            y0=min(y_coords),
            x1=max(x_coords),
            y1=max(y_coords),
            page=page_num
        )

    def _get_bbox_from_layout(
        self,
        layout: documentai.Document.Page.Layout,
        page_num: int
    ) -> Optional[BoundingBox]:
        """Extract bounding box from layout object."""
        if not layout.bounding_poly:
            return None

        vertices = layout.bounding_poly.normalized_vertices

        if not vertices:
            return None

        x_coords = [v.x for v in vertices]
        y_coords = [v.y for v in vertices]

        return BoundingBox(
            x0=min(x_coords),
            y0=min(y_coords),
            x1=max(x_coords),
            y1=max(y_coords),
            page=page_num
        )

