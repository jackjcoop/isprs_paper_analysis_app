"""
Support modules for PDF Compliance Analyzer.

This package contains the core functionality for document analysis:
- converter: Document format conversion (Word/LaTeX to PDF)
- document_ai_client: Google Cloud Document AI integration
- pymupdf_extractor: Font and style information extraction
- validator: Schema compliance validation
- output_generator: JSON output formatting
"""

from .converter import DocumentConverter, convert_document
from .document_ai_client import DocumentAIClient, BoundingBox, ExtractedElement
from .pymupdf_extractor import PyMuPDFExtractor, TextSpan, EnrichedElement
from .validator import ComplianceValidator, ValidationResult, Severity
from .output_generator import OutputGenerator
from .citation_validator import CitationValidator, ParsedCitation, ParsedReference, CitationMatch
from .anonymization_checker import AnonymizationChecker, Entity, AnonymizationResult, check_anonymization

__all__ = [
    'DocumentConverter',
    'convert_document',
    'DocumentAIClient',
    'BoundingBox',
    'ExtractedElement',
    'PyMuPDFExtractor',
    'TextSpan',
    'EnrichedElement',
    'ComplianceValidator',
    'ValidationResult',
    'Severity',
    'OutputGenerator',
    'CitationValidator',
    'ParsedCitation',
    'ParsedReference',
    'CitationMatch',
    'AnonymizationChecker',
    'Entity',
    'AnonymizationResult',
    'check_anonymization',
]

__version__ = '1.0.0'
