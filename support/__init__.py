"""
Support modules for PDF Compliance Analyzer.

This package contains the core functionality for document analysis:
- converter: Document format conversion (Word/LaTeX to PDF)
- document_ai_client: Google Cloud Document AI integration
- pymupdf_extractor: Font and style information extraction
- validator: Schema compliance validation
- output_generator: JSON output formatting
"""

# Lazy-safe imports: each submodule is imported independently so that a
# missing third-party dependency (e.g. google-cloud for Document AI) does
# not prevent the rest of the package from loading.
# We catch Exception (not just ImportError) because Python's import
# machinery can raise KeyError when a partially-loaded module leaves a
# corrupt entry in sys.modules.
try:
    from .converter import DocumentConverter, convert_document
except Exception:
    pass

try:
    from .document_ai_client import DocumentAIClient, BoundingBox, ExtractedElement
except Exception:
    pass

try:
    from .pymupdf_extractor import PyMuPDFExtractor, TextSpan, EnrichedElement
except Exception:
    pass

try:
    from .validator import ComplianceValidator, ValidationResult, Severity
except Exception:
    pass

try:
    from .output_generator import OutputGenerator
except Exception:
    pass

try:
    from .citation_validator import CitationValidator, ParsedCitation, ParsedReference, CitationMatch
except Exception:
    pass

try:
    from .anonymization_checker import AnonymizationChecker, Entity, AnonymizationResult, check_anonymization
except Exception:
    pass

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
