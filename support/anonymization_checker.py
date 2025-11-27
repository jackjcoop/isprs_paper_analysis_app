"""
Anonymization Checker Module
Uses spaCy NER to detect named entities for anonymization validation.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Entity:
    """Represents a named entity."""
    text: str
    label: str  # PERSON, ORG, GPE, etc.
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class AnonymizationResult:
    """Result of anonymization check."""
    is_anonymous: bool
    entities_found: List[Entity]
    total_person_entities: int
    total_org_entities: int
    total_gpe_entities: int
    sections_checked: List[str]


class AnonymizationChecker:
    """Checks for named entities to validate anonymization."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize anonymization checker with spaCy.

        Args:
            model_name: spaCy model name (default: en_core_web_sm)
        """
        self.model_name = model_name
        self.nlp = None
        self._load_model()

    def _load_model(self):
        """Load spaCy model with error handling."""
        try:
            import spacy
            try:
                self.nlp = spacy.load(self.model_name)
            except OSError:
                # Model not found, suggest download
                raise RuntimeError(
                    f"spaCy model '{self.model_name}' not found. "
                    f"Please install it with: python -m spacy download {self.model_name}"
                )
        except ImportError:
            raise RuntimeError(
                "spaCy is not installed. Please install it with: pip install spacy"
            )

    def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None
    ) -> List[Entity]:
        """
        Extract named entities from text.

        Args:
            text: Text to analyze
            entity_types: List of entity types to extract (default: PERSON, ORG, GPE)

        Returns:
            List of Entity objects
        """
        if entity_types is None:
            entity_types = ['PERSON', 'ORG', 'GPE']

        if not self.nlp:
            return []

        # Process text with spaCy
        doc = self.nlp(text)

        # Extract entities
        entities = []
        for ent in doc.ents:
            if ent.label_ in entity_types:
                entity = Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char
                )
                entities.append(entity)

        return entities

    def check_anonymization(
        self,
        extracted_elements: Dict[str, List],
        sections_to_check: List[str] = None
    ) -> AnonymizationResult:
        """
        Check if specified sections are properly anonymized.

        Args:
            extracted_elements: Dictionary of extracted elements
            sections_to_check: List of section names to check (default: Authors, Affiliations)

        Returns:
            AnonymizationResult object
        """
        if sections_to_check is None:
            sections_to_check = ['Authors', 'Affiliations']

        all_entities = []
        person_count = 0
        org_count = 0
        gpe_count = 0

        # Check each specified section
        for section_name in sections_to_check:
            elements = extracted_elements.get(section_name, [])

            for element in elements:
                # Extract entities from element text
                entities = self.extract_entities(element.text)

                for entity in entities:
                    all_entities.append(entity)

                    # Count by type
                    if entity.label == 'PERSON':
                        person_count += 1
                    elif entity.label == 'ORG':
                        org_count += 1
                    elif entity.label == 'GPE':
                        gpe_count += 1

        # Determine if anonymous
        # Consider it non-anonymous if PERSON entities are found
        # ORG and GPE might be acceptable (e.g., "University A", "Country X")
        is_anonymous = person_count == 0

        return AnonymizationResult(
            is_anonymous=is_anonymous,
            entities_found=all_entities,
            total_person_entities=person_count,
            total_org_entities=org_count,
            total_gpe_entities=gpe_count,
            sections_checked=sections_to_check
        )

    def get_anonymization_suggestions(
        self,
        entities: List[Entity]
    ) -> List[str]:
        """
        Get suggestions for anonymizing detected entities.

        Args:
            entities: List of entities found

        Returns:
            List of suggestion strings
        """
        suggestions = []

        person_entities = [e for e in entities if e.label == 'PERSON']
        org_entities = [e for e in entities if e.label == 'ORG']

        if person_entities:
            suggestions.append(
                f"Found {len(person_entities)} person name(s): "
                f"{', '.join(e.text for e in person_entities[:5])}"
                f"{'...' if len(person_entities) > 5 else ''}"
            )
            suggestions.append(
                "Suggestion: Replace with 'Author 1', 'Author 2', etc."
            )

        if org_entities:
            suggestions.append(
                f"Found {len(org_entities)} organization(s): "
                f"{', '.join(e.text for e in org_entities[:5])}"
                f"{'...' if len(org_entities) > 5 else ''}"
            )
            suggestions.append(
                "Suggestion: Replace with 'Institution A', 'Institution B', etc."
            )

        return suggestions

    def check_author_section_format(
        self,
        author_text: str,
        is_anonymous: bool
    ) -> Dict[str, Any]:
        """
        Check if author section follows proper format.

        Args:
            author_text: Text from authors section
            is_anonymous: Whether document should be anonymous

        Returns:
            Dictionary with format validation results
        """
        result = {
            'is_valid': True,
            'issues': []
        }

        if is_anonymous:
            # Check for patterns that indicate non-anonymization
            non_anon_patterns = [
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last name
                r'\b[A-Z]\.\s*[A-Z]\.\s*[A-Z][a-z]+\b',  # Initials and last name
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email
            ]

            import re
            for pattern in non_anon_patterns:
                if re.search(pattern, author_text):
                    result['is_valid'] = False
                    result['issues'].append(
                        f"Detected potential non-anonymous content: {pattern}"
                    )

            # Check for expected anonymous patterns
            if not any(phrase in author_text.lower() for phrase in ['author', 'anonymous', 'blinded']):
                result['issues'].append(
                    "Warning: Anonymous submission should typically include "
                    "'Author 1', 'Author 2', or similar placeholder"
                )

        else:
            # For non-anonymous, check for required elements
            entities = self.extract_entities(author_text, ['PERSON'])

            if len(entities) == 0:
                result['issues'].append(
                    "Warning: No author names detected. "
                    "Camera-ready version should include author names."
                )

            # Check for email addresses
            import re
            if not re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', author_text):
                result['issues'].append(
                    "Warning: No email addresses found. "
                    "Camera-ready version typically requires contact email."
                )

        return result


def check_anonymization(
    extracted_elements: Dict[str, List],
    sections_to_check: List[str] = None
) -> AnonymizationResult:
    """
    Convenience function to check anonymization.

    Args:
        extracted_elements: Dictionary of extracted elements
        sections_to_check: Sections to check (default: Authors, Affiliations)

    Returns:
        AnonymizationResult object
    """
    checker = AnonymizationChecker()
    return checker.check_anonymization(extracted_elements, sections_to_check)
