"""
Output Generator Module
Formats extracted data and validation results as JSON.
"""

import json
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path


class OutputGenerator:
    """Generates JSON output with extracted text and metadata."""

    @staticmethod
    def generate_json_output(
        pdf_path: str,
        extracted_elements: Dict[str, List[Any]],
        validation_results: List[Any],
        overall_pass: bool,
        labels_verified: Dict[str, bool]
    ) -> Dict:
        """
        Generate comprehensive JSON output.

        Args:
            pdf_path: Path to analyzed PDF
            extracted_elements: Extracted document elements
            validation_results: Validation results
            overall_pass: Overall validation status
            labels_verified: Label verification results

        Returns:
            Dictionary ready for JSON serialization
        """
        output = {
            "document_info": {
                "file_path": str(Path(pdf_path).absolute()),
                "file_name": Path(pdf_path).name,
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_version": "app_3_v1.0"
            },
            "validation": {
                "overall_pass": overall_pass,
                "total_checks": len(validation_results),
                "errors": sum(
                    1 for r in validation_results
                    if r.severity.value == "ERROR" and not r.passed
                ),
                "warnings": sum(
                    1 for r in validation_results
                    if r.severity.value == "WARNING" and not r.passed
                ),
                "passed": sum(1 for r in validation_results if r.passed),
                "checks": [
                    {
                        "name": r.check_name,
                        "passed": r.passed,
                        "severity": r.severity.value,
                        "message": r.message,
                        "details": r.details
                    }
                    for r in validation_results
                ]
            },
            "label_verification": labels_verified,
            "extracted_elements": {},
            "statistics": {}
        }

        # Convert extracted elements to serializable format
        for element_type, elements in extracted_elements.items():
            if not elements:
                continue

            serialized_elements = []
            for element in elements:
                if hasattr(element, 'to_dict'):
                    serialized_elements.append(element.to_dict())
                elif hasattr(element, '__dict__'):
                    # Convert dataclass or object to dict
                    elem_dict = {}
                    for key, value in element.__dict__.items():
                        if key == 'spans' and value:
                            # Convert spans to dicts
                            elem_dict[key] = [
                                {
                                    'text': s.text,
                                    'font_name': s.font_name,
                                    'font_size': s.font_size,
                                    'is_bold': s.is_bold,
                                    'is_italic': s.is_italic,
                                    'bbox': s.bbox
                                }
                                for s in value
                            ]
                        else:
                            elem_dict[key] = value
                    serialized_elements.append(elem_dict)
                else:
                    # Fallback for simple types
                    serialized_elements.append(str(element))

            output["extracted_elements"][element_type] = serialized_elements

        # Calculate statistics
        stats = OutputGenerator._calculate_statistics(extracted_elements)
        output["statistics"] = stats

        return output

    @staticmethod
    def _calculate_statistics(extracted_elements: Dict[str, List[Any]]) -> Dict:
        """
        Calculate statistics about extracted elements.

        Args:
            extracted_elements: Extracted document elements

        Returns:
            Dictionary of statistics
        """
        stats = {
            "element_counts": {},
            "font_analysis": {
                "unique_fonts": set(),
                "font_sizes": set()
            }
        }

        # Count elements
        for element_type, elements in extracted_elements.items():
            stats["element_counts"][element_type] = len(elements)

            # Analyze fonts
            for element in elements:
                if hasattr(element, 'font_name') and element.font_name:
                    stats["font_analysis"]["unique_fonts"].add(element.font_name)
                if hasattr(element, 'font_size') and element.font_size:
                    stats["font_analysis"]["font_sizes"].add(element.font_size)

        # Convert sets to sorted lists for JSON serialization
        stats["font_analysis"]["unique_fonts"] = sorted(
            list(stats["font_analysis"]["unique_fonts"])
        )
        stats["font_analysis"]["font_sizes"] = sorted(
            list(stats["font_analysis"]["font_sizes"])
        )

        return stats

    @staticmethod
    def save_json(output_dict: Dict, output_path: str):
        """
        Save JSON output to file.

        Args:
            output_dict: Dictionary to save
            output_path: Path for output file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_dict, f, indent=2, ensure_ascii=False)

    @staticmethod
    def get_default_output_path(pdf_path: str) -> str:
        """
        Get default output path for JSON file.

        Args:
            pdf_path: Path to input PDF

        Returns:
            Path for output JSON file
        """
        pdf_path = Path(pdf_path)
        output_path = pdf_path.parent / f"{pdf_path.stem}_analysis.json"
        return str(output_path)

    @staticmethod
    def format_element_summary(extracted_elements: Dict[str, List[Any]]) -> str:
        """
        Format a brief summary of extracted elements.

        Args:
            extracted_elements: Extracted document elements

        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("\nEXTRACTED ELEMENTS SUMMARY:")
        lines.append("-" * 70)

        # Required once
        lines.append("\nRequired Once:")
        for elem_type in ["Title", "Abstract", "Authors", "Affiliations", "Keywords"]:
            count = len(extracted_elements.get(elem_type, []))
            symbol = "✓" if count == 1 else ("✗" if count == 0 else "⚠")
            lines.append(f"  {symbol} {elem_type}: {count}")

        # Required multiple
        lines.append("\nRequired Multiple:")
        for elem_type in ["Headings", "In_Text_Citations_References", "References"]:
            count = len(extracted_elements.get(elem_type, []))
            symbol = "✓" if count > 0 else "✗"
            lines.append(f"  {symbol} {elem_type}: {count}")

        # Optional (if found)
        lines.append("\nOptional Elements Found:")
        optional_types = [
            "Abstract_title", "Keywords_title",
            "Sub_Headings", "Sub_sub_Headings", "Main_Text",
            "Figure_Number", "Figure_Title", "Table_Number", "Table_Title",
            "Equation", "Equation_Number",
            "In_Text_Citations_Figures", "In_Text_Citations_Tables", "Table",
            "Reference_Partial"
        ]
        for elem_type in optional_types:
            count = len(extracted_elements.get(elem_type, []))
            if count > 0:
                lines.append(f"  ℹ {elem_type}: {count}")

        return "\n".join(lines)
