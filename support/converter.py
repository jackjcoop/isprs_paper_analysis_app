"""
Document Converter Module
Converts Word (.docx) and LaTeX (.tex) documents to PDF format.
"""

import subprocess
from pathlib import Path


class DocumentConverter:
    """Handles conversion of various document formats to PDF."""

    @staticmethod
    def convert_to_pdf(input_path: str) -> str:
        """
        Convert a document to PDF format if needed.

        Args:
            input_path: Path to the input document

        Returns:
            Path to the PDF file (original if already PDF, converted otherwise)

        Raises:
            ValueError: If file format is not supported
            RuntimeError: If conversion fails
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        extension = input_path.suffix.lower()

        # Already a PDF - no conversion needed
        if extension == '.pdf':
            return str(input_path)

        # Word document - use docx2pdf
        elif extension in ['.docx', '.doc']:
            return DocumentConverter._convert_word_to_pdf(input_path)

        # LaTeX document - use pdflatex via subprocess
        elif extension == '.tex':
            return DocumentConverter._convert_latex_to_pdf(input_path)

        else:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                "Supported formats: .pdf, .docx, .doc, .tex"
            )

    @staticmethod
    def _convert_word_to_pdf(docx_path: Path) -> str:
        """
        Convert Word document to PDF using docx2pdf.

        Args:
            docx_path: Path to Word document

        Returns:
            Path to converted PDF
        """
        try:
            from docx2pdf import convert
        except ImportError:
            raise RuntimeError(
                "Word to PDF conversion requires docx2pdf. "
                "Install it with: pip install docx2pdf"
            )

        output_path = docx_path.with_suffix('.pdf')

        try:
            convert(str(docx_path), str(output_path))

            if output_path.exists():
                return str(output_path)
            else:
                raise RuntimeError(
                    f"docx2pdf conversion failed - output file not created"
                )

        except Exception as e:
            raise RuntimeError(
                f"Word to PDF conversion failed: {str(e)}"
            )

    @staticmethod
    def _convert_latex_to_pdf(tex_path: Path) -> str:
        """
        Convert LaTeX document to PDF using pdflatex.

        Args:
            tex_path: Path to LaTeX document

        Returns:
            Path to converted PDF
        """
        try:
            output_dir = tex_path.parent
            output_path = tex_path.with_suffix('.pdf')

            # Run pdflatex (may need multiple passes for references)
            for _ in range(2):  # Two passes for cross-references
                cmd = [
                    'pdflatex',
                    '-interaction=nonstopmode',
                    '-output-directory', str(output_dir),
                    str(tex_path)
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(output_dir)
                )

            if output_path.exists():
                # Clean up auxiliary files
                for ext in ['.aux', '.log', '.out']:
                    aux_file = tex_path.with_suffix(ext)
                    if aux_file.exists():
                        aux_file.unlink()

                return str(output_path)
            else:
                raise RuntimeError(
                    f"pdflatex conversion failed. "
                    f"Check LaTeX syntax in {tex_path}"
                )

        except FileNotFoundError:
            raise RuntimeError(
                "LaTeX to PDF conversion requires pdflatex. "
                "Please install a TeX distribution (e.g., TeX Live, MiKTeX) "
                "or convert the document manually."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"LaTeX compilation timed out for {tex_path}"
            )


def convert_document(input_path: str) -> str:
    """
    Convenience function to convert a document to PDF.

    Args:
        input_path: Path to input document

    Returns:
        Path to PDF file
    """
    return DocumentConverter.convert_to_pdf(input_path)
