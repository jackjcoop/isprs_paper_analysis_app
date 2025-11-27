"""
ISPRS Paper Formatting Analysis - Streamlit Web Application

A simple web interface for analyzing ISPRS paper compliance.
"""

import streamlit as st
import tempfile
import sys
import io
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

from main import PDFComplianceAnalyzer


# Page configuration
st.set_page_config(
    page_title="ISPRS Paper Analyzer",
    page_icon="📄",
    layout="centered"
)

# Custom CSS for cleaner appearance
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
    }
    .upload-text {
        text-align: center;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)


# --- Authentication ---
def check_password():
    """Returns True if user has correct credentials."""

    def login_form():
        """Display login form."""
        st.title("🔐 ISPRS Paper Analyzer")
        st.markdown("Please log in to access the application.")
        st.divider()

        with st.form("login_form"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

            if submitted:
                if (st.session_state["username"] == st.secrets["credentials"]["username"] and
                    st.session_state["password"] == st.secrets["credentials"]["password"]):
                    st.session_state["authenticated"] = True
                    # Clear credentials from session state
                    del st.session_state["password"]
                    del st.session_state["username"]
                    st.rerun()
                else:
                    st.session_state["authenticated"] = False

    # Check if already authenticated
    if st.session_state.get("authenticated"):
        return True

    # Show login form
    login_form()

    # Show error if login failed
    if st.session_state.get("authenticated") == False:
        st.error("❌ Invalid username or password")

    return False


# Check authentication before showing main app
if not check_password():
    st.stop()


# --- Main Application (only shown after login) ---

# Header
st.title("📄 ISPRS Paper Formatting Analysis")
st.markdown(
    "Upload your paper to check compliance with ISPRS full paper formatting guidelines."
)
st.markdown(
    "[📖 View full ISPRS guidelines](https://www.isprs.org/documents/orangebook/app5.aspx)"
)

st.divider()

# File upload
uploaded_file = st.file_uploader(
    "Upload your paper",
    type=["pdf"],
    help="Drag and drop a PDF file or click to browse"
)

# Options
check_anon = st.checkbox(
    "Check anonymization (for review submissions)",
    help="Verify the document is properly anonymized - no author names or identifying information"
)

st.divider()

# Analyze button
if uploaded_file is not None:
    st.info(f"📎 **File:** {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

    if st.button("🔍 Analyze Paper", type="primary", use_container_width=True):

        # Create progress placeholder
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        try:
            with st.spinner("Analyzing document... This may take a minute."):
                # Save uploaded file to temp location
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_pdf = Path(tmp_dir) / uploaded_file.name
                    tmp_pdf.write_bytes(uploaded_file.getvalue())

                    # Suppress console output from analyzer
                    stdout_capture = io.StringIO()
                    stderr_capture = io.StringIO()

                    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                        # Run analyzer
                        analyzer = PDFComplianceAnalyzer(
                            check_anonymization=check_anon,
                            generate_report=True
                        )
                        result = analyzer.analyze(str(tmp_pdf))

                    # Check for generated report
                    report_path = tmp_pdf.parent / f"{tmp_pdf.stem}_report.pdf"

                    if report_path.exists():
                        report_bytes = report_path.read_bytes()

                        # Show success message
                        st.success("✅ Analysis complete!")

                        # Show summary stats
                        if result and 'validation' in result:
                            validation = result['validation']
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Checks", validation.get('total_checks', 0))
                            with col2:
                                st.metric("Warnings", validation.get('warnings', 0))
                            with col3:
                                if validation.get('overall_pass', False):
                                    st.metric("Status", "✓ Passed")
                                else:
                                    st.metric("Status", "⚠ Issues Found")

                        st.divider()

                        # Download button
                        st.download_button(
                            label="📥 Download Analysis Report (PDF)",
                            data=report_bytes,
                            file_name=f"{uploaded_file.name.replace('.pdf', '')}_report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

                        st.caption(
                            "The report includes a summary of validation results "
                            "followed by your annotated document with highlighted issues."
                        )
                    else:
                        st.error("❌ Failed to generate report. Please try again.")

        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.caption("Please ensure your PDF is valid and try again.")

else:
    # Show placeholder when no file uploaded
    st.markdown(
        """
        <div style="text-align: center; padding: 40px; color: #888;">
            <p>👆 Upload a PDF file to get started</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer
st.divider()
st.caption(
    "ISPRS Paper Formatting Analyzer | "
    "Validates compliance with ISPRS full paper submission requirements"
)
