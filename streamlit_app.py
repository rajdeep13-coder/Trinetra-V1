import logging
from typing import Dict

from backend.data_preparation_gemini import (
    generate_data_prep_code,
    execute_data_prep_code,
    generate_and_execute_data_prep
)
from backend.data_quality_checks import run_data_quality_checks
from backend.pdf_processing import process_pdf, check_pdf_quality
from backend.image_processing import process_image, check_image_quality
import pandas as pd
import streamlit as st


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clear_all_data():
    """Clear all session state variables and cache"""
    
    for key in list(st.session_state.keys()):
        del st.session_state[key]
   
    st.cache_data.clear()
    st.cache_resource.clear()


def get_dataset_metadata(df: pd.DataFrame) -> dict:
    """Generate metadata for a dataframe including summary stats and column info"""
    return {
        "summary_stats": df.describe(include="all").to_dict(),
        "columns": {
            col: {
                "dtype": str(df[col].dtype),
                "unique_count": df[col].nunique(),
                "sample_values": df[col].head().tolist(),
            }
            for col in df.columns
        },
        "shape": df.shape,
    }


@st.cache_data(show_spinner=False)
def cached_run_data_quality_checks(df: pd.DataFrame, metadata: Dict) -> Dict:
    """
    Cached wrapper for running data quality checks
    """
    return run_data_quality_checks(df, metadata["summary_stats"])


def display_data_quality_results(filename: str, df: pd.DataFrame):
    """Display data quality results with spinner"""
    st.markdown(f"##### Data Quality Analysis: {filename}")

   
    with st.spinner("Running data quality checks..."):
        data_quality_results = cached_run_data_quality_checks(
            df, st.session_state["datasets_metadata"][filename]
        )

    logger.info("Proceeding to display results")

   
    for i, (check, result) in enumerate(data_quality_results.items()):
        if result["issue_detected"]:
            display_name = check.replace("_", " ").title()
            is_first_issue = i == 0
            with st.expander(f"⚠️ {display_name}", expanded=is_first_issue):
                if "results_df" in result and not result["results_df"].empty:
                    st.dataframe(
                        result["results_df"], hide_index=True, use_container_width=True
                    )
                else:
                    st.markdown(result["recommendation"])


def main():
    st.set_page_config(
        page_title="AI Data Prep Assistant",
        page_icon="🤖",
        layout="wide",
    )

   
    st.markdown("# 🤖 AI Data Prep Assistant")

   
    if "datasets" not in st.session_state:
        st.session_state["datasets"] = {}
    if "datasets_metadata" not in st.session_state:
        st.session_state["datasets_metadata"] = {}

    
    st.sidebar.title("Upload Files")
    uploaded_csvs = st.sidebar.file_uploader(
        "Choose CSV files", accept_multiple_files=True, type=["csv"]
    )
    
    uploaded_pdfs = st.sidebar.file_uploader(
        "Choose PDF files", accept_multiple_files=True, type=["pdf"]
    )
    
    uploaded_images = st.sidebar.file_uploader(
        "Choose Image files", accept_multiple_files=True, type=["jpg", "jpeg", "png"]
    )


    if st.sidebar.button("Clear All Data"):
        clear_all_data()
        st.rerun()

    if uploaded_csvs:
        for uploaded_file in uploaded_csvs:
            df = pd.read_csv(uploaded_file)
            st.session_state["datasets"][uploaded_file.name] = df
          
            st.session_state["datasets_metadata"][
                uploaded_file.name
            ] = get_dataset_metadata(df)
    
    if uploaded_pdfs:
        for uploaded_pdf in uploaded_pdfs:
          
            pdf_bytes = uploaded_pdf.read()
            cleaned_pdf, report = process_pdf(pdf_bytes)
            
          
            quality_issues = check_pdf_quality(pdf_bytes)
            
           
            st.subheader(f"PDF Processing Results: {uploaded_pdf.name}")
            
           
            if quality_issues["issue_detected"]:
                st.warning("Quality issues detected in PDF")
                st.dataframe(quality_issues["results_df"])
                st.info(quality_issues["recommendation"])
            
        
            st.write("Processing Report:")
            st.write(f"- Total Pages: {report['total_pages']}")
            st.write(f"- Repaired Pages: {report['repaired_pages']}")
            st.write(f"- OCR Applied: {report['ocr_pages']} pages")
            st.write(f"- Removed Blank Pages: {report['removed_pages']}")
            if report['ocr_pages'] > 0:
                st.write(f"- Average OCR Confidence: {report['avg_ocr_confidence']}%")
            
           
            if report['errors']:
                st.error("\n".join(report['errors']))
            
            
            st.download_button(
                label=f"Download Cleaned {uploaded_pdf.name}",
                data=cleaned_pdf,
                file_name=f"cleaned_{uploaded_pdf.name}",
                mime="application/pdf"
            )
    
    if uploaded_images:
        for uploaded_image in uploaded_images:
           
            image_bytes = uploaded_image.read()
            
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image_bytes)
            
            
            st.sidebar.subheader("Image Enhancement Controls")
            brightness = st.sidebar.slider("Brightness", 0.0, 2.0, 1.0, 0.1)
            contrast = st.sidebar.slider("Contrast", 0.0, 2.0, 1.0, 0.1)
            sharpness = st.sidebar.slider("Sharpness", 0.0, 2.0, 1.0, 0.1)
            saturation = st.sidebar.slider("Saturation", 0.0, 2.0, 1.0, 0.1)
            
            enhancement_params = {
                "brightness": brightness,
                "contrast": contrast,
                "sharpness": sharpness,
                "saturation": saturation
            }
            
            cleaned_image, report = process_image(image_bytes, uploaded_image.name, enhancement_params)
            
    
            quality_issues = check_image_quality(image_bytes)
            
            with col2:
                st.subheader("Processed Image")
                st.image(cleaned_image)
            
          
            st.subheader(f"Image Processing Results: {uploaded_image.name}")
            
         
            if quality_issues["issue_detected"]:
                st.warning("Quality issues detected in image")
                st.dataframe(quality_issues["results_df"])
                st.info(quality_issues["recommendation"])
            
           
            st.write("Processing Report:")
            st.write(f"- Original Format: {report['original_format']}")
            st.write(f"- Original Size: {report['original_size']}")
            st.write(f"- Final Size: {report['final_size']}")
            st.write("Operations performed:")
            for operation in report["operations"]:
                st.write(f"- {operation}")
            
           
            if report['errors']:
                st.error("\n".join(report['errors']))
            
           
            st.download_button(
                label=f"Download Cleaned {uploaded_image.name}",
                data=cleaned_image,
                file_name=f"cleaned_{uploaded_image.name}",
                mime=f"image/{report['original_format'].lower()}"
            )
           
            st.markdown("---")

   
    st.title("AI Data Preparation Assistant")

   
    tab1, tab2 = st.tabs(["Data Quality Explorer", "AI Data Prep"])

    with tab1:
       
        if not st.session_state.get("datasets"):
            st.info("Please upload CSV files in the sidebar.")
        else:
            for idx, (filename, df) in enumerate(st.session_state["datasets"].items()):
              
                st.subheader(f"Dataset: {filename}")

               
                with st.expander(f"📊 Data Preview", expanded=True):
                    st.dataframe(df.head(1000), height=550)

                
                with st.expander("Summary Statistics", expanded=False):
                    st.dataframe(df.describe(include="all"))

              
                display_data_quality_results(filename, df)

                
                st.markdown("---")

    with tab2:
        if not st.session_state.get("datasets"):
            st.info("Please upload CSV files in the sidebar.")
        else:
            st.subheader("Select Data Quality Issues to Resolve")
            selected_issues = {}
            for filename, df in st.session_state["datasets"].items():
                data_quality_results = cached_run_data_quality_checks(
                    df, st.session_state["datasets_metadata"][filename]
                )
                selected_issues[filename] = {}
                for check, result in data_quality_results.items():
                    if result["issue_detected"]:
                        if st.checkbox(f"{filename} - {check}"):
                            selected_issues[filename][check] = result

            st.subheader("Additional Data Preparation Steps")
            user_instructions = st.text_area(
                "Enter additional data preparation instructions"
            )

            if st.button("Generate and Execute Data Prep"):
                with st.spinner("Generating and executing data preparation code..."):
                    result = generate_and_execute_data_prep(
                        st.session_state["datasets_metadata"],
                        selected_issues,
                        user_instructions,
                    )

                    if isinstance(result, list):
                        st.session_state["processed_dataframes"] = result

                        
                        results_container = st.container()

                        with results_container:
                           
                            st.subheader("Generated Data Preparation Code")
                            if "generated_code" in st.session_state:
                                st.code(
                                    st.session_state["generated_code"]["code"],
                                    language="python",
                                )

                            # Show success message
                            st.success("✅ Data preparation completed successfully!")

                            
                            st.subheader("Processed Datasets")
                            for i, df in enumerate(
                                st.session_state["processed_dataframes"]
                            ):
                                with st.expander(
                                    f"Processed Dataset {i+1}", expanded=True
                                ):
                                    
                                    st.dataframe(df.head(1000), height=600)

                                    
                                    csv = df.to_csv(index=False).encode("utf-8")
                                    st.download_button(
                                        label=f"Download Dataset {i+1} as CSV",
                                        data=csv,
                                        file_name=f"processed_dataset_{i+1}.csv",
                                        mime="text/csv",
                                    )
                    else:
                        failed_code, error_msg = result
                        st.error("❌ Data preparation failed after maximum attempts!")

                     
                        st.subheader("Failed Code")
                        st.code(failed_code, language="python")

                        
                        st.error(f"Error Details:\n{error_msg}")

           
            elif st.session_state.get("processed_dataframes"):
                st.subheader("Processed Datasets")
                for i, df in enumerate(st.session_state["processed_dataframes"]):
                    with st.expander(f"Processed Dataset {i+1}", expanded=True):
                        st.dataframe(df.head(1000))
                        st.write("Shape:", df.shape)

                        
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label=f"Download Dataset {i+1} as CSV",
                            data=csv,
                            file_name=f"processed_dataset_{i+1}.csv",
                            mime="text/csv",
                        )


if __name__ == "__main__":
    main()
