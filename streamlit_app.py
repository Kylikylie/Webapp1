import streamlit as st
import pandas as pd
import io

# 1. Page Configuration
st.set_page_config(page_title="TB Data Integration", layout="wide", page_icon="📊")

# --- Helper Functions ---
@st.cache_data
def load_data(file):
    """Caches data loading to prevent re-reading on every toggle."""
    return pd.read_csv(file)

def clean_columns(df):
    """Standardize column names (lowercase, no spaces) for easier merging."""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

# --- UI Header ---
st.title("🧪 TB & Global Health Data Integration")
st.markdown("""
Streamline your workflow by merging GBD, World Bank, and GDD datasets. 
Upload all three files below to begin processing.
""")

# --- Sidebar / Uploads ---
with st.sidebar:
    st.header("Upload Center")
    gbd_file = st.file_uploader("GBD TB Data", type=["csv"])
    wb_file = st.file_uploader("World Bank Data", type=["csv"])
    gdd_file = st.file_uploader("GDD Data", type=["csv"])
    
    st.divider()
    st.info("Ensure all files contain common keys like 'Country' and 'Year'.")

# --- Main Logic ---
if gbd_file and wb_file and gdd_file:
    # Load and clean data
    with st.spinner("Processing datasets..."):
        gbd = clean_columns(load_data(gbd_file))
        wb = clean_columns(load_data(wb_file))
        gdd = clean_columns(load_data(gdd_file))

    # Organize UI with Tabs
    tab1, tab2, tab3 = st.tabs(["👀 Data Preview", "🔗 Merge & Process", "💾 Export"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("GBD Preview")
            st.dataframe(gbd.head(10), use_container_width=True)
        with col2:
            st.subheader("World Bank Preview")
            st.dataframe(wb.head(10), use_container_width=True)
        with col3:
            st.subheader("GDD Preview")
            st.dataframe(gdd.head(10), use_container_width=True)

    with tab2:
        st.header("Merging Configuration")
        
        # Allow user to pick merge keys (Defaults to 'location' and 'year')
        common_cols = list(set(gbd.columns) & set(wb.columns) & set(gdd.columns))
        
        merge_keys = st.multiselect(
            "Select columns to merge on:",
            options=gbd.columns.tolist(),
            default=[col for col in ['location', 'year', 'country'] if col in common_cols]
        )

        if st.button("Run Merge Operation"):
            try:
                # Sequential merging
                merged_df = gbd.merge(wb, on=merge_keys, how='inner')
                merged_df = merged_df.merge(gdd, on=merge_keys, how='inner')
                
                st.session_state['merged_data'] = merged_df
                st.success(f"Successfully merged! Final shape: {merged_df.shape}")
                st.dataframe(merged_df.head(20))
            except Exception as e:
                st.error(f"Merge failed: {e}")

    with tab3:
        if 'merged_data' in st.session_state:
            st.header("Final Export")
            final_df = st.session_state['merged_data']
            
            # Summary Statistics
            st.write("Quick Summary of Processed Data:")
            st.table(final_df.describe().T.head(5))

            # Download Button
            csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Integrated Dataset (CSV)",
                data=csv,
                file_name="tb_integrated_data.csv",
                mime="text/csv",
            )
        else:
            st.warning("Please run the merge operation in the 'Merge & Process' tab first.")

else:
    st.warning("Please upload all three files in the sidebar to proceed.")
    st.image("https://via.placeholder.com/800x200.png?text=Waiting+for+Data+Uploads...", use_container_width=True)
