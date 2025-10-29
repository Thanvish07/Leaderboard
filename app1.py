import streamlit as st
import pandas as pd
import numpy as np

# --- Raw Data Storage ---
RAW_ANOMALY_DATA = pd.read_csv("Anomaly_Detection_Results.csv")
RAW_FORECASTING_DATA = pd.read_csv("Forecasting.csv")
RAW_CLASSIFICATION_DATA = pd.read_csv("Classification_Results.csv")
RAW_IMPUTATION_DATA = pd.read_csv("Imputation_Results.csv")


def style_dataframe(df, metric_col, ascending=False):
    """
    Applies minimal styling and ensures number formatting.
    All coloring and highlighting removed as requested.
    """
    # Custom formatters for float values
    format_mapping = {
        col: '{:.4f}' for col in df.select_dtypes(include=np.number).columns if col != 'Mask'
    }
    # Special formatting for the 'Mask' column as it's an integer/percentage
    if 'Mask' in df.columns and df['Mask'].dtype == np.int64:
        format_mapping['Mask'] = '{}'

    styler = df.style.format(format_mapping)
    
    # Set table properties for a clean look
    styler = styler.set_properties(**{'text-align': 'center'})
    return styler

# --- Function to create a generic column selection block ---
def column_selector(available_cols, default_cols, key_suffix):
    """Creates a Streamlit container with checkboxes for column selection."""
            
    with st.container(border=True):
        st.markdown("<p style='font-weight:600;'>Select Columns to display:</p>", unsafe_allow_html=True)
        
        # Determine the number of columns for the checkbox layout
        num_cols_for_display = min(len(available_cols), 6) # Cap at 6 columns for neatness
        # Use st.columns instead of columns() method on type_container
        cols_display = st.columns(num_cols_for_display) 
        available_cols = available_cols
        # 'Icon' and 'Model' are always selected
        selected_cols = ['Icon','Model'] 
        
        for i, col_name in enumerate(available_cols):
            
            # The checkbox state should be derived from default_cols for initial rendering
            is_default = col_name in default_cols
            
            col_index = i % num_cols_for_display
            
            # Create a unique key for each checkbox
            checkbox_key = f"col_filter_{col_name.replace('(', '').replace(')', '').replace('-', '_')}_{key_suffix}"
            
            # Display the checkbox
            if cols_display[col_index].checkbox(
                col_name, 
                value=is_default, 
                key=checkbox_key
            ):
                selected_cols.append(col_name)

        return selected_cols if selected_cols else default_cols # Fallback to default if none are selected

def main():
    """Sets up the Streamlit application structure and loads data."""
    st.set_page_config(
        page_title="Energy Bench Leaderboard",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("Energy Bench Leaderboard")
    
    tab_Forecasting, tab_anomaly, tab_classification, tab_imputation, tab_about = st.tabs([
        "📈 Forecasting", 
        "🚨 Anomaly Detection", 
        "🗂️ Classification", 
        "💊 Imputation",
        "ℹ️ About"
    ])

    # --- AVAILABLE COLUMN DEFINITIONS ---
    FORECASTING_AVAILABLE_COLS = [
        'Out-of-distribution(OOD)_Commercial',
        'Out-of-distribution(OOD)_Residential',
        'In-Distribution(ID)_Commercial',
        'In-Distribution(ID)_Residential'
    ]
    # Default to all columns
    FORECASTING_DEFAULT_COLS = FORECASTING_AVAILABLE_COLS 
    
    ANOMALY_AVAILABLE_COLS = ['F1-score', 'Precision', 'Recall']
    ANOMALY_DEFAULT_COLS = ANOMALY_AVAILABLE_COLS 
    
    CLASSIFICATION_AVAILABLE_COLS = ['F1-score','Precision','Recall']
    CLASSIFICATION_DEFAULT_COLS = CLASSIFICATION_AVAILABLE_COLS

    IMPUTATION_AVAILABLE_COLS = ['Mask','MAE','MSE']
    IMPUTATION_DEFAULT_COLS = IMPUTATION_AVAILABLE_COLS
    
    
    # ----------------------------------------------------------------------------------
    # --- Forecasting Tab --- 
    # ----------------------------------------------------------------------------------
    with tab_Forecasting:
        with st.container(border=True):
            st.markdown("### Load Forecasting")
            st.write(f"""
            This task evaluates models on predicting future energy consumption. 
            We utilize a **Energybench Dataset** of **1.26 billion** hourly observations from **76,217 buildings** (commercial and residential).

            **Evaluation Metric**: **Normalized Root Mean Square Error (NRMSE)**. **Lower** values indicate better performance.
            """)

        # --- Model Type Checkboxes ---
        with st.container(border=True):
            st.markdown("<p style='font-weight:600;'>Model types:</p>", unsafe_allow_html=True)
            icon_map = {
                "Baseline": "⚪", "ML/DL": "🔷", "Zero-shot": "🔴", 
                "Fine-tuned": "🟣", "Pre-trained": "🟢", "Unknown": "❔"
            }
            RAW_FORECASTING_DATA["Type"] = RAW_FORECASTING_DATA["Type"].fillna("Unknown")
            RAW_FORECASTING_DATA.insert(0, "Icon", RAW_FORECASTING_DATA["Type"].map(icon_map).fillna("❔"))
            type_container = st.container()
            unique_types_forecast = RAW_FORECASTING_DATA.Type.unique()
            cols_types = type_container.columns(len(unique_types_forecast), gap="small")
            selected_types_check = []
            
            for i, model_type in enumerate(unique_types_forecast):
                icon = icon_map.get(model_type, "❔")
                model_type_str = str(model_type)
                checkbox_label = f"{icon} {model_type_str}"
                col_index = i % len(cols_types)
                checkbox_key = f"type_filter_forecast_{model_type_str}_{i}"
                if cols_types[col_index].checkbox(checkbox_label, value=True, key=checkbox_key):
                    selected_types_check.append(model_type_str)

        # Filter data based on selected type checkboxes
        filtered_forecasting = RAW_FORECASTING_DATA[RAW_FORECASTING_DATA.Type.isin(selected_types_check)].copy()
        
        # --- Commercial/Residential Filter (Kept, but only for deciding which tables to show) ---
        with st.container(border=True):
            st.markdown("<p style='font-weight:600;'>Building Type:</p>", unsafe_allow_html=True)
            col_comm, col_res = st.columns(2)
            
            show_commercial = col_comm.checkbox("Commercial", value=True, key="forecast_comm_filter")
            show_residential = col_res.checkbox("Residential", value=True, key="forecast_res_filter")

        
        # --- Column Selection (Using all available columns for the metric selector) ---
        selected_cols_forecast_metrics = column_selector(
            FORECASTING_AVAILABLE_COLS, 
            FORECASTING_DEFAULT_COLS, 
            "forecast"
        )
        
        # Separate the selected metric columns into Commercial and Residential sets
        comm_cols_to_show = [
            col for col in selected_cols_forecast_metrics 
            if 'Commercial' in col
        ]
        res_cols_to_show = [
            col for col in selected_cols_forecast_metrics 
            if 'Residential' in col
        ]
        
        # The base columns for both tables
        base_cols = ['Icon', 'Model']
        
        # ----------------------------------------------------------------------------------
        # --- Commercial Results Table ---
        # ----------------------------------------------------------------------------------
        if show_commercial and comm_cols_to_show:
            st.header("🏢 Commercial Buildings")
            
            # Final columns for the commercial table: Icon, Model, and selected Commercial metrics
            selected_cols_comm = base_cols + comm_cols_to_show

            # Prepare commercial dataframe
            df_display_comm = filtered_forecasting[selected_cols_comm]
            
            # Sort logic: Lower Out-of-distribution(OOD)_Commercial is better (ascending=True for error metrics)
            sort_col_comm = 'Out-of-distribution(OOD)_Commercial'
            if sort_col_comm in df_display_comm.columns:
                df_display_comm = df_display_comm.sort_values(by=sort_col_comm, ascending=True).reset_index(drop=True)
            else:
                sort_col_comm = None # Fallback if OOD Commercial isn't selected
            
            st.dataframe(
                style_dataframe(df_display_comm, sort_col_comm, ascending=True if sort_col_comm else False),
                use_container_width=False,
                hide_index=True
            )

        # ----------------------------------------------------------------------------------
        # --- Residential Results Table ---
        # ----------------------------------------------------------------------------------
        if show_residential and res_cols_to_show:
            st.header("🏡 Residential Buildings")
            
            # Final columns for the residential table: Icon, Model, and selected Residential metrics
            selected_cols_res = base_cols + res_cols_to_show
            
            # Prepare residential dataframe
            df_display_res = filtered_forecasting[selected_cols_res]
            
            # Sort logic: Lower Out-of-distribution(OOD)_Residential is better (ascending=True for error metrics)
            sort_col_res = 'Out-of-distribution(OOD)_Residential'
            if sort_col_res in df_display_res.columns:
                df_display_res = df_display_res.sort_values(by=sort_col_res, ascending=True).reset_index(drop=True)
            else:
                sort_col_res = None # Fallback if OOD Residential isn't selected

            st.dataframe(
                style_dataframe(df_display_res, sort_col_res, ascending=True if sort_col_res else False),
                use_container_width=False,
                hide_index=True
            )
        
        if (not show_commercial and not show_residential) or (not comm_cols_to_show and not res_cols_to_show):
             st.warning("Please select at least one building type filter and/or at least one metric column to display results.")

        with st.container(border=True):
            st.markdown("<p style='font-weight:600;'></p>", unsafe_allow_html=True)
            st.write("""⚪ Baseline: A simple model used as a benchmark to evaluate the performance of more complex models\n
🔷 ML/DL: These are task-specific models that are trained from scratch on the given dataset.\n
🔴 Zero-shot: These are pretrained models that can generalize to unseen tasks or datasets without additional training, leveraging pretrained knowledge to make predictions directly.\n
🟣 Fine-tuned: Pretrained models adapted to a specific task through additional training on the target dataset.\n
🟢 Pretrained: Models trained on large-scale datasets to capture general patterns, which can later be adapted for specific tasks.""")


    # ----------------------------------------------------------------------------------
    # --- Anomaly Detection Tab  --- 
    # ----------------------------------------------------------------------------------
    with tab_anomaly:
        with st.container(border=True):
            st.markdown("### Energy Anomaly Detection")
            st.write("""
            This benchmark uses the **Large-scale Energy Anomaly Detection (LEAD) dataset** from 200 buildings with anomaly labels. 
            Data preparation included median imputation for missing values and Standard Scaling.

            **Evaluation Metrics**: **F1-score**, **Precision**, and **Recall**. **Higher** values indicate better performance.
            """)

        # --- Model Type Checkboxes ---
        with st.container(border=True):
            st.markdown("<p style='font-weight:600;'>Model types:</p>", unsafe_allow_html=True)
            icon_map = {
                "Statistical": "🔶", "ML/DL": "🔷", "Zero-shot": "🔴", 
                "Fine-tuned": "🟣", "Pre-trained": "🟢"
            }
            RAW_ANOMALY_DATA.insert(0, "Icon", RAW_ANOMALY_DATA["Type"].map(icon_map))
            type_container = st.container()
            unique_types_anomaly = RAW_ANOMALY_DATA.Type.unique()
            cols_types = type_container.columns(len(unique_types_anomaly), gap="small")
            selected_types_check = []
            
            for i, model_type in enumerate(unique_types_anomaly):
                icon = icon_map.get(model_type, "❔")
                checkbox_label = f"{icon} {model_type}"
                checkbox_key = f"type_filter_anomaly_{model_type}_{i}"
                if cols_types[i % len(unique_types_anomaly)].checkbox(checkbox_label, value=True, key=checkbox_key):
                    selected_types_check.append(model_type)
            
        filtered_anomaly = RAW_ANOMALY_DATA[RAW_ANOMALY_DATA.Type.isin(selected_types_check)].copy()

        # --- Column Selection ---
        selected_cols_anomaly_metrics = column_selector(
            ANOMALY_AVAILABLE_COLS, 
            ANOMALY_DEFAULT_COLS, 
            "anomaly"
        )
        selected_cols_anomaly = ['Icon', 'Model'] + [col for col in selected_cols_anomaly_metrics if col not in ['Icon', 'Model']]
        
        # Display Anomaly Detection Results
        df_display_anomaly = filtered_anomaly[selected_cols_anomaly]
        
        # Sort logic: Higher F1-score is better (ascending=False)
        sort_col = 'F1-score'
        if sort_col in df_display_anomaly.columns:
            df_display_anomaly = df_display_anomaly.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
        else:
            sort_col = None

        st.dataframe(
            style_dataframe(df_display_anomaly, sort_col, ascending=False),
            use_container_width=False, 
            hide_index=True
        )

        with st.container(border=True):
            st.markdown("<p style='font-weight:600;'></p>", unsafe_allow_html=True)
            st.write("""⚪ **Baseline**: A simple model used as a benchmark to evaluate the performance of more complex models\n
🔷 **ML/DL**: These are task-specific models that are trained from scratch on the given dataset.\n
🔴 **Zero-shot**: These are pretrained models that can generalize to unseen tasks or datasets without additional training, leveraging pretrained knowledge to make predictions directly.\n
🟣 **Fine-tuned**: Pretrained models adapted to a specific task through additional training on the target dataset.\n
🟢 **Pretrained**: Models trained on large-scale datasets to capture general patterns, which can later be adapted for specific tasks.""")

    # ----------------------------------------------------------------------------------
    # --- Classification Tab ---       
    # ----------------------------------------------------------------------------------
    with tab_classification:
        with st.container(border=True):
            st.markdown("### Appliance Classification")
            st.write("""
            This task focuses on **appliance ownership prediction** (a binary classification) using 15-minute simulated energy data from **1,000 California commercial buildings** (ComStock dataset). 
            We use appliance-level load data (e.g., cooling, fans, heat rejection, heating, refrigerator, washing machine).

            **Evaluation Metrics**: **F1-score**, **Precision**, and **Recall**. **Higher** values indicate better performance.
            """)

        # --- Model Type Checkboxes ---
        with st.container(border=True):
            st.markdown("<p style='font-weight:600;'>Model types:</p>", unsafe_allow_html=True)
            icon_map = {
                "ML/DL": "🔷", "Fine-tuned": "🟣", "Pre-trained": "🟢", "Unknown": "❔"
            }
            RAW_CLASSIFICATION_DATA["Type"] = RAW_CLASSIFICATION_DATA["Type"].fillna("Unknown")
            RAW_CLASSIFICATION_DATA.insert(0, "Icon", RAW_CLASSIFICATION_DATA["Type"].map(icon_map).fillna("❔"))
            type_container = st.container()
            unique_types_class = RAW_CLASSIFICATION_DATA.Type.unique()
            cols_types = type_container.columns(len(unique_types_class), gap="small")
            selected_types_check = []
            
            for i, model_type in enumerate(unique_types_class):
                icon = icon_map.get(model_type, "❔")
                model_type_str = str(model_type)
                checkbox_label = f"{icon} {model_type_str}"
                col_index = i % len(cols_types)
                checkbox_key = f"class_type_filter_{model_type_str}_{i}"
                if cols_types[col_index].checkbox(checkbox_label, value=True, key=checkbox_key):
                    selected_types_check.append(model_type_str)

        # Filter data based on selected type checkboxes
        filtered_classification = RAW_CLASSIFICATION_DATA[RAW_CLASSIFICATION_DATA.Type.isin(selected_types_check)].copy()

        # --- Column Selection ---
        selected_cols_class_metrics = column_selector(
            CLASSIFICATION_AVAILABLE_COLS, 
            CLASSIFICATION_DEFAULT_COLS, 
            "classification"
        )
        selected_cols_class = ['Icon', 'Model'] + [col for col in selected_cols_class_metrics if col not in ['Icon', 'Model']]
        
        # Display Classification Results
        df_display_class = filtered_classification[selected_cols_class].reset_index(drop=True)
        
        # Sort logic: Higher F1-score is better (ascending=False)
        sort_col = 'F1-score'
        if sort_col in df_display_class.columns:
            df_display_class = df_display_class.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
        else:
            sort_col = None

        st.dataframe(
            style_dataframe(df_display_class, sort_col, ascending=False),
            use_container_width=False,
            hide_index=True
        )

        with st.container(border=True):
            st.markdown("<p style='font-weight:600;'></p>", unsafe_allow_html=True)
            st.write("""🔷 **ML/DL**: These are task-specific models that are trained from scratch on the given dataset.\n
🟣 **Fine-tuned**: Pretrained models adapted to a specific task through additional training on the target dataset.\n
🟢 **Pretrained**: Models trained on large-scale datasets to capture general patterns, which can later be adapted for specific tasks.""")

    # ----------------------------------------------------------------------------------
    # --- Imputation Tab --- 
    # ----------------------------------------------------------------------------------
    with tab_imputation:
        with st.container(border=True):
            st.markdown("### Missing Data Imputation")
            st.write("""
            This task evaluates the ability to accurately fill in missing values in energy meter data from **78 commercial buildings** from BDG-2 Dataset. 
            Missing data (masking) is simulated at various percentage levels (5% to 20%).

            **Evaluation Metrics**: **Mean Absolute Error (MAE)** and **Mean Squared Error (MSE)**. **Lower** values indicate better performance.
            """)
        # --- Model Type Checkboxes (Existing) ---
        with st.container(border=True):
            st.markdown("<p style='font-weight:600;'>Model types:</p>", unsafe_allow_html=True)
            icon_map = {
                "Baseline": "⚪", "ML/DL": "🔷", "Zero-shot": "🔴", 
                "Fine-tuned": "🟣", "Unknown": "❔"
            }
            RAW_IMPUTATION_DATA["Type"] = RAW_IMPUTATION_DATA["Type"].fillna("Unknown")
            RAW_IMPUTATION_DATA.insert(0, "Icon", RAW_IMPUTATION_DATA["Type"].map(icon_map).fillna("❔"))
            type_container = st.container()
            unique_types_imput = RAW_IMPUTATION_DATA.Type.unique()
            cols_types = type_container.columns(len(unique_types_imput), gap="small")
            selected_types_check = []
            
            for i, model_type in enumerate(unique_types_imput):
                icon = icon_map.get(model_type, "❔")
                model_type_str = str(model_type)
                checkbox_label = f"{icon} {model_type_str}"
                col_index = i % len(cols_types)
                checkbox_key = f"imput_type_filter_{model_type_str}_{i}"
                if cols_types[col_index].checkbox(checkbox_label, value=True, key=checkbox_key):
                    selected_types_check.append(model_type_str)

        # Filter data based on selected type checkboxes
        filtered_imputation_by_type = RAW_IMPUTATION_DATA[RAW_IMPUTATION_DATA.Type.isin(selected_types_check)].copy()

        # --- Mask Percentage Checkbox Filter ---
        # st.markdown("---")
        with st.container(border=True):
            st.markdown("<p style='font-weight:600;'>Mask Percentage:</p>", unsafe_allow_html=True)
            
            # Use unique mask values from the data and sort them
            unique_masks = sorted(filtered_imputation_by_type.Mask.unique().tolist())
            cols_masks = st.columns(len(unique_masks))
            selected_masks = []
            
            for i, mask_val in enumerate(unique_masks):
                # Use value=True to have all masks selected by default
                if cols_masks[i].checkbox(f"{mask_val}%", value=True, key=f"imputation_mask_checkbox_{mask_val}"):
                    selected_masks.append(mask_val)


        # Filter data based on selected mask percentages
        if selected_masks:
            filtered_imputation = filtered_imputation_by_type[filtered_imputation_by_type.Mask.isin(selected_masks)].copy()
        else:
            # If no mask is selected, display a warning state
            st.warning("Please select at least one Mask Percentage to display results.")
            st.stop()


        # --- Column Selection (Existing) ---
        selected_cols_imput_metrics = column_selector(
            IMPUTATION_AVAILABLE_COLS, 
            IMPUTATION_DEFAULT_COLS, 
            "imputation"
        )
        selected_cols_imput = ['Icon', 'Model'] + [col for col in selected_cols_imput_metrics if col not in ['Icon', 'Model']]
        
        # Display Imputation Results
        df_display_imput = filtered_imputation[selected_cols_imput].reset_index(drop=True)
        
        # Sort logic: Lower MSE is better (ascending=True for error metrics)
        sort_col = 'MSE'
        if sort_col in df_display_imput.columns:
            df_display_imput = df_display_imput.sort_values(by=sort_col, ascending=True).reset_index(drop=True)
        else:
            sort_col = None

        st.dataframe(
            style_dataframe(df_display_imput, sort_col, ascending=True),
            use_container_width=False,
            hide_index=True
        )
        
        with st.container(border=True):
            st.markdown("<p style='font-weight:600;'></p>", unsafe_allow_html=True)
            st.write("""⚪ **Baseline**: A simple model used as a benchmark to evaluate the performance of more complex models\n
🔷 **ML/DL**: These are task-specific models that are trained from scratch on the given dataset.\n
🔴 **Zero-shot**: These are pretrained models that can generalize to unseen tasks or datasets without additional training, leveraging pretrained knowledge to make predictions directly.\n
🟣 **Fine-tuned**: Pretrained models adapted to a specific task through additional training on the target dataset.""")
            

    # ----------------------------------------------------------------------------------
    # --- About Tab --- 
    # ----------------------------------------------------------------------------------
    with tab_about:
        with st.container(border=True):
            st.markdown("## ⚡️ Energy Bench Leaderboard: Project Overview", unsafe_allow_html=True)
            st.write("""
    **EnergyFM** is a family of **pre-trained models** specifically designed for **energy meter data analytics**. It supports a range of downstream tasks, demonstrating the potential of specialized Foundation Models in the energy sector.
    """)
            st.markdown("---")
            
            st.markdown("### 🧠 Core Foundation Models")
            st.write("""
    EnergyFM is built upon IBM's lightweight Time Series Foundation Model (TSFM) architectures, the **Tiny Time Mixers (TTMs)** and **TSPulse**, which use efficient **MLP-Mixer** designs.

    * **Energy-TTMs**: Optimized for **Short-Term Load Forecasting**.
    * **Energy-TSPulse**: A versatile model supporting **Anomaly Detection, Classification, and Imputation**.
    """)
            st.markdown("---")

            st.markdown("### 🌍 Pre-training Data Scale")
            st.write("""
    The models were pre-trained on a massive, real-world dataset:
    * **Total Observations**: **1.26 billion** hourly meter readings.
    * **Building Coverage**: **76,217** real-world buildings.
    * **Scope**: Encompassing both **commercial and residential** types across diverse countries and temporal spans.
    """)
            st.markdown("---")

            st.markdown("### 🏆 Evaluation Methodology")
            st.write("""
    Performance is benchmarked against traditional models and state-of-the-art TSFMs in two primary settings:
    * **Zero-shot**: Evaluating generalization capability *without* additional training on the target data.
    * **Fine-tuned**: Evaluating performance after adapting the pre-trained model with additional training (transfer learning).
    """)

if __name__ == "__main__":
    main()
