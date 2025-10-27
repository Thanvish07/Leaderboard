import streamlit as st
import pandas as pd
import io
import numpy as np

# --- 1. Raw Data Storage ---
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
        col: '{:.4f}' for col in df.select_dtypes(include=np.number).columns
    }
    styler = df.style.format(format_mapping)
    
    # Set table properties for a clean look
    styler = styler.set_properties(**{'text-align': 'center'})
    return styler

# --- 2. Data Cleaning and Preprocessing Functions ---
def clean_standard_df(data_source, task_name):
    """
    Cleans Anomaly and Classification data, handles missing values, and ranks by F1-score.
    Includes aggregation for duplicate models and handles the new 'Type' column.
    """
    if isinstance(data_source, str):
        data_source = io.StringIO(data_source)
    df = pd.read_csv(data_source)
    
    # Check if 'Type' column exists in the data
    if 'Type' in df.columns:
        # Normalize the 'Type' column (e.g., Fine-Tuned -> Fine-tuned)
        df['Type'] = df['Type'].str.capitalize()
        group_cols = ['Type', 'Model']
    else:
        group_cols = ['Model']
        # Add a placeholder 'Type' column for consistency across all dataframes
        df['Type'] = 'N/A' 
        
    # Replace '-' with NaN and convert relevant columns to numeric
    for col in ['Precision', 'Recall', 'F1-score']:
        df[col] = df[col].replace('-', np.nan).astype(float)
        
    # Aggregate duplicate models by taking the mean of their metrics.
    # Grouping by Model and Type ensures different types of the same model are treated separately.
    df = df.groupby(group_cols)[['Precision', 'Recall', 'F1-score']].mean().reset_index()
    
    # Calculate Rank: Higher F1-score is better (ascending=False)
    df['Rank'] = df['F1-score'].rank(method='min', ascending=False).astype('Int64')
    df['Task'] = task_name
    return df.sort_values(by='Rank')

# --- NEW: Function to create a generic column selection block ---
def column_selector(available_cols, default_cols, key_suffix):
    """Creates a Streamlit container with checkboxes for column selection."""
    with st.container(border=True):
        st.markdown("<p style='font-weight:600;'>Select Columns</p>", unsafe_allow_html=True)
        
        # Determine the number of columns for the checkbox layout
        num_cols_for_display = min(len(available_cols), 6) # Cap at 6 columns for neatness
        cols_display = st.columns(num_cols_for_display)
        
        selected_cols = []
        
        for i, col_name in enumerate(available_cols):
            # The 'Icon' and 'Model' columns should generally be locked and default to True
            is_default = col_name in default_cols
            col_index = i % num_cols_for_display
            
            # Use unique key for each checkbox
            checkbox_key = f"col_filter_{col_name.replace('(', '').replace(')', '').replace('-', '_')}_{key_suffix}"
            
            # Lock 'Icon' and 'Model' by making them disabled if they are mandatory
            # However, for maximum user flexibility, we will just make them default 'True'
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
    
    tab_Forecasting, tab_anomaly, tab_classification, tab_imputation = st.tabs([
        "🏆 Forecasting", 
        "🔍 Anomaly Detection", 
        "🏷️ Classification", 
        "🩹 Imputation"
        # , "ℹ️ About"
    ])

    # --- AVAILABLE COLUMN DEFINITIONS ---
    FORECASTING_ORIGINAL_COLS = [
        'Out-of-distribution(OOD)_Commercial',
        'Out-of-distribution(OOD)_Residential',
        'In-Distribution(ID)_Commercial',
        'In-Distribution(ID)_Residential'
    ]
    # Use \n tag to force line breaks in the column names for better display
    FORECASTING_DISPLAY_COLS = [
        'Out-of-distribution(OOD)\nCommercial',
        'Out-of-distribution(OOD)\nResidential',
        'In-Distribution(ID)\nCommercial',
        'In-Distribution(ID)\nResidential'
    ]
    
    FORECASTING_COLUMN_MAP = dict(zip(FORECASTING_ORIGINAL_COLS, FORECASTING_DISPLAY_COLS))

    FORECASTING_AVAILABLE_COLS = ['Icon', 'Model'] + FORECASTING_DISPLAY_COLS
    FORECASTING_DEFAULT_COLS = FORECASTING_AVAILABLE_COLS 
    
    ANOMALY_AVAILABLE_COLS = ['Icon', 'Model', 'F1-score', 'Precision', 'Recall']
    ANOMALY_DEFAULT_COLS = ANOMALY_AVAILABLE_COLS
    
    CLASSIFICATION_AVAILABLE_COLS = ['Icon','Model','F1-score','Precision','Recall']
    CLASSIFICATION_DEFAULT_COLS = CLASSIFICATION_AVAILABLE_COLS

    IMPUTATION_AVAILABLE_COLS = ['Icon','Model','Mask','MAE','MSE']
    IMPUTATION_DEFAULT_COLS = IMPUTATION_AVAILABLE_COLS
    
    
    # ----------------------------------------------------------------------------------
    # --- Forecasting Tab --- 
    # ----------------------------------------------------------------------------------
    with tab_Forecasting:
        st.write("We curated a large-scale energy consumption dataset consisting of 1.26 billion hourly observations collected from 76,217 real-world buildings, encompassing both commercial and residential types across diverse countries and temporal spans.")
        
        # --- Model Type Checkboxes ---
        with st.container(border=True):
            st.markdown("<p style='font-weight:600;'>Model types</p>", unsafe_allow_html=True)
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

        filtered_forecasting = RAW_FORECASTING_DATA[RAW_FORECASTING_DATA.Type.isin(selected_types_check)].copy()
        
        # --- Column Selection ---
        selected_cols_forecast = column_selector(
            FORECASTING_AVAILABLE_COLS, 
            FORECASTING_DEFAULT_COLS, 
            "forecast"
        )
        
        # Rename the columns to their HTML-formatted display names
        df_display_forecast_temp = filtered_forecasting.rename(columns=FORECASTING_COLUMN_MAP)
        
        # Filter by the selected display columns
        df_display_forecast = df_display_forecast_temp[selected_cols_forecast]
        
        sort_col_original = 'Out-of-distribution(OOD)_Commercial'
        sort_col_display = FORECASTING_COLUMN_MAP.get(sort_col_original, sort_col_original)
        
        if sort_col_display in df_display_forecast.columns:
            df_display_forecast = df_display_forecast.sort_values(by=sort_col_display, ascending=True).reset_index(drop=True)
        
        st.dataframe(
            style_dataframe(df_display_forecast, sort_col_display, ascending=True),
            use_container_width=False,
            hide_index=True,
        )
        
        with st.container(border=True):
            st.markdown("<p style='font-weight:600;'></p>", unsafe_allow_html=True)
            st.write("""⚪ Baseline: A simple model used as a benchmark to evaluate the performance of more complex models\n
🔷 ML/DL: These are task-specific models that are trained from scratch on the given dataset.\n
🔴 Zero-shot: These are pretrained models that can generalize to unseen tasks or datasets without additional training, leveraging pretrained knowledge to make predictions directly.\n
🟣 Fine-tuned: Pretrained models adapted to a specific task through additional training on the target dataset.\n
🟢 Pretrained: Models trained on large-scale datasets to capture general patterns, which can later be adapted for specific tasks.""")

    # ----------------------------------------------------------------------------------
    # --- Anomaly Detection Tab --- 
    # ----------------------------------------------------------------------------------
    with tab_anomaly:
        st.write("We use the Large-scale Energy Anomaly Detection (LEAD) dataset which contains electricity meter readings from 200 buildings and anomaly labels. Since the meter readings include missing values, we applied a median imputation technique to handle them. All readings were then normalized using the Standard Scaler. Model performance was evaluated using the F1-score as the primary evaluation metric.")
        
        # --- Model Type Checkboxes ---
        with st.container(border=True):
            st.markdown("<p style='font-weight:600;'>Model types</p>", unsafe_allow_html=True)
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
        selected_cols_anomaly = column_selector(
            ANOMALY_AVAILABLE_COLS, 
            ANOMALY_DEFAULT_COLS, 
            "anomaly"
        )
        
        # Display Anomaly Detection Results
        df_display_anomaly = filtered_anomaly[selected_cols_anomaly]
        
        sort_col = 'F1-score'
        if sort_col in df_display_anomaly.columns:
            df_display_anomaly = df_display_anomaly.sort_values(by=sort_col, ascending=False).reset_index(drop=True)

        st.dataframe(
            style_dataframe(df_display_anomaly, sort_col, ascending=False),
            use_container_width=False,
            hide_index=True
        )

        with st.container(border=True):
            st.markdown("<p style='font-weight:600;'></p>", unsafe_allow_html=True)
            st.write("""🔶 Statistical: A simple model used as a benchmark to evaluate the performance of more complex models\n
🔷 ML/DL: These are task-specific models that are trained from scratch on the given dataset.\n
🔴 Zero-shot: These are pretrained models that can generalize to unseen tasks or datasets without additional training, leveraging pretrained knowledge to make predictions directly.\n
🟣 Fine-tuned: Pretrained models adapted to a specific task through additional training on the target dataset.\n
🟢 Pretrained: Models trained on large-scale datasets to capture general patterns, which can later be adapted for specific tasks.""")

    # ----------------------------------------------------------------------------------
    # --- Classification Tab ---       
    # ----------------------------------------------------------------------------------
    with tab_classification:
        st.write("The ComStock dataset provides 15-minute simulated energy data for U.S. commercial buildings. We selected 1,000 California buildings, using 60-minute appliance-level load data (cooling, fans, heat rejection, heating, refrigerator, washing machine) from 2018. Each appliance has binary labels. Data were split 70% for training and 30% for testing.")
        
        # --- Model Type Checkboxes ---
        with st.container(border=True):
            st.markdown("<p style='font-weight:600;'>Model types</p>", unsafe_allow_html=True)
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

        filtered_classification = RAW_CLASSIFICATION_DATA[RAW_CLASSIFICATION_DATA.Type.isin(selected_types_check)].copy()

        # --- Column Selection ---
        selected_cols_class = column_selector(
            CLASSIFICATION_AVAILABLE_COLS, 
            CLASSIFICATION_DEFAULT_COLS, 
            "classification"
        )
        
        # Display Classification Results
        df_display_class = filtered_classification[selected_cols_class].reset_index(drop=True)
        
        sort_col = 'F1-score'
        if sort_col in df_display_class.columns:
            df_display_class = df_display_class.sort_values(by=sort_col, ascending=False).reset_index(drop=True)

        st.dataframe(
            style_dataframe(df_display_class, sort_col, ascending=False),
            use_container_width=False,
            hide_index=True
        )

        with st.container(border=True):
            st.markdown("<p style='font-weight:600;'></p>", unsafe_allow_html=True)
            st.write("""🔷 ML/DL: These are task-specific models that are trained from scratch on the given dataset.\n
🟣 Fine-tuned: Pretrained models adapted to a specific task through additional training on the target dataset.\n
🟢 Pretrained: Models trained on large-scale datasets to capture general patterns, which can later be adapted for specific tasks.""")

    # ----------------------------------------------------------------------------------
    # --- Imputation Tab --- 
    # ----------------------------------------------------------------------------------
    with tab_imputation:
        st.write("We used meter data from 78 commercial buildings, which form a subset of the BDG2 dataset. Initially, all missing values were replaced with zeros. A Min–Max scaler was applied to each meter reading to normalize the values within the range [0, 1]. The dataset was divided into training (7 months), validation (2 months), and testing (3 months) sets. To evaluate the model’s robustness against incomplete data, masking was applied to simulate missing values at 5%, 10%, 15%, and 20% levels. Model performance was assessed using two key metrics: Mean Absolute Error (MAE) and Mean Squared Error (MSE).")
        
        # --- Model Type Checkboxes ---
        with st.container(border=True):
            st.markdown("<p style='font-weight:600;'>Model types</p>", unsafe_allow_html=True)
            icon_map = {
                "Baseline": "⚪", "ML/DL": "🔷", "Zero-shot": "🔴", 
                "Fine-tuned": "🟣"
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

        filtered_imputation = RAW_IMPUTATION_DATA[RAW_IMPUTATION_DATA.Type.isin(selected_types_check)].copy()

        # --- Column Selection ---
        selected_cols_imput = column_selector(
            IMPUTATION_AVAILABLE_COLS, 
            IMPUTATION_DEFAULT_COLS, 
            "imputation"
        )
        
        # Display Imputation Results
        df_display_imput = filtered_imputation[selected_cols_imput].reset_index(drop=True)
        
        sort_col = 'MSE'
        if sort_col in df_display_imput.columns:
            df_display_imput = df_display_imput.sort_values(by=sort_col, ascending=True).reset_index(drop=True)

        st.dataframe(
            style_dataframe(df_display_imput, sort_col, ascending=True),
            use_container_width=False,
            hide_index=True
        )
        
        with st.container(border=True):
            st.markdown("<p style='font-weight:600;'></p>", unsafe_allow_html=True)
            st.write("""⚪ Baseline: A simple model used as a benchmark to evaluate the performance of more complex models\n
🔷 ML/DL: These are task-specific models that are trained from scratch on the given dataset.\n
🔴 Zero-shot: These are pretrained models that can generalize to unseen tasks or datasets without additional training, leveraging pretrained knowledge to make predictions directly.\n
🟣 Fine-tuned: Pretrained models adapted to a specific task through additional training on the target dataset.""")


#     with tab_about:
#         with st.container(border=True):
#             st.markdown("<p style='font-weight:600;'></p>", unsafe_allow_html=True)
#             st.write("""⚪ Baseline: A simple model used as a benchmark to evaluate the performance of more complex models\n
# 🔷 ML/DL: These are task-specific models that are trained from scratch on the given dataset.\n
# 🔴 Zero-shot: It is a pretrained models that can generalize to unseen tasks or datasets without additional training, leveraging pretrained knowledge to make predictions directly.\n
# 🟣 Fine-tuned: Pretrained models adapted to a specific task through additional training on the target dataset.\n
# 🟢 Pretrained: We curated a large-scale energy consumption dataset consisting of 1.26 billion hourly observations collected from 76,217 real-world buildings, encompassing both commercial and residential types across diverse countries and temporal spans.""")


if __name__ == "__main__":
    main()
