import streamlit as st
import pandas as pd
import io
import numpy as np

# --- 1. Raw Data Storage (Mimicking CSV Files) ---
# These strings serve as the default data source if no files are uploaded.

# UPDATED: Anomaly Data with 'Type'
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
        #"ℹ️ About"
    ])

    
    with tab_Forecasting:
        st.write("We curated a large-scale energy consumption dataset consisting of 1.26 billion hourly observations collected from 76,217 real-world buildings, encompassing both commercial and residential types across diverse countries and temporal spans.")
        with st.container(border=True):  # Streamlit container for the border/box
            st.markdown("<p style='font-weight:600;'>Model types</p>", unsafe_allow_html=True)

            # Define icon mapping for each Type
            icon_map = {
                "Baseline": "⚪",
                "ML/DL": "🔷",
                "Zero-shot": "🔴",
                "Fine-tuned": "🟣",
                "Pre-trained": "🟢"
            }

            # Fill NaN values in 'Type' to avoid float issues
            RAW_FORECASTING_DATA["Type"] = RAW_FORECASTING_DATA["Type"].fillna("Unknown")

            # Add an icon column (first column)
            RAW_FORECASTING_DATA.insert(
                0, "Icon", RAW_FORECASTING_DATA["Type"].map(icon_map).fillna("❔")
            )

            type_container = st.container()
            cols_types = type_container.columns(5, gap="small")

            selected_types_check = []
            i = 0

            # Use unique combination of type + index for key to avoid duplicates
            for icon, model_type in zip(RAW_FORECASTING_DATA.Icon.unique(), RAW_FORECASTING_DATA.Type.unique()):
                model_type_str = str(model_type)
                checkbox_label = f"{icon} {model_type_str}"
                col_index = i % len(cols_types)
                
                # Unique key using model_type + iteration index
                checkbox_key = f"type_filter_{model_type_str}_{i}"

                if cols_types[col_index].checkbox(checkbox_label, value=True, key=checkbox_key):
                    selected_types_check.append(model_type_str)
                i += 1

        # Filter data based on selected checkboxes
        filtered = RAW_FORECASTING_DATA[RAW_FORECASTING_DATA.Type.isin(selected_types_check)]

        # Display Forecasting Results
        display_cols = [
            'Icon', 'Model',
            'Out-of-distribution(OOD)_Commercial',
            'Out-of-distribution(OOD)_Residential',
            'In-Distribution(ID)_Commercial',
            'In-Distribution(ID)_Residential'
        ]
        df_display = filtered[display_cols]

        st.dataframe(
            style_dataframe(df_display, 'Out-of-distribution(OOD)_Commercial', ascending=False),
            use_container_width=False,
            hide_index=True
        )

        with st.container(border=True):
            st.markdown("<p style='font-weight:600;'></p>", unsafe_allow_html=True)
            st.write("""⚪ Baseline: A simple model used as a benchmark to evaluate the performance of more complex models\n
🔷 ML/DL: These are task-specific models that are trained from scratch on the given dataset.\n
🔴 Zero-shot: These are pretrained models that can generalize to unseen tasks or datasets without additional training, leveraging pretrained knowledge to make predictions directly.\n
🟣 Fine-tuned: Pretrained models adapted to a specific task through additional training on the target dataset.\n
🟢 Pretrained: Models trained on large-scale datasets to capture general patterns, which can later be adapted for specific tasks.""")


    with tab_anomaly:
        st.write("We use the Large-scale Energy Anomaly Detection (LEAD) dataset which contains electricity meter readings from 200 buildings and anomaly labels. Since the meter readings include missing values, we applied a median imputation technique to handle them. All readings were then normalized using the Standard Scaler. Model performance was evaluated using the F1-score as the primary evaluation metric.")
                # --- Model Type Checkboxes (Styled as Buttons in a Box) ---
        with st.container(border=True): # Use Streamlit container for the border/box
            st.markdown("<p style='font-weight:600;'>Model types</p>", unsafe_allow_html=True)

                 # Define icon mapping for each Type
            icon_map = {
                "Statistical": "🔶",
                "ML/DL": "🔷",
                "Zero-shot": "🔴",
                "Fine-tuned": "🟣",
                "Pre-trained": "🟢"
            }

            # Add an icon column (first column)
            RAW_ANOMALY_DATA.insert(0, "Icon", RAW_ANOMALY_DATA["Type"].map(icon_map))


            type_container = st.container()
            cols_types = type_container.columns(5, gap="small")




            selected_types_check = []
            i = 0
            for icon, model_type in zip(RAW_ANOMALY_DATA.Icon.unique(), RAW_ANOMALY_DATA.Type.unique()):
                print(model_type)
                # The visual feedback (col''or change) is now handled solely by CSS on the label
                if cols_types[i].checkbox( str(icon) + model_type, value=True, key=f"type_filter_{model_type}"):
                    selected_types_check.append(model_type)
                i += 1


            filtered = RAW_ANOMALY_DATA[RAW_ANOMALY_DATA.Type.isin(selected_types_check)]


        
        # Display Anomaly Detection Results
        # Ensure 'Type' is included in the display columns
        display_cols = ['Icon', 'Model', 'F1-score', 'Precision', 'Recall']
            
        df_display = filtered[display_cols]
        df_display = df_display.sort_values(by='F1-score', ascending=False).reset_index(drop=True)

   

        st.dataframe(
            style_dataframe(df_display, 'F1-score', ascending=False),
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


        # --- Classification Tab ---    
    with tab_classification:
        st.write("The ComStock dataset provides 15-minute simulated energy data for U.S. commercial buildings. We selected 1,000 California buildings, using 60-minute appliance-level load data (cooling, fans, heat rejection, heating, refrigerator, washing machine) from 2018. Each appliance has binary labels. Data were split 70% for training and 30% for testing.")
        with st.container(border=True):  # Streamlit container for the border/box
            st.markdown("<p style='font-weight:600;'>Model types</p>", unsafe_allow_html=True)

            # Define icon mapping for each Type
            icon_map = {
                "ML/DL": "🔷",
                "Fine-tuned": "🟣",
                "Pre-trained": "🟢"
            }

            # Fill NaN values in 'Type' to avoid float issues
            RAW_CLASSIFICATION_DATA["Type"] = RAW_CLASSIFICATION_DATA["Type"].fillna("Unknown")

            # Add an icon column (first column)
            RAW_CLASSIFICATION_DATA.insert(
                0, "Icon", RAW_CLASSIFICATION_DATA["Type"].map(icon_map).fillna("❔")
            )

            # Container for checkboxes
            type_container = st.container()
            cols_types = type_container.columns(3, gap="small")

            selected_types_check = []
            i = 0

            # Use unique combination of type + index for key to avoid duplicates
            for icon, model_type in zip(RAW_CLASSIFICATION_DATA.Icon.unique(), RAW_CLASSIFICATION_DATA.Type.unique()):
                model_type_str = str(model_type)
                checkbox_label = f"{icon} {model_type_str}"
                col_index = i % len(cols_types)

                # Unique key using model_type + iteration index
                checkbox_key = f"class_type_filter_{model_type_str}_{i}"

                if cols_types[col_index].checkbox(checkbox_label, value=True, key=checkbox_key):
                    selected_types_check.append(model_type_str)
                i += 1

        # Filter data based on selected checkboxes
        filtered = RAW_CLASSIFICATION_DATA[RAW_CLASSIFICATION_DATA.Type.isin(selected_types_check)]

        # Display Classification Results
        display_cols = ['Icon','Model','F1-score','Precision','Recall']
        df_display = filtered[display_cols].reset_index(drop=True)

        st.dataframe(
            style_dataframe(df_display, 'F1-score', ascending=False),
            use_container_width=False,
            hide_index=True
        )

        with st.container(border=True):
            st.markdown("<p style='font-weight:600;'></p>", unsafe_allow_html=True)
            st.write("""🔷 ML/DL: These are task-specific models that are trained from scratch on the given dataset.\n
🟣 Fine-tuned: Pretrained models adapted to a specific task through additional training on the target dataset.\n
🟢 Pretrained: Models trained on large-scale datasets to capture general patterns, which can later be adapted for specific tasks.""")


    with tab_imputation:
        st.write("We used meter data from 78 commercial buildings, which form a subset of the BDG2 dataset. Initially, all missing values were replaced with zeros. A Min–Max scaler was applied to each meter reading to normalize the values within the range [0, 1]. The dataset was divided into training (7 months), validation (2 months), and testing (3 months) sets. To evaluate the model’s robustness against incomplete data, masking was applied to simulate missing values at 5%, 10%, 15%, and 20% levels. Model performance was assessed using two key metrics: Mean Absolute Error (MAE) and Mean Squared Error (MSE).")
        with st.container(border=True):
            st.markdown("<p style='font-weight:600;'>Model types</p>", unsafe_allow_html=True)

            # Define icon mapping for each Type
            icon_map = {
                "Baseline": "⚪",
                "ML/DL": "🔷",
                "Zero-shot": "🔴",
                "Fine-tuned": "🟣",
            }

            # Fill NaN values in 'Type' to avoid float issues
            RAW_IMPUTATION_DATA["Type"] = RAW_IMPUTATION_DATA["Type"].fillna("Unknown")

            # Add an icon column (first column)
            RAW_IMPUTATION_DATA.insert(
                0, "Icon", RAW_IMPUTATION_DATA["Type"].map(icon_map).fillna("❔")
            )

            # Container for checkboxes
            type_container = st.container()
            cols_types = type_container.columns(4, gap="small")

            selected_types_check = []
            i = 0

            # Use unique combination of type + index for key to avoid duplicates
            for icon, model_type in zip(RAW_IMPUTATION_DATA.Icon.unique(), RAW_IMPUTATION_DATA.Type.unique()):
                model_type_str = str(model_type)
                checkbox_label = f"{icon} {model_type_str}"
                col_index = i % len(cols_types)

                # Unique key using model_type + iteration index
                checkbox_key = f"class_type_filter_{model_type_str}_{i}"

                if cols_types[col_index].checkbox(checkbox_label, value=True, key=checkbox_key):
                    selected_types_check.append(model_type_str)
                i += 1

        # Filter data based on selected checkboxes
        filtered = RAW_IMPUTATION_DATA[RAW_IMPUTATION_DATA.Type.isin(selected_types_check)]

        # Display Classification Results
        display_cols = ['Icon','Model','Mask','MAE','MSE']
        df_display = filtered[display_cols].reset_index(drop=True)

        st.dataframe(
            style_dataframe(df_display, 'Mask', ascending=False),
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

