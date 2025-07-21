import os
os.environ["PYTHONWARNINGS"] = "ignore"  
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score
import datetime,time
import io
import pickle
from Modules.pre_process import pre_process
from Modules.ml_model import ml_model
from Modules.pipeline import pipeline

if not hasattr(st.session_state, 'initialized'):
    # Core data variables
    st.session_state.initialized = True
    st.session_state.data = None
    st.session_state.preprocessed = None
    st.session_state.target = None
    st.session_state.task_type = None
    st.session_state.num_mean = None
    st.session_state.num_median = None
    st.session_state.cat_missing = None
    
    # Model training variables
    st.session_state.model = None
    st.session_state.score = None
    st.session_state.model_scores = None
    st.session_state.processor = None
    st.session_state.trained_pipeline = None
    st.session_state.encoder = None  

# App Config
st.set_page_config(
    page_title="ML Pipeline Pro",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visibility
st.markdown("""
<style>
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        color: #333333;
    }
    .stRadio div {
        flex-direction: column;
    }
    .stRadio div[role="radiogroup"] > label {
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 5px;
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #dee2e6;
        transition: all 0.3s ease;
    }
    .stRadio div[role="radiogroup"] > label:hover {
        background-color: #e9ecef;
        color: #333333;
        border-color: #1DA1F2;
    }
    .stRadio div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) {
        background-color: #1DA1F2;
        color: #ffffff;
        border-color: #1DA1F2;
    }
    .stRadio div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked):hover {
        background-color: #0d8bd9;
        color: #ffffff;
    }
    .fileUploader {
        border: 2px dashed #1DA1F2;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        text-align: center;
    }
    .metric-card b {
        color: #1DA1F2;
        font-size: 14px;
    }
    .metric-card div {
        font-size: 24px;
        font-weight: bold;
        color: #333333;
        margin-top: 5px;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .stRadio div[role="radiogroup"] > label {
            background-color: #2b2b2b;
            color: #ffffff;
            border-color: #4a4a4a;
        }
        .stRadio div[role="radiogroup"] > label:hover {
            background-color: #3a3a3a;
            color: #ffffff;
        }
        .metric-card {
            background-color: #2b2b2b;
            color: #ffffff;
            border-color: #4a4a4a;
        }
        .metric-card b {
            color: #1DA1F2;
        }
        .metric-card div {
            color: #ffffff;
        }
    }
    
    /* Streamlit specific overrides */
    .stRadio > div > div > div > label > div {
        color: inherit !important;
    }
    
    /* Quick guide styling */
    .quick-guide {
        background-color: #f8f9fa;
        color: #333333;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
    
    @media (prefers-color-scheme: dark) {
        .quick-guide {
            background-color: #2b2b2b;
            color: #ffffff;
            border-color: #4a4a4a;
        }
    }
    
    .download-button {
        background-color: #28a745;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
        margin: 5px;
    }
    
    .download-button:hover {
        background-color: #218838;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to create download buffer for different file formats
def create_download_buffer(data, file_format):
    """Create downloadable buffer for different file formats"""
    buffer = io.BytesIO()
    
    if file_format == 'csv':
        csv_data = data.to_csv(index=False)
        buffer.write(csv_data.encode())
        mime_type = 'text/csv'
    elif file_format == 'json':
        json_data = data.to_json(orient='records', indent=2)
        buffer.write(json_data.encode())
        mime_type = 'application/json'
    elif file_format == 'xml':
        xml_data = data.to_xml(index=False)
        buffer.write(xml_data.encode())
        mime_type = 'application/xml'
    
    buffer.seek(0)
    return buffer, mime_type

# Sidebar Navigation
with st.sidebar:
    st.markdown("## Navigation")
    
    # Custom radio buttons for navigation
    nav_options = ["📊 Data Processing", "🤖 Model Training", "💾 Download Pipeline", "🔮 Make Predictions"]
    selected_nav = st.radio(
        "",
        nav_options,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
    <div class="quick-guide">
        <h4>Quick Guide</h4>
        <ol>
            <li>Upload & preprocess data</li>
            <li>Train models</li>
            <li>Download pipeline</li>
            <li>Make predictions</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# ------------------ 1. DATA PROCESSING ------------------
if selected_nav == "📊 Data Processing":
    st.header("🔍 Data Processing")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"], 
                               help="Supports CSV and Excel files")
        
        if file:
            try:
                if file.name.endswith('.csv'):
                    data = pd.read_csv(file)
                else:
                    data = pd.read_excel(file)
                
                st.session_state.data = data.copy()
                
                with st.expander("🔎 Data Preview", expanded=True):
                    st.dataframe(data.head())
                    
                    # Data stats cards
                    cols = st.columns(4)
                    with cols[0]:
                        st.markdown('<div class="metric-card"><b>📊 Rows</b><div>{:,}</div></div>'.format(data.shape[0]), unsafe_allow_html=True)
                    with cols[1]:
                        st.markdown('<div class="metric-card"><b>📈 Columns</b><div>{:,}</div></div>'.format(data.shape[1]), unsafe_allow_html=True)
                    with cols[2]:
                        st.markdown('<div class="metric-card"><b>⚠️ Missing Values</b><div>{:,}</div></div>'.format(data.isna().sum().sum()), unsafe_allow_html=True)
                    with cols[3]:
                        st.markdown('<div class="metric-card"><b>🔍 Duplicates</b><div>{:,}</div></div>'.format(data.duplicated().sum()), unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    with col2:
        if st.session_state.data is not None:
            st.subheader("Data Visualization")
            viz_type = st.selectbox("Chart Type", 
                                   ["Histogram", "Box Plot", "Scatter Plot", "Correlation"])
            col = st.selectbox("Select Column", st.session_state.data.columns)
            
            if viz_type == "Histogram":
                fig = px.histogram(st.session_state.data, x=col)
                st.plotly_chart(fig, use_container_width=True)
            elif viz_type == "Box Plot":
                fig = px.box(st.session_state.data, y=col)
                st.plotly_chart(fig, use_container_width=True)
            elif viz_type == "Scatter Plot":
                col2 = st.selectbox("Select Y-axis", st.session_state.data.columns)
                fig = px.scatter(st.session_state.data, x=col, y=col2)
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.imshow(st.session_state.data.corr(numeric_only=True))
                st.plotly_chart(fig, use_container_width=True)
    
    if st.session_state.data is not None:
        st.subheader("⚙️ Preprocessing Settings")
        
        with st.form("preprocess_form"):
            target = st.selectbox("Target Column", st.session_state.data.columns)
            task_type = st.radio("Task Type", ["Classification", "Regression"], horizontal=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Outliers**")
                outlier_option = st.radio(
                    "Select one option:",
                    options=["None", "Capping", "Trimming"],
                    horizontal=True,
                    index=0  # Default to "None"
                )
                cap_outliers = (outlier_option == "Capping")
                trim_outliers = (outlier_option == "Trimming")
            
                
            with col2:
                st.markdown("**File Format**")
                file_format = st.radio(
                    "Select output format:",
                    options=["CSV", "JSON", "XML"],
                    horizontal=True,
                    index=0
                )
                format_mapping = {"CSV": "csv", "JSON": "json", "XML": "xml"}
                selected_format = format_mapping[file_format]
                
            
            if st.form_submit_button("🚀 Preprocess Data"):
                with st.spinner("Processing data..."):
                    try:
                        p = pre_process(
                            st.session_state.data.copy(), 
                            target, 
                            cap=cap_outliers, 
                            trim=trim_outliers,
                            format=selected_format
                        )
                        
                        p.outliers()
                        p.missingvalues()
                        p.transform()
                        p.scaling_encoding()
                        st.session_state.preprocessed = p.data
                        st.session_state.target = target
                        st.session_state.task_type = 'C' if task_type == "Classification" else 'R'
                        st.session_state.num_mean = p.num_missing_mean
                        st.session_state.num_median = p.num_missing_median
                        st.session_state.cat_missing = p.cat_missing
                        st.session_state.processor = p  # Store the processor object for download later
                        
                        st.success("✅ Preprocessing complete!")
                        
                        with st.expander("View Processed Data"):
                            st.dataframe(p.data.head())
                            
                            # Show preprocessing details
                            st.markdown("**Preprocessing Summary**")
                            if p.num_missing_mean:
                                st.write(f"Numerical columns filled with mean: {', '.join(p.num_missing_mean)}")
                            if p.num_missing_median:
                                st.write(f"Numerical columns filled with median: {', '.join(p.num_missing_median)}")
                            if p.cat_missing:
                                st.write(f"Categorical columns filled with mode: {', '.join(p.cat_missing)}")
                            
                    except Exception as e:
                        st.error(f"❌ Error during preprocessing: {str(e)}")
        
        # Download processed data section
        if 'processor' in st.session_state and st.session_state.processor is not None:
            st.subheader("💾 Download Processed Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                file_name = st.text_input("File name (without extension)", "processed_data")
                
            with col2:
                st.markdown("**Download Format**")
                download_format = st.selectbox(
                    "Select format:",
                    options=["CSV", "JSON", "XML"],
                    index=0
                )
            
            if st.button("📥 Download Processed Data"):
                try:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    download_format_lower = download_format.lower()
                    full_filename = f"{file_name}_{timestamp}.{download_format_lower}"
                    
                    # Create download buffer
                    buffer, mime_type = create_download_buffer(st.session_state.preprocessed, download_format_lower)
                    
                    st.download_button(
                        label=f"📥 Download {download_format} File",
                        data=buffer,
                        file_name=full_filename,
                        mime=mime_type,
                        use_container_width=True
                    )
                    
                    st.success(f"✅ Processed data ready for download as: {full_filename}")
                    
                except Exception as e:
                    st.error(f"❌ Failed to prepare download: {e}")

# ------------------ 2. MODEL TRAINING ------------------
elif selected_nav == "🤖 Model Training":
    st.header("🤖 Model Training")
    
    if 'preprocessed' not in st.session_state or st.session_state.preprocessed is None:
        st.warning("Please preprocess data first!")
    else:
        with st.form("model_train_form"):
            if st.form_submit_button("🏋️ Train Models"):
                with st.spinner("Training in progress..."):
                    try:
                        model_selector = ml_model(
                            st.session_state.preprocessed.copy(),
                            st.session_state.task_type,
                            st.session_state.target,
                        )
                        
                        model_selector.train_test()
                        model_selector.model_selection()
                        
                        st.session_state.model = model_selector.best_model
                        st.session_state.score = model_selector.best_score
                        st.session_state.model_scores = model_selector.scores
                        
                        st.success(f"✅ Best Model: {type(model_selector.best_model).__name__} (Score: {model_selector.best_score:.4f})")
                        
                        # Model Comparison
                        st.subheader("Model Comparison")
                        fig = px.bar(
                            pd.DataFrame(model_selector.scores), 
                            x='Model', 
                            y='Score',
                            color='Score',
                            title="Model Performance Comparison"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Confusion Matrix for Classification
                        if st.session_state.task_type == 'C':
                            st.subheader("Model Evaluation")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Confusion Matrix**")
                                y_pred = model_selector.best_model.predict(model_selector.X_test)
                                cm = confusion_matrix(model_selector.y_test, y_pred)
                                fig = px.imshow(
                                    cm,
                                    labels=dict(x="Predicted", y="Actual"),
                                    x=sorted(model_selector.y_test.unique()),
                                    y=sorted(model_selector.y_test.unique()),
                                    text_auto=True
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.markdown("**Classification Report**")
                                report = classification_report(model_selector.y_test, y_pred, output_dict=True)
                                report_df = pd.DataFrame(report).transpose()
                                st.dataframe(report_df.style.highlight_max(axis=0))
                        else:
                            # Regression metrics
                            st.subheader("Regression Metrics")
                            y_pred = model_selector.best_model.predict(model_selector.X_test)
                            metrics = {
                                "R² Score": r2_score(model_selector.y_test, y_pred),
                                "MAE": mean_absolute_error(model_selector.y_test, y_pred),
                                "MSE": mean_squared_error(model_selector.y_test, y_pred),
                                "RMSE": np.sqrt(mean_squared_error(model_selector.y_test, y_pred))
                            }
                            metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                            st.dataframe(metrics_df.style.highlight_min(axis=0, subset=['Value']))
                            
                    except Exception as e:
                        st.error(f"❌ Error during model training: {str(e)}")

# ------------------ 3. DOWNLOAD PIPELINE ------------------
elif selected_nav == "💾 Download Pipeline":
    st.header("💾 Download Pipeline")
    
    if 'model' not in st.session_state:
        st.warning("Please train a model first!")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.text_input("Model Name", "my_model")
            
        with col2:
            st.markdown("**Pipeline Type**")
            pipeline_format = st.selectbox(
                "Select format:",
                options=["Joblib (.joblib)", "Pickle (.pkl)"],
                index=0
            )
        
        if st.button("🔧 Prepare Pipeline for Download"):
            try:
                # Create pipeline

                pipe = pipeline(
                    data=st.session_state.data.copy(),
                    target=st.session_state.target,
                    num_missing_mean=st.session_state.num_mean,
                    num_missing_median=st.session_state.num_median,
                    cat_missing=st.session_state.cat_missing,
                    best_model=st.session_state.model,
                    target_type=st.session_state.task_type
                )
                pipe.train_test()
                pipe.save_model()
                # Store the trained pipeline in session state
                st.session_state.trained_pipeline = pipe.pipelinee
                st.session_state.encoder = pipe.encoder
                
                st.success("✅ Pipeline prepared successfully!")
                
                # Show model details
                with st.expander("Model Details"):
                    st.write(f"**Model Type:** {'Classification' if st.session_state.task_type == 'C' else 'Regression'}")
                    st.write(f"**Target Variable:** {st.session_state.target}")
                    st.write(f"**Best Score:** {st.session_state.score:.4f}")
                    st.write("**Model Parameters:**")
                    st.json(st.session_state.model.get_params())
                    
            except Exception as e:
                st.error(f"❌ Error preparing pipeline: {str(e)}")
        
        # Download section
        if 'trained_pipeline' in st.session_state and st.session_state.trained_pipeline is not None:
            st.subheader("📥 Download Trained Pipeline")
            
            try:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if "Joblib" in pipeline_format:
                    file_extension = "joblib"
                    filename = f"{model_name}_{timestamp}.{file_extension}"
                    
                    # Create joblib buffer
                    buffer = io.BytesIO()
                    joblib.dump(st.session_state.trained_pipeline, buffer)
                    buffer.seek(0)
                    mime_type = "application/octet-stream"
                    
                else:  # Pickle
                    file_extension = "pkl"
                    filename = f"{model_name}_{timestamp}.{file_extension}"
                    
                    # Create pickle buffer
                    buffer = io.BytesIO()
                    pickle.dump(st.session_state.trained_pipeline, buffer)
                    buffer.seek(0)
                    mime_type = "application/octet-stream"
                
                st.download_button(
                    label=f"📥 Download {pipeline_format}",
                    data=buffer,
                    file_name=filename,
                    mime=mime_type,
                    use_container_width=True
                )
                
                st.success(f"✅ Pipeline ready for download as: {filename}")
                st.info("💡 **Note:** The downloaded pipeline can be loaded using `joblib.load()` or `pickle.load()` in your local environment.")
                
            except Exception as e:
                st.error(f"❌ Error creating download: {str(e)}")

# ------------------ 4. PREDICTIONS ------------------
elif selected_nav == "🔮 Make Predictions":
    st.header("🔮 Make Predictions")
    
    pipeline_model = st.session_state.trained_pipeline
    
    
    if pipeline_model is not None:
        try:
            # Get feature columns from pipeline
            feature_cols = []
            if hasattr(pipeline_model, 'named_steps') and hasattr(pipeline_model.named_steps['preprocessor'], 'feature_names_in_'):
                feature_cols = list(pipeline_model.named_steps['preprocessor'].feature_names_in_)
            
            if not feature_cols:
                st.error("❌ Could not determine required features from pipeline")
            else:
                # Input form with proper categorical handling
                with st.form("prediction_form"):
                    st.subheader("📝 Input Features")
                    input_data = {}
                    cols = st.columns(2)
                    
                    for i, col in enumerate(feature_cols):
                        with cols[i % 2]:
                            try:
                                if (st.session_state.get('data') is not None and 
                                    col in st.session_state.data.columns and 
                                    not pd.api.types.is_numeric_dtype(st.session_state.data[col])):
                                    
                                    # Handle categorical columns
                                    unique_vals = st.session_state.data[col].dropna().unique()
                                    unique_vals = sorted([str(x) for x in unique_vals])
                                    
                                    input_data[col] = st.selectbox(
                                        f"{col} (categorical)",
                                        options=unique_vals,
                                        index=0 if unique_vals else None
                                    )
                                else:
                                    # Handle numeric columns
                                    default_val = 0
                                    if (st.session_state.get('data') is not None and 
                                        col in st.session_state.data.columns):
                                        default_val = float(st.session_state.data[col].median())
                                    
                                    input_data[col] = st.number_input(
                                        f"{col} (numeric)",
                                        value=default_val,
                                        step=1.0
                                    )
                            except Exception as input_error:
                                st.error(f"⚠️ Error creating input for {col}: {str(input_error)}")
                                input_data[col] = None

                    
                    submitted = st.form_submit_button("🔮 Predict")
                    
                    if submitted:
                        with st.spinner("Making prediction..."):
                            try:
                                # Validate all inputs
                                if any(v is None for v in input_data.values()):
                                    st.error("❌ Some inputs are invalid. Please check all fields.")
                                else:
                                    # Create input DataFrame with correct dtypes
                                    input_df = pd.DataFrame([input_data])
                                    
                                    # Convert categoricals to strings
                                    for col in input_df.columns:
                                        if pd.api.types.is_string_dtype(st.session_state.data[col]):
                                            input_df[col] = input_df[col].astype(str)
                                    
                                    prediction = pipeline_model.predict(input_df)[0]
                                    if st.session_state.task_type=='C':
                                        if st.session_state.encoder:
                                            prediction = st.session_state.encoder.inverse_transform([prediction])[0]
                                    st.subheader("🎯 Prediction Result")
                                    if hasattr(pipeline_model, 'predict_proba'):
                                        # Classification output
                                        st.metric("Predicted Class", prediction)
                                        proba = pipeline_model.predict_proba(input_df)[0]
                                        classes = pipeline_model.classes_ if hasattr(pipeline_model, 'classes_') else range(len(proba))
                                        fig = px.bar(
                                            x=classes,
                                            y=proba,
                                            labels={'x': 'Class', 'y': 'Probability'},
                                            title="Class Probabilities"
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        # Regression output
                                        st.metric("Predicted Value", f"{prediction:.2f}")
                                        if st.session_state.get('target') is not None:
                                            fig = px.histogram(
                                                st.session_state.data, 
                                                x=st.session_state.target
                                            )
                                            fig.add_vline(
                                                x=prediction, 
                                                line_dash="dash", 
                                                line_color="red",
                                                annotation_text="Prediction"
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                            except Exception as pred_error:
                                st.error(f"❌ Prediction failed: {str(pred_error)}")
                                st.error("Please check your input values match the training data format")
        
        except Exception as e:
            st.error(f"❌ Error in prediction section: {str(e)}")
    
    else:
        st.warning("ℹ️ No pipeline available. Please train and save a model or upload a pipeline file")
def maintain_session():
    if 'last_ping' not in st.session_state:
        st.session_state.last_ping = time.time()
    if time.time() - st.session_state.last_ping > 60:  # Ping every 60 seconds
        st.session_state.last_ping = time.time()
        st.rerun()  # Gentle refresh

if __name__ == "__main__":
    maintain_session()
    st.markdown("""
    <hr style="margin-top: 50px;">
    <div style="text-align: center; padding: 10px 0; color: gray;">
        © 2025 M Salman - MSY. All rights reserved.
    </div>
    """, unsafe_allow_html=True)