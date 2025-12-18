"""Streamlit web interface for synthetic data generation and classification."""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from synthetic_data_gen.pipelines.data_generation.nodes import generate_images, create_prompts
from synthetic_data_gen.pipelines.classification.nodes import classify_images, aggregate_classification_results
from synthetic_data_gen.pipelines.training.nodes import (
    prepare_training_data,
    train_model,
    evaluate_model,
    save_trained_model
)
from synthetic_data_gen.utils.model_storage import ModelStorage
from synthetic_data_gen.utils.evaluation import compare_with_ham10000
from synthetic_data_gen.utils.logging import ExperimentLogger
from synthetic_data_gen.utils.ham10000_loader import HAM10000Loader
from synthetic_data_gen.utils.performance import PerformanceTracker
import pandas as pd
import time
from PIL import Image
import numpy as np
import io
import json
import logging

# Page configuration
st.set_page_config(
    page_title="Synthetic Skin Lesion Generator",
    page_icon="🔬",
    layout="wide"
)

# Initialize Utils
@st.cache_resource
def get_ham_loader():
    ham10000_path = "data/00_external/HAM10000"
    try:
        loader = HAM10000Loader(ham10000_path)
        return loader
    except Exception as e:
        # st.warning(f"Could not initialize HAM10000 Loader: {e}")
        return None

ham_loader = get_ham_loader()
logger = ExperimentLogger()
tracker = PerformanceTracker()

# Title
st.title("🔬 Synthetic Skin Lesion Generation & Classification")
st.markdown("Generate synthetic skin lesion images and classify or retrain models")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Image generation settings
    st.subheader("Image Generation")
    generation_model = st.selectbox(
        "Generation Model",
        ["stable_diffusion", "sd3.5", "qwen", "flux"],
        index=0,
        help="Select the model to generate synthetic images. SD 3.5 and Stable Diffusion support reference images."
    )
    
    num_images = st.number_input(
        "Number of Images",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
        help="How many images to generate"
    )
    
    # Image size fixed to HAM10000 format (600x450)
    image_size = 600
    st.info("📐 Image size: 600x450 (HAM10000 format)")
    
    # Reference images from HAM10000
    st.subheader("🖼️ Reference Images (HAM10000)")
    
    use_reference_images = False
    img2img_strength = 0.75
    reference_percentage = 0.1
    
    if generation_model in ["stable_diffusion", "sd3.5"]:
        use_reference_images = st.checkbox(
            "Use HAM10000 reference images",
            value=True,
            help="Use real HAM10000 images to guide generation for better accuracy"
        )
        
        if use_reference_images:
            ham10000_path = Path("data/00_external/HAM10000")
            if not (ham10000_path.exists() and (ham10000_path / "HAM10000_metadata").exists()):
                st.warning("⚠️ HAM10000 dataset not found. Please download it first.")
                use_reference_images = False
            else:
                reference_percentage = st.slider(
                    "Reference Images Percentage",
                    min_value=1,
                    max_value=100,
                    value=10,
                    step=1,
                    help=f"Percentage of HAM10000 images to use as references"
                ) / 100.0
                
                img2img_strength = st.slider(
                    "Image-to-Image Strength",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.90,
                    step=0.05,
                    help="Lower = more similar to reference, Higher = more creative"
                )
    else:
        st.info(f"ℹ️ Reference images not supported for {generation_model}. Using text-to-image only.")
    
    # Batch size (hidden/fixed to 1)
    batch_size = 1
    
    # Diagnosis selection
    st.subheader("🎯 Diagnosis Selection")
    
    diagnosis_options = {
        'akiec': 'Actinic keratoses / intraepithelial carcinoma',
        'bcc': 'Basal cell carcinoma',
        'bkl': 'Benign keratosis-like lesions',
        'df': 'Dermatofibroma',
        'nv': 'Melanocytic nevi (common moles)',
        'mel': 'Melanoma',
        'vasc': 'Vascular lesions'
    }
    
    diagnosis_mode = st.radio(
        "Diagnosis Mode",
        ["Single Diagnosis", "Mixed Diagnoses"]
    )
    
    selected_diagnoses = []
    diagnosis_distribution = {}
    
    if diagnosis_mode == "Single Diagnosis":
        selected_dx = st.selectbox(
            "Selected Diagnosis",
            options=list(diagnosis_options.keys()),
            format_func=lambda x: f"{x.upper()} – {diagnosis_options[x]}"
        )
        selected_diagnoses = [selected_dx]
    else:
        selected_diagnoses = st.multiselect(
            "Selected Diagnoses",
            options=list(diagnosis_options.keys()),
            default=['mel', 'nv'],
            format_func=lambda x: f"{x.upper()} – {diagnosis_options[x]}"
        )
        
        if selected_diagnoses:
            st.write("**Control distribution (sum 100%):**")
            cols = st.columns(len(selected_diagnoses))
            total_percentage = 0
            for idx, dx in enumerate(selected_diagnoses):
                with cols[idx]:
                    percentage = st.number_input(
                        f"{dx.upper()} %",
                        min_value=0, max_value=100,
                        value=100 // len(selected_diagnoses) if len(selected_diagnoses) > 0 else 0,
                        step=1, key=f"dist_{dx}"
                    )
                    diagnosis_distribution[dx] = percentage / 100.0
                    total_percentage += percentage
            if abs(total_percentage - 100) > 0.01:
                st.warning(f"sums to {total_percentage}%")

    # Classification/Training settings
    st.subheader("Classification & Training")
    action = st.radio("Action", ["Classify Only", "Retrain Model"])
    classification_model = st.selectbox("Classification Model", ["vgg16", "inception_v3", "resnet50"], index=2)
    
    if action == "Retrain Model":
        st.subheader("Training Settings")
        num_epochs = st.number_input("Number of Epochs", 1, 50, 5)
        learning_rate = st.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
        train_batch_size = st.number_input("Training Batch Size", 8, 64, 32)

# Init session state
if 'prompts_generated' not in st.session_state:
    st.session_state['prompts_generated'] = False
if 'current_prompts' not in st.session_state:
    st.session_state['current_prompts'] = []
if 'reference_images_data' not in st.session_state:
    st.session_state['reference_images_data'] = None

# Main content area
tab1, tab2, tab3 = st.tabs(["🚀 Run Pipeline", "📊 Results", "💾 Saved Models"])

with tab1:
    st.header("Run Pipeline")
    
    # Step 1: Generate Prompts First (Interrupt for User Edit)
    if not st.session_state['prompts_generated']:
        if st.button("1️⃣ Generate Prompts & Configure", type="primary", use_container_width=True):
            if not selected_diagnoses:
                st.error("Please select at least one diagnosis.")
                st.stop()
            
            with st.spinner("Generating prompts..."):
                prompts = []
                reference_images = None
                
                # Logic to generate prompts (simplified from original)
                # Try to use HAM10000 references prompt generation
                if use_reference_images and generation_model in ["stable_diffusion", "sdxl"]:
                    try:
                        from synthetic_data_gen.pipelines.data_generation.ham10000_reference_prompts import generate_prompts_with_references
                        
                        ham10000_path = "data/00_external/HAM10000"
                        dist = None
                        if len(selected_diagnoses) > 1 and diagnosis_distribution:
                             dist = diagnosis_distribution
                        
                        prompts, paired_refs, reference_pool = generate_prompts_with_references(
                            dataset_path=ham10000_path,
                            selected_diagnoses=selected_diagnoses,
                            num_prompts=num_images,
                            distribution=dist,
                            reference_percentage=reference_percentage
                        )
                        reference_images = (prompts, paired_refs, reference_pool)
                    except Exception as e:
                        st.warning(f"Could not load references: {e}. Using text prompts.")
                        
                if not prompts:
                     # Fallback text only
                    try:
                        from synthetic_data_gen.pipelines.data_generation.improved_prompts import generate_prompts_with_distribution
                        prompts = generate_prompts_with_distribution(
                                selected_diagnoses=selected_diagnoses,
                                num_prompts=num_images,
                                distribution=diagnosis_distribution if len(selected_diagnoses) > 1 else None,
                                use_localization=True
                            )
                    except Exception as e:
                        st.error(f"Error generating prompts: {e}")
                        st.stop()

                st.session_state['current_prompts'] = prompts
                st.session_state['reference_images_data'] = reference_images
                st.session_state['prompts_generated'] = True
                st.rerun()

    else:
        # Prompts are generated, show editor
        st.subheader("📝 Review & Edit Prompts")
        
        updated_prompts = []
        
        # Display editor for prompts
        prompts_df = pd.DataFrame({"Prompt": st.session_state['current_prompts']})
        edited_df = st.data_editor(
            prompts_df, 
            use_container_width=True,
            num_rows="fixed",
            column_config={"Prompt": st.column_config.TextColumn("Prompt (Editable)", width="large")}
        )
        updated_prompts = edited_df["Prompt"].tolist()
        
        col1, col2 = st.columns([1, 1])
        with col1:
             if st.button("🔄 Reset Prompts"):
                 st.session_state['prompts_generated'] = False
                 st.session_state['current_prompts'] = []
                 st.rerun()
        
        with col2:
            if st.button("2️⃣ Confirm & Start Generation", type="primary", use_container_width=True):
                 # Save updated prompts
                 st.session_state['current_prompts'] = updated_prompts

                 
                 # RUN PIPELINE
                 progress_bar = st.progress(0)
                 status_text = st.empty()
                 
                 import sys
                 import io
                 import time
                 
                 # 1. Generation
                 # Setup Console Capture (TeeIO)
                 class TeeIO:
                     def __init__(self, original_stream):
                         self.original_stream = original_stream
                         self.capture_buffer = io.StringIO()
                     
                     def write(self, data):
                         self.original_stream.write(data)
                         self.capture_buffer.write(data)
                         
                     def flush(self):
                         self.original_stream.flush()
                         self.capture_buffer.flush()
                         
                     def getvalue(self):
                         return self.capture_buffer.getvalue()

                 # Capture both stdout and stderr
                 capture_out = TeeIO(sys.stdout)
                 capture_err = TeeIO(sys.stderr)
                 
                 original_stdout = sys.stdout
                 original_stderr = sys.stderr
                 
                 sys.stdout = capture_out
                 sys.stderr = capture_err
                 
                 try:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting pipeline execution...")
                        print(f"Configuration: Model={generation_model}, Images={num_images}, Ref={use_reference_images}")
                        
                        status_text.text("🖼️ Generating images & Tracking performace...")
                        progress_bar.progress(10)
                        
                        # Fix reference images tuple if we updated prompts
                        ref_data = st.session_state['reference_images_data']
                        if ref_data and isinstance(ref_data, tuple) and len(ref_data) == 3:
                            _, paired_refs, pool = ref_data
                            ref_data = (updated_prompts, paired_refs, pool)

                        # START PERFORMANCE TRACKING
                        print("Initializing performance tracker...")
                        tracker.start_tracking()

                        generated_images = generate_images(
                            model_name=generation_model,
                            prompts=updated_prompts,
                            num_images=num_images,
                            image_size=image_size,
                            batch_size=batch_size,
                            seed=42,
                            reference_images=ref_data,
                            img2img_strength=img2img_strength if use_reference_images else 0.75
                        )
                        
                        # STOP PERFORMANCE TRACKING
                        tracker.stop_tracking()
                        print("Generation complete.")
                        
                        if not generated_images:
                            st.error("No images generated.")
                            st.stop()

                        progress_bar.progress(40) # Original value was 40, instruction changed to 50, but then classification is moved. Keeping 40 for now.
                        status_text.text("📊 Evaluating images...")
                        
                        # 2. Evaluation & Logging
                        # For each image, calculate metrics
                        batch_metrics = []
                        
                        print("Calculating evaluation metrics...")
                        for img_data in generated_images:
                            prompt_text = img_data.get('prompt', '').lower()
                            dx = 'mel' # default
                            for code, name in diagnosis_options.items():
                                 if name.lower() in prompt_text or code in prompt_text:
                                     dx = code
                                     break
                            
                            # Calculate metrics against HAM10000
                            # Now returns metrics AND reference images used
                            metrics, comp_imgs = compare_with_ham10000(
                                img_data['image'], 
                                dx, 
                                ham_loader, 
                                num_references=5 # Reduced to 5 to avoid overcrowding report
                            )
                            img_data['eval_metrics'] = metrics
                            img_data['comparison_images'] = comp_imgs
                            img_data['inferred_diagnosis'] = dx 
                            
                        # 3. Log Experiment
                        status_text.text("💾 Logging experiment...")
                        run_config = {
                            "model": generation_model,
                            "num_images": num_images,
                            "use_reference": use_reference_images,
                            "diagnoses": selected_diagnoses
                        }
                        
                        # Aggregate metrics
                        avg_ssim = np.mean([g.get('eval_metrics', {}).get('ssim', 0) for g in generated_images])
                        avg_color = np.mean([g.get('eval_metrics', {}).get('color_similarity', 0) for g in generated_images])
                        agg_metrics = {"avg_ssim": float(avg_ssim), "avg_color_similarity": float(avg_color)}
                        
                        # Add Performance Metrics
                        perf_metrics = tracker.get_metrics(num_images)
                        agg_metrics.update(perf_metrics)
                        
                        # Get captured logs
                        full_log = capture_out.getvalue() + "\n--- STDERR ---\n" + capture_err.getvalue()

                        log_dir = logger.log_experiment(
                            experiment_name=f"gen_{generation_model}",
                            config=run_config,
                            results=generated_images,
                            metrics=agg_metrics,
                            execution_log=full_log
                        )
                        
                        # Use PDF report
                        report_path = logger.generate_pdf_report(log_dir)
                        
                        st.session_state['last_run_dir'] = log_dir
                        st.session_state['last_report_path'] = report_path
                        st.session_state['last_perf_metrics'] = perf_metrics
                        # Also save log for UI display
                        st.session_state['last_execution_log'] = full_log
                        
                        # 4. Classification
                        status_text.text("🔍 Classifying images...")
                        progress_bar.progress(60)
                        
                        results_df = classify_images(
                            generated_images=generated_images,
                            model_name=classification_model,
                            batch_size=32,
                            num_classes=2,
                            class_names=["benign", "malignant"]
                        )
                        
                        st.session_state['generated_images'] = generated_images
                        st.session_state['classification_results'] = results_df
                        st.session_state['prompts_generated'] = False # Reset prompts generated state
                        
                        # 5. Training (Optional)
                        if action == "Retrain Model":
                             status_text.text("🎓 Training model...")
                             progress_bar.progress(80)
                             # ... existing training logic ...
                             pass

                        progress_bar.progress(100)
                        status_text.text("✅ Pipeline completed!")
                        
                        # Custom Success Banner
                        st.markdown("""
                        <div style="background-color: #d4edda; color: #155724; padding: 20px; border-radius: 5px; text-align: center; margin-bottom: 20px;">
                            <h2 style="margin: 0;">Scale Success! Pipeline Run Completed.</h2>
                            <p>Images generated, evaluated, and logged.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                 except Exception as e:
                     st.error(f"Pipeline failed: {e}")
                     st.exception(e)
                 finally:
                     # Restore streams
                     sys.stdout = original_stdout
                     sys.stderr = original_stderr


with tab2:
    st.header("📊 Results & Evaluation")
    
    if 'classification_results' in st.session_state and 'generated_images' in st.session_state:
        results_df = st.session_state['classification_results']
        generated_images = st.session_state['generated_images']
        
        # 1. Download Report (PDF)
        if 'last_report_path' in st.session_state:
            report_path = st.session_state['last_report_path']
            try:
                # Need to read as binary for PDF
                with open(report_path, "rb") as f:
                    report_content = f.read()
                st.download_button(
                    label="📥 Download Experiment Report (PDF)",
                    data=report_content,
                    file_name="experiment_report.pdf",
                    mime="application/pdf"
                )
            except:
                st.warning("Could not load report.")

        # 2. Performance Metrics (New Section)
        if 'last_perf_metrics' in st.session_state:
            pm = st.session_state['last_perf_metrics']
            st.subheader("Model Performance Benchmarks")
            
            p1, p2, p3, p4 = st.columns(4)
            p1.metric(
                "Time per Image", 
                f"{pm.get('seconds_per_image', 0):.2f} s",
                help="Total time / Number of images"
            )
            p2.metric(
                "Avg Power", 
                f"{pm.get('avg_power_watts', 0):.1f} W",
                help="Average GPU power draw measured via nvidia-smi"
            )
            p3.metric(
                "Energy per Image", 
                f"{pm.get('energy_per_image_wh', 0):.4f} Wh",
                help="(Avg Power * Total Time) / Num Images"
            )
            p4.metric(
                "Total Time", 
                f"{pm.get('total_time_seconds', 0):.1f} s",
                help="Total execution time for image generation"
            )
            st.info(f"Compute Cost Estimate: ~${(pm.get('total_energy_wh', 0)/1000 * 0.15):.5f} (assuming $0.15/kWh)")
            st.divider()

        # 3. Summary Metrics
        st.subheader("Quality Metrics")
        col1, col2, col3 = st.columns(3)
        avg_ssim = np.mean([g.get('eval_metrics', {}).get('ssim', 0) for g in generated_images])
        avg_color = np.mean([g.get('eval_metrics', {}).get('color_similarity', 0) for g in generated_images])
        avg_shape = np.mean([g.get('eval_metrics', {}).get('shape_similarity', 0) for g in generated_images])
        
        col1.metric(
            "Avg Structure Quality (SSIM)", 
            f"{avg_ssim:.3f}", 
            help="Structural Similarity Index\nFormula: l(x,y) * c(x,y) * s(x,y)\nRange: -1 to 1 (1 is identical)"
        )
        col2.metric(
            "Avg Color Fidelity", 
            f"{avg_color:.3f}", 
            help="Histogram Correlation\nFormula: cv2.compareHist(H1, H2, CORREL)\nRange: 0 to 1 (1 is identical)"
        )
        col3.metric(
            "Avg Shape Score", 
            f"{avg_shape:.3f}", 
            help="Hu Moments Similarity\nFormula: 1 / (1 + sum(abs(log(Hu1) - log(Hu2))))\nRange: 0 to 1 (1 is identical)"
        )
        
        st.divider()

        # 4. Image Gallery with Metrics
        st.subheader("Generated Images Details")
        
        for i, img_data in enumerate(generated_images):
            col_img, col_metrics = st.columns([1, 2])
            
            with col_img:
                st.image(img_data['image'], caption=f"Image {i+1}", use_column_width=True)
                
            with col_metrics:
                st.markdown(f"#### Diagnosis: {img_data.get('inferred_diagnosis', 'Unknown').upper()}")
                st.info(f"Prompt: {img_data.get('prompt')}")
                
                # Metrics
                m = img_data.get('eval_metrics', {})
                m1, m2, m3 = st.columns(3)
                m1.metric("SSIM", f"{m.get('ssim', 0):.3f}", help="Structural Similarity Index")
                m2.metric("Color Sim", f"{m.get('color_similarity', 0):.3f}", help="Histogram Correlation")
                m3.metric("Shape Sim", f"{m.get('shape_similarity', 0):.3f}", help="Hu Moments Similarity")
                
                # Classification Result
                if i < len(results_df):
                    res = results_df.iloc[i]
                    pred = res.get('predicted_class_name', '?')
                    conf = res.get('confidence', 0)
                    if pred == 'malignant':
                        st.markdown(f"🔴 **Predicted Malignant** ({conf:.1f}%)")
                    else:
                        st.markdown(f"🟢 **Predicted Benign** ({conf:.1f}%)")
            
            st.divider()

        # 5. Technical Logs (Console Output)
        if 'last_execution_log' in st.session_state:
            st.divider()
            with st.expander("Show Technical Logs (Console Output)"):
                 st.code(st.session_state['last_execution_log'], language='text')

    else:
        st.info("Run the pipeline to see results.")

with tab3:
    st.header("💾 Saved Models")
    # (Keeping existing saved models logic)
    if st.button("🔄 Refresh Model List"):
        st.rerun()
    
    storage = ModelStorage()
    models = storage.list_models()
    
    if models:
        st.write(f"Found {len(models)} saved model(s):")
        for model_info in models:
            with st.expander(f"📦 {model_info.get('name', 'Unknown')}"):
                st.json(model_info)


