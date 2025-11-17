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
import pandas as pd
import time
from PIL import Image
import numpy as np
import io

# Page configuration
st.set_page_config(
    page_title="Synthetic Skin Lesion Generator",
    page_icon="🔬",
    layout="wide"
)

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
        ["stable_diffusion", "sdxl", "qwen", "flux"],
        index=0,
        help="Select the model to generate synthetic images. SDXL and Stable Diffusion support reference images."
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
    # Note: Generated at 600x448 (divisible by 8) then resized to 600x450
    image_size = 600
    st.info("📐 Image size: 600x450 (HAM10000 format, generated at 600x448 for compatibility)")
    
    # Reference images from HAM10000
    st.subheader("🖼️ Reference Images (HAM10000)")
    use_reference_images = st.checkbox(
        "Use HAM10000 reference images",
        value=True,
        help="Use real HAM10000 images to guide generation for better accuracy"
    )
    
    img2img_strength = 0.75  # Default value
    reference_percentage = 0.1  # Default 10%
    if use_reference_images:
        ham10000_path = Path("data/00_external/HAM10000")
        if not (ham10000_path.exists() and (ham10000_path / "HAM10000_metadata").exists()):
            st.warning("⚠️ HAM10000 dataset not found. Please download it first.")
            use_reference_images = False
        else:
            # Calculate total available images for selected diagnoses
            try:
                from synthetic_data_gen.utils.ham10000_loader import HAM10000Loader
                import pandas as pd
                
                metadata_path = ham10000_path / "HAM10000_metadata"
                if metadata_path.exists():
                    metadata = pd.read_csv(metadata_path)
                    # This will be updated when diagnoses are selected, but we show a placeholder
                    total_ham10000 = len(metadata)
                    st.info(f"📊 HAM10000 dataset: {total_ham10000} total images available")
            except:
                total_ham10000 = 10000  # Default estimate
            
            reference_percentage = st.slider(
                "Reference Images Percentage",
                min_value=1,
                max_value=100,
                value=10,
                step=1,
                help=f"Percentage of HAM10000 images to use as references (1% = ~100 images, 100% = all {total_ham10000 if 'total_ham10000' in locals() else 10000} images)"
            ) / 100.0  # Convert to 0.0-1.0 range
            
            img2img_strength = st.slider(
                "Image-to-Image Strength",
                min_value=0.0,
                max_value=1.0,
                value=0.90,
                step=0.05,
                help="Lower = more similar to reference, Higher = more creative (0.90 recommended for maximum reference following)"
            )
            
            st.warning("⚠️ **Important:** Use high strength (0.90-0.95) to ensure generated images closely follow the HAM10000 reference images and avoid body parts. Lower values may produce unwanted results.")
    
    batch_size = st.number_input(
        "Batch Size",
        min_value=1,
        max_value=8,
        value=2,
        step=1,
        help="Number of images to generate at once"
    )
    
    # Diagnosis selection
    st.subheader("🎯 Diagnosis Selection")
    
    diagnosis_options = {
        'akiec': 'Actinic keratoses / intraepithelial carcinoma',
        'bcc': 'Basal cell carcinoma',
        'bkl': 'Benign keratosis-like lesions',
        'df': 'Dermatofibroma',
        'nv': 'Melanocytic nevi (common moles)',
        'mel': 'Melanoma',
        'vasc': 'Vascular lesions (e.g., hemangioma, angioma)'
    }
    
    diagnosis_mode = st.radio(
        "Diagnosis Mode",
        ["Single Diagnosis", "Mixed Diagnoses"],
        help="Generate images of one type or a mix of types"
    )
    
    selected_diagnoses = []
    diagnosis_distribution = {}
    
    if diagnosis_mode == "Single Diagnosis":
        selected_dx = st.selectbox(
            "Select Diagnosis",
            options=list(diagnosis_options.keys()),
            format_func=lambda x: f"{x.upper()} – {diagnosis_options[x]}",
            help="Choose the type of lesion to generate"
        )
        selected_diagnoses = [selected_dx]
    else:
        # Multiple diagnoses with distribution control
        st.write("**Select diagnoses to include:**")
        selected_diagnoses = st.multiselect(
            "Diagnoses",
            options=list(diagnosis_options.keys()),
            default=['mel', 'nv'],
            format_func=lambda x: f"{x.upper()} – {diagnosis_options[x]}",
            help="Select multiple diagnosis types"
        )
        
        if selected_diagnoses:
            st.write("**Control distribution (percentages must sum to 100%):**")
            
            cols = st.columns(len(selected_diagnoses))
            total_percentage = 0
            
            for idx, dx in enumerate(selected_diagnoses):
                with cols[idx]:
                    percentage = st.number_input(
                        f"{dx.upper()} %",
                        min_value=0,
                        max_value=100,
                        value=100 // len(selected_diagnoses) if len(selected_diagnoses) > 0 else 0,
                        step=1,
                        key=f"dist_{dx}"
                    )
                    diagnosis_distribution[dx] = percentage / 100.0
                    total_percentage += percentage
            
            if abs(total_percentage - 100) > 0.01:
                st.warning(f"⚠️ Distribution sums to {total_percentage}%. Please adjust to 100%.")
            else:
                st.success(f"✅ Distribution: {total_percentage}%")
        else:
            st.warning("⚠️ Please select at least one diagnosis type.")
    
    # Classification/Training settings
    st.subheader("Classification & Training")
    action = st.radio(
        "Action",
        ["Classify Only", "Retrain Model"],
        help="Choose to only classify images or also retrain the model"
    )
    
    classification_model = st.selectbox(
        "Classification Model",
        ["vgg16", "inception_v3", "resnet50"],
        index=2,
        help="Model to use for classification/training"
    )
    
    # Training settings (only shown if retraining)
    if action == "Retrain Model":
        st.subheader("Training Settings")
        num_epochs = st.number_input(
            "Number of Epochs",
            min_value=1,
            max_value=50,
            value=5,
            step=1
        )
        
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.01,
            value=0.001,
            step=0.0001,
            format="%.4f"
        )
        
        train_batch_size = st.number_input(
            "Training Batch Size",
            min_value=8,
            max_value=64,
            value=32,
            step=8
        )

# Main content area
tab1, tab2, tab3 = st.tabs(["🚀 Run Pipeline", "📊 Results", "💾 Saved Models"])

with tab1:
    st.header("Run Pipeline")
    
    if st.button("▶️ Start Pipeline", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        try:
            # Step 1: Generate images
            status_text.text("🖼️ Step 1/3: Generating images...")
            progress_bar.progress(10)
            
            # Generate prompts and reference images
            if not selected_diagnoses:
                st.error("❌ Please select at least one diagnosis type.")
                st.stop()
            
            prompts = []
            reference_images = None
            
            if use_reference_images and generation_model in ["stable_diffusion", "sdxl"]:
                # Use HAM10000 reference images for image-guided generation
                try:
                    from synthetic_data_gen.pipelines.data_generation.ham10000_reference_prompts import (
                        generate_prompts_with_references
                    )
                    
                    ham10000_path = "data/00_external/HAM10000"
                    
                    # Normalize distribution if provided
                    dist = None
                    if len(selected_diagnoses) > 1:
                        total_percentage = sum(diagnosis_distribution.values()) if diagnosis_distribution else 0
                        if diagnosis_distribution and abs(total_percentage - 100) < 0.01:
                            dist = diagnosis_distribution
                    
                    prompts, paired_references, reference_pool = generate_prompts_with_references(
                        dataset_path=ham10000_path,
                        selected_diagnoses=selected_diagnoses,
                        num_prompts=num_images,
                        distribution=dist,
                        reference_percentage=reference_percentage
                    )
                    
                    # Store as tuple to pass reference pool to generation
                    reference_images = (prompts, paired_references, reference_pool)
                    
                    st.success(f"✅ Generated {len(prompts)} prompts. Loaded {len(reference_pool)} reference images in pool ({reference_percentage*100:.1f}% of available). Will randomly select from pool during generation.")
                    
                except Exception as e:
                    st.warning(f"⚠️ Could not load HAM10000 references: {e}. Using text-only prompts.")
                    use_reference_images = False
                    reference_images = None
            
            if not prompts:
                # Fallback to text-only prompts
                try:
                    from synthetic_data_gen.pipelines.data_generation.improved_prompts import (
                        generate_single_diagnosis_prompts,
                        generate_prompts_with_distribution
                    )
                    
                    if len(selected_diagnoses) == 1:
                        prompts = generate_single_diagnosis_prompts(
                            diagnosis=selected_diagnoses[0],
                            num_prompts=num_images,
                            use_localization=True
                        )
                    else:
                        total_percentage = sum(diagnosis_distribution.values()) if diagnosis_distribution else 0
                        if diagnosis_distribution and abs(total_percentage - 100) < 0.01:
                            prompts = generate_prompts_with_distribution(
                                selected_diagnoses=selected_diagnoses,
                                num_prompts=num_images,
                                distribution=diagnosis_distribution,
                                use_localization=True
                            )
                        else:
                            prompts = generate_prompts_with_distribution(
                                selected_diagnoses=selected_diagnoses,
                                num_prompts=num_images,
                                distribution=None,
                                use_localization=True
                            )
                    
                    st.info(f"✅ Generated {len(prompts)} text-only prompts")
                    
                except Exception as e:
                    st.error(f"❌ Error generating prompts: {e}")
                    st.exception(e)
                    st.stop()
            
            with st.spinner(f"Generating {num_images} images using {generation_model}..."):
                try:
                    # Debug: show prompt info
                    if prompts:
                        with st.expander("🔍 View first prompt"):
                            st.text(prompts[0])
                    
                    # Generate images with or without reference images
                    generated_images = generate_images(
                        model_name=generation_model,
                        prompts=prompts,
                        num_images=num_images,
                        image_size=image_size,
                        batch_size=batch_size,
                        seed=42,
                        reference_images=reference_images,
                        img2img_strength=img2img_strength if use_reference_images else 0.75
                    )
                    
                    if not generated_images:
                        st.error("❌ No images were generated! Check the logs for errors.")
                        st.stop()
                    
                except Exception as e:
                    st.error(f"❌ Error during image generation: {e}")
                    st.exception(e)
                    st.stop()
            
            progress_bar.progress(40)
            status_text.text(f"✅ Generated {len(generated_images)} images")
            st.success(f"Successfully generated {len(generated_images)} images!")
            
            # Ensure all images are PIL Images and in RGB mode before storing
            for img_data in generated_images:
                img = img_data.get('image')
                if img is not None and isinstance(img, Image.Image):
                    # Ensure RGB mode for proper display
                    if img.mode != 'RGB':
                        img_data['image'] = img.convert('RGB')
            
            # Step 2: Classify images
            status_text.text("🔍 Step 2/3: Classifying images...")
            progress_bar.progress(50)
            
            with st.spinner(f"Classifying images using {classification_model}..."):
                results_df = classify_images(
                    generated_images=generated_images,
                    model_name=classification_model,
                    batch_size=32,
                    num_classes=2,
                    class_names=["benign", "malignant"]
                )
            
            progress_bar.progress(70)
            status_text.text("✅ Classification complete")
            
            # Store results in session state
            st.session_state['generated_images'] = generated_images
            st.session_state['classification_results'] = results_df
            st.session_state['classification_model'] = classification_model
            
            # Step 3: Training (if selected)
            if action == "Retrain Model":
                status_text.text("🎓 Step 3/3: Training model...")
                progress_bar.progress(75)
                
                with st.spinner(f"Training {classification_model} with synthetic data..."):
                    # Prepare data
                    train_dataset, val_dataset, test_dataset = prepare_training_data(
                        generated_images=generated_images,
                        use_synthetic=True,
                        synthetic_ratio=1.0
                    )
                    
                    # Train model
                    trained_model, training_history = train_model(
                        model_name=classification_model,
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        num_classes=2,
                        batch_size=train_batch_size,
                        learning_rate=learning_rate,
                        num_epochs=num_epochs
                    )
                    
                    # Evaluate model
                    evaluation_metrics = evaluate_model(
                        model=trained_model,
                        test_dataset=test_dataset,
                        batch_size=train_batch_size
                    )
                    
                    # Save model
                    training_params = {
                        "num_epochs": num_epochs,
                        "learning_rate": learning_rate,
                        "batch_size": train_batch_size,
                        "num_images": num_images
                    }
                    
                    model_path = save_trained_model(
                        model=trained_model,
                        model_name=classification_model,
                        evaluation_metrics=evaluation_metrics,
                        training_history=training_history,
                        training_params=training_params
                    )
                
                progress_bar.progress(100)
                status_text.text("✅ Training complete")
                
                # Store training results
                st.session_state['evaluation_metrics'] = evaluation_metrics
                st.session_state['training_history'] = training_history
                st.session_state['model_path'] = model_path
                st.session_state['trained'] = True
            else:
                progress_bar.progress(100)
                st.session_state['trained'] = False
            
            status_text.text("✅ Pipeline completed successfully!")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
            progress_bar.progress(0)
            status_text.text("❌ Pipeline failed")

with tab2:
    st.header("📊 Results")
    
    if 'classification_results' in st.session_state:
        results_df = st.session_state['classification_results']
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Images", len(results_df))
        
        with col2:
            if 'confidence' in results_df.columns:
                avg_conf = results_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf:.1f}%")
        
        with col3:
            if 'predicted_class_name' in results_df.columns:
                benign_count = len(results_df[results_df['predicted_class_name'] == 'benign'])
                st.metric("Benign", benign_count)
        
        with col4:
            if 'predicted_class_name' in results_df.columns:
                malignant_count = len(results_df[results_df['predicted_class_name'] == 'malignant'])
                st.metric("Malignant", malignant_count)
        
        # Generated Images Gallery
        if 'generated_images' in st.session_state:
            st.markdown("---")
            st.subheader("🖼️ Generated Images")
            
            generated_images = st.session_state['generated_images']
            st.markdown(f"*Showing {len(generated_images)} generated image(s)*")
            
            # Debug info (can be removed later)
            with st.expander("🔧 Debug Info (click to expand)"):
                st.write(f"Number of images in session: {len(generated_images)}")
                if generated_images:
                    first_img = generated_images[0].get('image')
                    st.write(f"First image type: {type(first_img)}")
                    if hasattr(first_img, 'mode'):
                        st.write(f"First image mode: {first_img.mode}")
                    if hasattr(first_img, 'size'):
                        st.write(f"First image size: {first_img.size}")
            
            # Display images in a grid (adjust columns based on number of images)
            if len(generated_images) == 0:
                st.warning("No images to display")
            else:
                if len(generated_images) <= 3:
                    num_cols = len(generated_images)
                else:
                    num_cols = 3
                
                # Ensure num_cols is at least 1 to avoid division by zero
                num_cols = max(1, num_cols)
                num_rows = (len(generated_images) + num_cols - 1) // num_cols
                
                for row in range(num_rows):
                    cols = st.columns(num_cols)
                    for col_idx, col in enumerate(cols):
                        img_idx = row * num_cols + col_idx
                        if img_idx < len(generated_images):
                            with col:
                                # Get image and metadata
                                img_data = generated_images[img_idx]
                                image = img_data.get('image')
                                
                                # Ensure image is a PIL Image and convert if needed
                                if image is None:
                                    st.error(f"Image {img_idx + 1}: No image data found")
                                    continue
                                
                                # Convert to PIL Image if it's not already
                                if not isinstance(image, Image.Image):
                                    try:
                                        if isinstance(image, np.ndarray):
                                            # Convert numpy array to PIL Image
                                            if image.max() <= 1.0:
                                                image = (image * 255).astype(np.uint8)
                                            image = Image.fromarray(image)
                                        else:
                                            st.error(f"Image {img_idx + 1}: Unknown image type: {type(image)}")
                                            continue
                                    except Exception as e:
                                        st.error(f"Image {img_idx + 1}: Error converting image: {e}")
                                        continue
                                
                                # Ensure image is in RGB mode
                                if image.mode != 'RGB':
                                    try:
                                        image = image.convert('RGB')
                                    except Exception as e:
                                        st.warning(f"Image {img_idx + 1}: Could not convert to RGB: {e}")
                                
                                # Get classification result for this image
                                if img_idx < len(results_df):
                                    result = results_df.iloc[img_idx]
                                    predicted_class = result.get('predicted_class_name', 'unknown')
                                    confidence = result.get('confidence', 0)
                                    
                                    # Create a card-like container
                                    st.markdown(f"### Image {img_idx + 1}")
                                    
                                    # Display image with border
                                    try:
                                        # Verify image is valid before displaying
                                        if not hasattr(image, 'size') or image.size[0] == 0 or image.size[1] == 0:
                                            st.error(f"Image {img_idx + 1}: Invalid image size")
                                            continue
                                        
                                        # Try to get a fresh copy of the image data
                                        # Sometimes PIL Images in session state need to be refreshed
                                        img_bytes = io.BytesIO()
                                        image.save(img_bytes, format='PNG')
                                        img_bytes.seek(0)
                                        refreshed_image = Image.open(img_bytes)
                                        
                                        st.image(refreshed_image, width='stretch', channels='RGB')
                                    except Exception as e:
                                        st.error(f"Error displaying image {img_idx + 1}: {e}")
                                        # Try to show image info for debugging
                                        st.text(f"Image type: {type(image)}")
                                        st.text(f"Image mode: {image.mode if hasattr(image, 'mode') else 'N/A'}")
                                        st.text(f"Image size: {image.size if hasattr(image, 'size') else 'N/A'}")
                                        # Try direct display as fallback
                                        try:
                                            st.image(image, width='stretch')
                                        except:
                                            st.text("Could not display image")
                                        continue
                                    
                                    # Classification badge
                                    if predicted_class == 'malignant':
                                        st.markdown(
                                            f'<div style="background-color: #ffebee; padding: 10px; border-radius: 5px; margin: 10px 0;">'
                                            f'🔴 <strong>Predicted: Malignant</strong><br>'
                                            f'Confidence: {confidence:.1f}%'
                                            f'</div>',
                                            unsafe_allow_html=True
                                        )
                                    else:
                                        st.markdown(
                                            f'<div style="background-color: #e8f5e9; padding: 10px; border-radius: 5px; margin: 10px 0;">'
                                            f'🟢 <strong>Predicted: Benign</strong><br>'
                                            f'Confidence: {confidence:.1f}%'
                                            f'</div>',
                                            unsafe_allow_html=True
                                        )
                                    
                                    # Show prompt if available
                                    prompt = img_data.get('prompt', '')
                                    if prompt:
                                        with st.expander("📝 View Generation Prompt"):
                                            st.text(prompt)
                                    
                                    # Show generation model
                                    gen_model = img_data.get('model', 'unknown')
                                    st.caption(f"Generated with: {gen_model}")
                                    
                                    st.markdown("---")
        
        # Detailed results table
        st.subheader("Detailed Results")
        st.dataframe(results_df, width='stretch')
        
        # Class distribution chart
        if 'predicted_class_name' in results_df.columns:
            st.subheader("Class Distribution")
            class_counts = results_df['predicted_class_name'].value_counts()
            st.bar_chart(class_counts)
        
        # Confidence distribution
        if 'confidence' in results_df.columns:
            st.subheader("Confidence Distribution")
            st.bar_chart(results_df['confidence'])
        
        # Training results (if available)
        if st.session_state.get('trained', False):
            st.subheader("🎓 Training Results")
            
            metrics = st.session_state.get('evaluation_metrics', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.2f}%")
            
            with col2:
                st.metric("Test Precision", f"{metrics.get('test_precision', 0):.2f}%")
            
            with col3:
                st.metric("Test Recall", f"{metrics.get('test_recall', 0):.2f}%")
            
            with col4:
                st.metric("Test F1", f"{metrics.get('test_f1', 0):.2f}%")
            
            # Training history chart
            if 'training_history' in st.session_state:
                history = st.session_state['training_history']
                st.subheader("Training History")
                
                history_df = pd.DataFrame({
                    'Epoch': range(1, len(history['train_loss']) + 1),
                    'Train Loss': history['train_loss'],
                    'Val Loss': history['val_loss'],
                    'Train Acc': history['train_acc'],
                    'Val Acc': history['val_acc']
                })
                
                st.line_chart(history_df.set_index('Epoch'))
            
            # Model path
            if 'model_path' in st.session_state:
                st.info(f"💾 Model saved to: `{st.session_state['model_path']}`")
    else:
        st.info("👈 Run the pipeline first to see results here")

with tab3:
    st.header("💾 Saved Models")
    
    if st.button("🔄 Refresh Model List"):
        st.rerun()
    
    storage = ModelStorage()
    models = storage.list_models()
    
    if models:
        st.write(f"Found {len(models)} saved model(s):")
        
        for model_info in models:
            with st.expander(f"📦 {model_info.get('name', 'Unknown')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Path:**")
                    st.code(model_info.get('path', 'Unknown'), language=None)
                
                with col2:
                    if 'timestamp' in model_info:
                        st.write("**Timestamp:**")
                        st.write(model_info['timestamp'])
                
                if 'metrics' in model_info:
                    st.write("**Metrics:**")
                    metrics = model_info['metrics']
                    metrics_df = pd.DataFrame([metrics]).T
                    metrics_df.columns = ['Value']
                    st.dataframe(metrics_df, width='stretch')
                
                if 'training_params' in model_info:
                    st.write("**Training Parameters:**")
                    st.json(model_info['training_params'])
    else:
        st.info("No saved models found. Train a model first!")

# Footer
st.markdown("---")
st.markdown("**Synthetic Skin Lesion Generation Pipeline** | Built with Streamlit")

