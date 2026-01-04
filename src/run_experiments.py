"""
Experiment Runner Script for Skin Cancer Synthetic Data Generation.
"""

import os
import sys
import argparse
import logging
import random
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import itertools
from fpdf import FPDF
import json
import shutil
import numpy as np
from PIL import Image

# Add src to python path to allow imports
sys.path.append(str(Path(__file__).parent))

from synthetic_data_gen.pipelines.data_generation.nodes import (
    generate_stable_diffusion_images,
    generate_sdxl_images, # Actually SD3.5
    generate_qwen_images,
    generate_stable_diffusion_img2img
)
from synthetic_data_gen.utils.ham10000_loader import HAM10000Loader
from synthetic_data_gen.utils.evaluation import compare_with_ham10000
from synthetic_data_gen.utils.performance import PerformanceTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ExperimentRunner")

# Constants
LESION_TYPES = {
    'nv': 'melanocytic nevus',
    'mel': 'melanoma',
    'bkl': 'benign keratosis',
    'bcc': 'basal cell carcinoma',
    'akiec': 'actinic keratosis',
    'vasc': 'vascular lesion',
    'df': 'dermatofibroma'
}
LESION_KEYS = list(LESION_TYPES.keys())

PROMPT_DIR = Path(__file__).parent.parent / "conf" / "prompts"
LOG_DIR = Path(__file__).parent.parent / "logs" / "experiments"
DATA_DIR = Path(__file__).parent.parent / "data" / "00_external" / "HAM10000"

def load_template(template_name: str) -> str:
    """Load a prompt template from file."""
    path = PROMPT_DIR / f"{template_name}.txt"
    if not path.exists():
        logger.error(f"Template {template_name} not found at {path}")
        raise FileNotFoundError(f"Template {template_name} not found")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def get_ham_loader():
    """Initialize HAM10000 Loader."""
    if DATA_DIR.exists():
        try:
            return HAM10000Loader(str(DATA_DIR))
        except Exception as e:
            logger.warning(f"Could not init HAM10000Loader: {e}")
            return None
    logger.warning(f"HAM10000 dataset not found at {DATA_DIR}")
    return None

def setup_run_directory(experiment_name: str) -> Tuple[Path, Path]:
    """Create run directory and images subdirectory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOG_DIR / f"{experiment_name}_{timestamp}"
    images_dir = run_dir / "images"
    
    run_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created run directory: {run_dir}")
    return run_dir, images_dir

def save_image_and_refs(image: Any, run_dir: Path, idx: int, ref_images: List[Any] = None) -> Tuple[str, List[str]]:
    """Save generated image and optional reference images."""
    # Save main image
    filename = f"image_{idx:03d}.png"
    abs_path = run_dir / "images" / filename
    if image is not None:
        image.save(abs_path)
    
    gen_rel_path = str(abs_path) 
    
    ref_paths = []
    if ref_images:
        for r_idx, ref_img in enumerate(ref_images):
            # Ensure ref_img is PIL Image
            if not isinstance(ref_img, Image.Image):
                 continue
            ref_filename = f"ref_{idx:03d}_{r_idx}.png"
            ref_path = run_dir / "images" / ref_filename
            ref_img.save(ref_path)
            ref_paths.append(str(ref_path))
            
    return str(abs_path), ref_paths

def save_advanced_pdf_report(results: List[Dict], run_dir: Path, experiment_name: str, timestamp: str, config: Dict):
    """Generate a sophisticated PDF report."""
    filename = run_dir / "report.pdf"
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- Page 1: Summary ---
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, "Experiment Report", ln=True, align="C")
    
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Run ID: {run_dir.name}", ln=True)
    pdf.cell(0, 10, f"Date: {timestamp}", ln=True)
    pdf.ln(5)
    
    # Metrics Table
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Performance & Evaluation Metrics", ln=True, fill=True)
    pdf.ln(2)
    
    # Aggregate calculations
    df = pd.DataFrame(results)
    avg_ssim = df['ssim'].mean() if 'ssim' in df.columns else 0.0
    avg_color = df['color_similarity'].mean() if 'color_similarity' in df.columns else 0.0
    
    # Performance aggregation
    total_time = 0.0
    avg_power = 0.0
    energy_per_img = 0.0
    total_energy_wh = 0.0
    
    if 'performance' in df.columns:
        # performance is a dict in each row. Since it's per batch, and rows are images,
        # we need to be careful not to sum duplicated batch data if we were tracking batches.
        # But here run_generation_and_eval returns list of results for that specific call.
        # So taking mean of averages or sum of totals depends on how we want to present.
        # Simplest: Average per-image metrics derived from the 'performance' dicts.
        
        # Extract per-image metrics
        times_per_img = [r.get('performance', {}).get('seconds_per_image', 0) for r in results]
        powers = [r.get('performance', {}).get('avg_power_watts', 0) for r in results]
        energies_per_img = [r.get('performance', {}).get('energy_per_image_wh', 0) for r in results]
        
        total_time = sum(times_per_img) # Approx total run time
        avg_power = np.mean(powers) if powers else 0.0
        energy_per_img = np.mean(energies_per_img) if energies_per_img else 0.0
        total_energy_wh = sum(energies_per_img)
        
    cost_estimate = (total_energy_wh / 1000.0) * 0.15 # $0.15 per kWh
    
    metrics = [
        ("Avg SSIM", f"{avg_ssim:.4f}"),
        ("Avg Color Similarity", f"{avg_color:.4f}"),
        ("Total Images", f"{len(df)}"),
        ("Models Used", ", ".join(df['model'].unique()) if 'model' in df.columns else "N/A"),
        ("Total Time (s)", f"{total_time:.2f}"),
        ("Time per Image (s)", f"{total_time/len(df):.2f}" if len(df) > 0 else "0.00"),
        ("Avg Power (W)", f"{avg_power:.2f}"),
        ("Energy/Image (Wh)", f"{energy_per_img:.4f}"),
        ("Est. Cost (EUR)", f"{cost_estimate:.6f}")
    ]
    
    pdf.set_font("Arial", "", 11)
    for key, val in metrics:
        pdf.cell(95, 8, key, border=1)
        pdf.cell(95, 8, val, border=1, ln=True)
    pdf.ln(10)
    
    # Configuration
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Configuration", ln=True, fill=True)
    pdf.set_font("Courier", "", 8)
    # Convert config to string safely
    def config_serializer(obj):
        if isinstance(obj, (datetime, Path)):
            return str(obj)
        return str(obj)
        
    config_str = json.dumps(config, indent=2, default=config_serializer)
    pdf.multi_cell(0, 4, config_str)
    
    # --- Subsequent Pages: Image Details ---
    for i, res in enumerate(results):
        pdf.add_page()
        
        # Header
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"Generated Image {i+1}", ln=True, fill=True)
        pdf.ln(5)
        
        # Image placement logic
        y_start = pdf.get_y()
        img_path = res.get('image_path')
        
        # Check if image exists
        if img_path and os.path.exists(img_path):
            try:
                # Place Image Left (w=100mm)
                pdf.image(img_path, x=10, y=y_start, w=100)
            except Exception as e:
                logger.error(f"Error embedding image in PDF: {e}")
                pdf.set_xy(10, y_start)
                pdf.cell(100, 10, f"[Image Error: {e}]", border=1)
        else:
            pdf.set_xy(10, y_start)
            pdf.cell(100, 10, "[Image Not Found]", border=1)
            
        # Info Column (Right side, x=120)
        pdf.set_xy(120, y_start)
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 5, "Metadata:", ln=True)
        
        pdf.set_font("Arial", "", 9)
        pdf.set_x(120)
        pdf.multi_cell(80, 5, f"Model: {res.get('model')}\nDiagnosis: {res.get('inferred_diagnosis')}\nSize: {res.get('image_size', 'N/A')}")
        
        pdf.ln(2)
        pdf.set_xy(120, pdf.get_y())
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 5, "Metrics:", ln=True)
        
        pdf.set_font("Arial", "", 9)
        pdf.set_x(120)
        pdf.cell(0, 5, f"SSIM: {res.get('ssim', 0):.3f}", ln=True)
        pdf.set_x(120)
        pdf.cell(0, 5, f"Color Sim: {res.get('color_similarity', 0):.3f}", ln=True)
        pdf.set_x(120)
        pdf.cell(0, 5, f"Shape Sim: {res.get('shape_similarity', 0):.3f}", ln=True)
        
        # Move cursor below image area (approx 75mm height + margin)
        pdf.set_xy(10, y_start + 85)
        
        # Prompt
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 5, "Prompt:", ln=True)
        pdf.set_font("Arial", "I", 9)
        pdf.multi_cell(0, 5, str(res.get('prompt', '')))
        pdf.ln(5)
        
        # Reference Images
        ref_paths = res.get('reference_image_paths', [])
        if ref_paths:
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 8, "Reference Images (Comparison):", ln=True)
            
            x_curr = 10
            y_curr = pdf.get_y()
            
            for r_path in ref_paths[:5]:
                if os.path.exists(r_path):
                    try:
                        pdf.image(r_path, x=x_curr, y=y_curr, w=30, h=30)
                        x_curr += 35
                    except:
                        pass
                        
            pdf.ln(35) # Valid spacing after row

    try:
        pdf.output(str(filename))
        logger.info(f"PDF Report saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save PDF report: {e}")

def save_run_data(results: List[Dict], run_dir: Path, experiment_name: str, config: Dict):
    """Save CSV, JSON and PDF reports."""
    timestamp = run_dir.name.split('_')[-1]
    
    # 1. results_summary.csv
    csv_path = run_dir / "results_summary.csv"
    df = pd.DataFrame(results)
    # Clean up non-serializable for CSV
    df_csv = df.copy()
    keys_to_drop = ['image', 'reference_images']
    for k in keys_to_drop:
        if k in df_csv.columns:
            del df_csv[k]
    
    df_csv.to_csv(csv_path, index=False)
    logger.info(f"CSV Summary saved to {csv_path}")
    
    # 2. experiment_log.json
    json_path = run_dir / "experiment_log.json"
    
    # Prepare JSON serializable results
    json_results = []
    for r in results:
        r_copy = r.copy()
        for k in ['image', 'reference_images']:
            if k in r_copy:
                del r_copy[k]
        json_results.append(r_copy)
    
    log_data = {
        "run_id": run_dir.name,
        "timestamp": timestamp,
        "config": config,
        "results": json_results,
        "aggregate_metrics": {
            "total_images": len(results),
            "avg_ssim": float(df['ssim'].mean()) if 'ssim' in df.columns else 0.0
        }
    }
    
    def json_serial(obj):
        if isinstance(obj, (datetime, Path)):
            return str(obj)
        # Handle numpy types
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return str(obj)

    with open(json_path, 'w') as f:
        json.dump(log_data, f, indent=2, default=json_serial)
    logger.info(f"JSON Log saved to {json_path}")
    
    # 3. PDF Report
    save_advanced_pdf_report(results, run_dir, experiment_name, timestamp, config)

def run_generation_and_eval(model_name: str, prompts: List[str], num_images: int, dx_code: str, ham_loader: Any, batch_choice: str = "sd") -> List[Dict]:
    """Generate images and evaluate them."""
    
    if os.environ.get("DRY_RUN") == "1":
         logger.info(f"[DRY RUN] Would generate {num_images} images with {model_name}")
         return [{"model": model_name, "prompt": p, "image": None, "ssim": 0.0, "color_similarity": 0.0, "shape_similarity": 0.0, "reference_images": []} for p in prompts[:num_images]]

    if batch_choice == "sd":
        gen_func = lambda: generate_stable_diffusion_images(prompts, num_images, model_id="stable-diffusion-v1-5/stable-diffusion-v1-5")
    elif batch_choice == "sd3.5":
        gen_func = lambda: generate_sdxl_images(prompts, num_images)
    elif batch_choice == "qwen":
        gen_func = lambda: generate_qwen_images(prompts, num_images)
    else:
        gen_func = lambda: generate_stable_diffusion_images(prompts, num_images)

    # Track performance
    tracker = PerformanceTracker()
    tracker.start_tracking()
    
    try:
        gen_list = gen_func()
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        tracker.stop_tracking()
        return []
        
    tracker.stop_tracking()
    perf_metrics = tracker.get_metrics(num_images)

    # Evaluation loop
    results = []
    for item in gen_list:
        img = item.get("image")
        prompt = item.get("prompt")
        
        metrics = {'ssim': 0.0, 'color_similarity': 0.0, 'shape_similarity': 0.0}
        ref_imgs = []
        
        if img and ham_loader:
            try:
                metrics, ref_imgs = compare_with_ham10000(img, dx_code, ham_loader)
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
        
        item.update(metrics)
        item['reference_images'] = ref_imgs
        # Attach performance metrics to each item (shared for batch)
        item['performance'] = perf_metrics
        results.append(item)
        
    return results

def run_experiment_1(models: List[str]):
    """Exp 1: 3 Templates x 7 Diagnoses x 1 Image per model."""
    exp_name = "experiment_1"
    run_dir, images_dir = setup_run_directory(exp_name)
    results = []
    templates = ["template_1", "template_2", "template_3"]
    ham_loader = get_ham_loader()
    
    img_counter = 0
    
    for template_name in templates:
        try:
            template_text = load_template(template_name)
        except FileNotFoundError:
            continue
        
        for dx_code, dx_name in LESION_TYPES.items():
            prompt = template_text.replace("{diagnosis}", dx_name)
            
            for model in models:
                logger.info(f"Generating for {dx_code} with {model} ({template_name})")
                
                items = run_generation_and_eval(model, [prompt], 1, dx_code, ham_loader, batch_choice=model)

                for item in items:
                    img_obj = item.get("image")
                    refs = item.get("reference_images", [])
                    
                    saved_path, ref_saved_paths = save_image_and_refs(img_obj, run_dir, img_counter, refs)
                    img_counter += 1
                    
                    record = {
                        "prompt": prompt,
                        "model": model,
                        "model_id": item.get("model_id", model),
                        "diagnosis": dx_code,
                        "inferred_diagnosis": dx_code,
                        "template": template_name,
                        "image_path": saved_path,
                        "reference_image_paths": ref_saved_paths,
                        "ssim": item.get("ssim", 0),
                        "color_similarity": item.get("color_similarity", 0),
                        "shape_similarity": item.get("shape_similarity", 0),
                        "performance": item.get("performance", {})
                    }
                    results.append(record)
    
    config = {"experiment": exp_name, "models": models, "templates": templates}
    save_run_data(results, run_dir, exp_name, config)

def run_experiment_2(models: List[str]):
    """Exp 2: Pairwise Combinations (Sequential Generation)."""
    exp_name = "experiment_2"
    run_dir, images_dir = setup_run_directory(exp_name)
    results = []
    combinations = list(itertools.combinations(LESION_KEYS, 2))
    templates = ["template_1", "template_2", "template_3"]
    ham_loader = get_ham_loader()
    img_counter = 0
    
    for template_name in templates:
        try:
            template_text = load_template(template_name)
        except FileNotFoundError:
            continue
            
        for dx1, dx2 in combinations:
            dx1_name = LESION_TYPES[dx1]
            dx2_name = LESION_TYPES[dx2]
            
            # Construct two separate prompts
            prompt1 = template_text.replace("{diagnosis}", dx1_name)
            prompt2 = template_text.replace("{diagnosis}", dx2_name)
            
            # Use original prompt construction logic if template doesn't have {diagnosis} 
            # (e.g. if it was already replaced? No, we load fresh)
            
            prompts = [prompt1, prompt2]
            
            for model in models:
                logger.info(f"Generating sequential pair {dx1}, {dx2} with {model} ({template_name})")
                
                # We pass BOTH prompts to one generation call. 
                # This generates 2 images in one batch (or sequence).
                # We use dx1 as the primary label for the batch evaluation context, 
                # but results will have individual prompts.
                # Note: compare_with_ham10000 key will be dx1 for the first image and we need to handle the second.
                # run_generation_and_eval iterates results. 
                # BUT run_generation_and_eval takes ONE dx_code for evaluation.
                # This is a limitation. We need to handle evaluation separately or update run_generation_and_eval.
                # ACTUALLY: run_generation_and_eval takes `dx_code` and uses it for ALL images in batch.
                # This is wrong for mixed batches.
                # OPTION: Generate them separately? No, user wants "batch".
                # OPTION: Fix evaluation in run_generation_and_eval to handle list of dx_codes?
                # Simpler: Call run_generation_and_eval, then override the 'inferred_diagnosis' and re-evaluate if needed.
                # Or just assume dx1 for both? No.
                # Let's simple call it twice? "First a MEL image and than the second image of the batch is a BCC image".
                # If the pipeline allows batching different prompts, we get [Img1, Img2].
                # Img1 matches dx1. Img2 matches dx2.
                # I will modify the loop after generation to fix the evaluation.
                
                # To do this correctly without changing run_generation_and_eval signature too much:
                # We can allow dx_code to be a list?
                # Or we just accept that run_generation_and_eval will use dx1 for reference finding for BOTH.
                # That's bad for metrics.
                # Strategy: Generate, then manually correct metrics for the second image.
                
                # However, run_generation_and_eval does the generation AND evaluation.
                # If I pass dx_code=dx1, both get evaluated against dx1 references.
                # The second image (dx2) will have terrible metrics against dx1 references.
                
                # Better approach for clean code: Generate separately?
                # User says: "first the model to first generate image of one diagnosis and then an image of a different diagnosis... second image of the batch"
                # If I generate separately, they are in different "batches" (technically).
                # If I want them in the same "batch" for the model context (if that matters), I must pass them together.
                
                # I will pass them together to `run_generation_and_eval` with `dx_code=dx1`.
                # Then regarding the result items:
                # Item 0 is dx1. Item 1 is dx2.
                # I will Re-EVALUATE distinct items if necessary or just accept the limitation?
                # No, I should fix it.
                # I will effectively disable the internal eval for this call by passing None loader?
                # And do eval manually here? Yes.
                
                items = run_generation_and_eval(model, prompts, 2, dx_code=dx1, ham_loader=None, batch_choice=model)

                # Now manually evaluate correct diagnosis
                diagnosis_map = [dx1, dx2]
                
                for i, item in enumerate(items):
                     current_dx = diagnosis_map[i]
                     img_obj = item.get("image")
                     
                     # Manual Eval
                     metrics = {'ssim': 0.0, 'color_similarity': 0.0, 'shape_similarity': 0.0}
                     ref_imgs = []
                     if img_obj and ham_loader:
                         try:
                             metrics, ref_imgs = compare_with_ham10000(img_obj, current_dx, ham_loader)
                         except Exception as e:
                             logger.error(f"Manual eval failed: {e}")
                     
                     # Update item
                     item.update(metrics)
                     item['reference_images'] = ref_imgs
                     
                     saved_path, ref_saved_paths = save_image_and_refs(img_obj, run_dir, img_counter, ref_imgs)
                     img_counter += 1
                     
                     record = {
                            "prompt": item.get('prompt'),
                            "model": model,
                            "model_id": item.get("model_id", model),
                            "diagnosis": current_dx,
                            "inferred_diagnosis": current_dx,
                            "pair_context": f"{dx1} then {dx2}",
                            "template": template_name,
                            "image_path": saved_path,
                            "reference_image_paths": ref_saved_paths,
                            "ssim": item.get("ssim", 0),
                            "color_similarity": item.get("color_similarity", 0),
                            "shape_similarity": item.get("shape_similarity", 0),
                            "performance": item.get("performance", {})
                        }
                     results.append(record)
                 
    config = {"experiment": exp_name, "models": models}
    save_run_data(results, run_dir, exp_name, config)

def run_experiment_random_distribution(models: List[str], total_count_per_template: int, exp_number: str):
    """Exp 3/4: Random Split with Template Iteration."""
    exp_name = f"experiment_{exp_number}_random_{total_count_per_template}"
    run_dir, images_dir = setup_run_directory(exp_name)
    results = []
    templates = ["template_1", "template_2", "template_3"]
    ham_loader = get_ham_loader()
    img_counter = 0
    
    for template_name in templates:
        try:
            template_text = load_template(template_name)
        except FileNotFoundError:
            continue
            
        # Generate assignments for this template
        assignments = {dx: 0 for dx in LESION_KEYS}
        for _ in range(total_count_per_template):
            assignments[random.choice(LESION_KEYS)] += 1
            
        logger.info(f"Random assignments for {template_name}: {assignments}")
        
        for dx_code, count in assignments.items():
            if count == 0: continue
            
            dx_name = LESION_TYPES[dx_code]
            prompt = template_text.replace("{diagnosis}", dx_name)
            
            for model in models:
                logger.info(f"Generating {count} images for {dx_code} with {model} ({template_name})")
                items = run_generation_and_eval(model, [prompt]*count, count, dx_code, ham_loader, batch_choice=model)
                
                for item in items:
                     img_obj = item.get("image")
                     refs = item.get("reference_images", [])
                     saved_path, ref_saved_paths = save_image_and_refs(img_obj, run_dir, img_counter, refs)
                     img_counter += 1
                     
                     record = {
                            "prompt": prompt,
                            "model": model,
                            "model_id": item.get("model_id", model),
                            "diagnosis": dx_code,
                            "inferred_diagnosis": dx_code,
                            "template": template_name,
                            "split_count": count,
                            "image_path": saved_path,
                            "reference_image_paths": ref_saved_paths,
                            "ssim": item.get("ssim", 0),
                            "color_similarity": item.get("color_similarity", 0),
                            "shape_similarity": item.get("shape_similarity", 0),
                            "performance": item.get("performance", {})
                        }
                     results.append(record)
                 
    config = {"experiment": exp_name, "models": models, "total_count_per_template": total_count_per_template}
    save_run_data(results, run_dir, exp_name, config)



def main():
    parser = argparse.ArgumentParser(description="Run Skin Cancer Generation Experiments")
    parser.add_argument("--exp", type=str, choices=["1", "2", "3", "4", "all"], default="all", help="Experiment number to run")
    parser.add_argument("--models", type=str, default="sd", help="Comma-separated list of models (sd,sd3.5,qwen)")
    parser.add_argument("--validate", action="store_true", help="Run in fast validation mode (dry run logic/1 step)")
    
    args = parser.parse_args()
    
    models = args.models.split(",")
    
    if args.validate:
        logger.info("VALIDATION MODE ENABLED")
        os.environ["DRY_RUN"] = "1"
    
    try:
        if args.exp == "1" or args.exp == "all":
            run_experiment_1(models)
        if args.exp == "2" or args.exp == "all":
            run_experiment_2(models)
        if args.exp == "3" or args.exp == "all":
             # 10 images total per template (30 total)
             run_experiment_random_distribution(models, total_count_per_template=10, exp_number="3")
        if args.exp == "4" or args.exp == "all":
             # 100 images total per template (300 total)
             run_experiment_random_distribution(models, total_count_per_template=100, exp_number="4")
             
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
