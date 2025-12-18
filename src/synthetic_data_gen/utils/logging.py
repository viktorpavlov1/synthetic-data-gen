
import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import shutil
from fpdf import FPDF
import math

logger = logging.getLogger(__name__)

class ExperimentLogger:
    """Logs experiment details and results."""
    
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True, parents=True)
        
    def log_experiment(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        results: List[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]] = None,
        execution_log: Optional[str] = None
    ) -> str:
        """
        Log experiment results to JSON and CSV.
        
        Args:
            experiment_name: Name of the experiment (e.g., "generation_run")
            config: Configuration dictionary (model, seed, etc.)
            results: List of result dictionaries (one per image)
            metrics: Aggregate metrics for the run
            execution_log: String containing captured logs
            
        Returns:
            Path to the experiment directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{experiment_name}_{timestamp}"
        run_dir = self.logs_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Save logs if provided
        if execution_log:
            try:
                log_path = run_dir / "execution.log"
                with open(log_path, "w", encoding='utf-8') as f:
                    f.write(execution_log)
            except Exception as e:
                logger.error(f"Failed to save execution log: {e}")

        # Save full experiment data
        experiment_data = {
            "run_id": run_id,
            "timestamp": timestamp,
            "config": config,
            "results": [], # Will store stripped version (no image objects) and image paths
            "aggregate_metrics": metrics or {}
        }
        
        # Process results to save images and strip objects
        images_dir = run_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        processed_results = []
        for i, res in enumerate(results):
            # Shallow copy to avoid modifying original
            res_copy = res.copy()
            
            # Save main image
            img = res_copy.get('image')
            if img:
                img_filename = f"image_{i:03d}.png"
                img_path = images_dir / img_filename
                try:
                    img.save(img_path)
                    res_copy['image_path'] = str(img_path.relative_to(self.logs_dir.parent))
                except Exception as e:
                    logger.error(f"Failed to save image {i}: {e}")
                
                if 'image' in res_copy: del res_copy['image']

            # Save reference image (input ref)
            if 'reference_image' in res_copy and res_copy.get('reference_image'):
                 ref_img = res_copy.get('reference_image')
                 ref_filename = f"ref_input_{i:03d}.png"
                 ref_path = images_dir / ref_filename
                 try:
                    ref_img.save(ref_path)
                    res_copy['reference_image_path'] = str(ref_path.relative_to(self.logs_dir.parent))
                 except Exception as e:
                    logger.error(f"Failed to save ref image {i}: {e}")
                 del res_copy['reference_image']

            # Save comparison images (for metrics)
            if 'comparison_images' in res_copy and res_copy.get('comparison_images'):
                comp_imgs = res_copy.get('comparison_images')
                comp_paths = []
                for idx, c_img in enumerate(comp_imgs):
                    c_filename = f"comp_{i:03d}_{idx}.png"
                    c_path = images_dir / c_filename
                    try:
                        # Resize to thumbnail size if too big to save space? User said "smaller"
                        # But typically 600x450 is small enough.
                        c_img.save(c_path)
                        comp_paths.append(str(c_path.relative_to(self.logs_dir.parent)))
                    except Exception as e:
                        logger.error(f"Failed to save comp image {i}_{idx}: {e}")
                
                res_copy['comparison_image_paths'] = comp_paths
                del res_copy['comparison_images']
            
            processed_results.append(res_copy)
            
        experiment_data['results'] = processed_results
        
        # Save JSON log
        json_path = run_dir / "experiment_log.json"
        with open(json_path, 'w') as f:
            json.dump(experiment_data, f, indent=2)
            
        # Save CSV summary
        try:
            df = pd.DataFrame(processed_results)
            csv_path = run_dir / "results_summary.csv"
            df.to_csv(csv_path, index=False)
        except Exception as e:
            logger.warning(f"Could not save CSV summary: {e}")
            
        logger.info(f"Experiment logged to {run_dir}")
        return str(run_dir)

    def generate_report(self, run_dir: str) -> str:
        """Markdown report generation (Legacy - use PDF)."""
        # Kept for compatibility but PDF is preferred
        return str(Path(run_dir) / "report.md")

    def generate_pdf_report(self, run_dir: str) -> str:
        """
        Generate a detailed PDF report for the experiment.
        Includes images, reference thumbnails, metrics, and execution log.
        """
        run_path = Path(run_dir)
        json_path = run_path / "experiment_log.json"
        log_path = run_path / "execution.log"
        
        if not json_path.exists():
             raise FileNotFoundError(f"Experiment log not found at {json_path}")
             
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        project_root = self.logs_dir.parent
        
        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 15)
                self.cell(0, 10, 'Experiment Report', 0, 1, 'C')
                self.ln(5)
            
            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
        pdf = PDF()
        pdf.add_page()
        pdf.set_font('Arial', '', 12)
        
        # 1. Header Info
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(40, 10, f"Run ID:", 0)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"{data.get('run_id')}", 0, 1)
        
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(40, 10, f"Date:", 0)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"{data.get('timestamp')}", 0, 1)
        pdf.ln(5)
        
        # 2. Performance & Metrics
        metrics = data.get('aggregate_metrics', {})
        if metrics:
            pdf.set_fill_color(240, 240, 240)
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, "Performance & Evaluation Metrics", 0, 1, 'L', 1)
            pdf.ln(2)
            
            pdf.set_font('Arial', '', 11)
            col_width = 90
            row_height = 8
            
            for k, v in metrics.items():
                if isinstance(v, float):
                     val = f"{v:.4f}"
                else:
                     val = str(v)
                
                key_name = k.replace('_', ' ').title()
                if "Wh" in key_name: key_name = key_name.replace("Wh", "(Wh)")
                if "Watts" in key_name: key_name = key_name.replace("Watts", "(W)")
                
                pdf.cell(col_width, row_height, key_name, 1)
                pdf.cell(col_width, row_height, val, 1)
                pdf.ln(row_height)
            pdf.ln(10)
            
        # 3. Configuration
        pdf.set_fill_color(240, 240, 240)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Configuration", 0, 1, 'L', 1)
        pdf.ln(2)
        
        pdf.set_font('Courier', '', 10)
        config_str = json.dumps(data.get('config', {}), indent=2)
        pdf.multi_cell(0, 5, config_str)
        pdf.ln(10)
        
        # 4. Images
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Generated Images", 0, 1, 'L', 1)
        pdf.ln(5)
        
        results = data.get('results', [])
        for i, res in enumerate(results):
            # Check for page break needed
            if pdf.get_y() > 220: pdf.add_page()

            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, f"Image {i+1}", 0, 1)
            
            # Metrics
            img_metrics = res.get('eval_metrics', {})
            if img_metrics:
                pdf.set_font('Arial', '', 10)
                metric_text = ", ".join([f"{k}: {v:.3f}" for k, v in img_metrics.items() if isinstance(v, (int, float))])
                pdf.multi_cell(0, 5, f"Metrics: {metric_text}")
                pdf.ln(2)
                
            pdf.set_font('Arial', 'I', 10)
            prompt = res.get('prompt', 'N/A')
            pdf.multi_cell(0, 5, f"Prompt: {prompt}")
            pdf.ln(2)
            
            # Main Image
            img_path_rel = res.get('image_path')
            start_y = pdf.get_y()
            if img_path_rel:
                try:
                    full_path = project_root / img_path_rel
                    if full_path.exists():
                        # Main image approx 100mm wide
                        pdf.image(str(full_path), w=100, x=10)
                except Exception as e:
                    pdf.cell(0, 5, f"Err img: {e}", 0, 1)
            
            # Comparison Images (Thumbnails to the right or below)
            # Let's put them below for now or to the right if space
            # To the right: x=120
            comp_paths_rel = res.get('comparison_image_paths', [])
            if comp_paths_rel:
                pdf.set_xy(120, start_y)
                pdf.set_font('Arial', 'B', 8)
                pdf.cell(0, 5, "Reference Images (Comparison):", 0, 1)
                
                # Draw thumbnails grid 3x...
                thumb_w = 25
                x_start = 120
                curr_x = x_start
                curr_y = pdf.get_y()
                
                for idx, c_path in enumerate(comp_paths_rel[:6]): # Limit to 6 to save space
                    try:
                         full_c_path = project_root / c_path
                         if full_c_path.exists():
                             pdf.image(str(full_c_path), w=thumb_w, x=curr_x, y=curr_y)
                             curr_x += thumb_w + 2
                             if (idx + 1) % 3 == 0:
                                 curr_x = x_start
                                 curr_y += 20 # approx height of thumb
                    except Exception: 
                        pass
                
                # Move Y down below largest block
                pdf.set_y(max(pdf.get_y(), start_y + 80)) # Ensure space for main image height
            else:
                 pdf.ln(80) # Space for main image if no refs

            pdf.ln(5)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)

        # 5. Appendix: Logs
        if log_path.exists():
            pdf.add_page()
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, "Appendix: Execution Logs", 0, 1, 'L', 1)
            pdf.ln(5)
            
            pdf.set_font('Courier', '', 8)
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Filter basic info to keep short? Or just dump last 200 lines?
                    # Dump all for now but handle pagination
                    for line in lines:
                        # Clean ansi codes if any (simplified)
                        clean_line = line.strip()
                        if clean_line:
                            pdf.multi_cell(0, 4, clean_line)
            except Exception as e:
                pdf.cell(0, 5, f"Could not read logs: {e}", 0, 1)

        # Save PDF with timestamp
        timestamp = data.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
        pdf_filename = f"experiment_report_{timestamp}.pdf"
        pdf_path = run_path / pdf_filename
        
        try:
            pdf.output(str(pdf_path))
        except Exception as e:
            logger.error(f"Failed to save PDF: {e}")
            return ""
            
        return str(pdf_path)
