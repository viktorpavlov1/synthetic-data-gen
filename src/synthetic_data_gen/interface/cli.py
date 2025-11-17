"""Command-line interface for the synthetic data generation pipeline."""

import click
import logging
from pathlib import Path
import sys
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Synthetic Skin Lesion Generation and Classification Pipeline."""
    pass


@cli.command()
@click.option(
    '--generation-model',
    type=click.Choice(['stable_diffusion', 'qwen', 'flux'], case_sensitive=False),
    default='stable_diffusion',
    help='Image generation model to use'
)
@click.option(
    '--num-images',
    type=int,
    default=10,
    help='Number of images to generate'
)
@click.option(
    '--action',
    type=click.Choice(['classify', 'retrain'], case_sensitive=False),
    default='classify',
    help='Action to perform: classify or retrain'
)
@click.option(
    '--classification-model',
    type=click.Choice(['vgg16', 'inception_v3', 'resnet50'], case_sensitive=False),
    default='resnet50',
    help='Classification model to use'
)
@click.option(
    '--image-size',
    type=int,
    default=512,
    help='Size of generated images'
)
@click.option(
    '--batch-size',
    type=int,
    default=4,
    help='Batch size for image generation'
)
@click.option(
    '--num-epochs',
    type=int,
    default=10,
    help='Number of training epochs (for retrain action)'
)
@click.option(
    '--learning-rate',
    type=float,
    default=0.001,
    help='Learning rate for training (for retrain action)'
)
@click.option(
    '--output-dir',
    type=click.Path(),
    default='data/07_model_output',
    help='Output directory for results'
)
def run(
    generation_model: str,
    num_images: int,
    action: str,
    classification_model: str,
    image_size: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    output_dir: str
):
    """
    Run the synthetic data generation and classification pipeline.
    
    Examples:
    
    \b
    # Generate 20 images with Stable Diffusion and classify them
    python -m synthetic_data_gen.interface.cli run \\
        --generation-model stable_diffusion \\
        --num-images 20 \\
        --action classify \\
        --classification-model resnet50
    
    \b
    # Generate images with Flux and retrain ResNet50
    python -m synthetic_data_gen.interface.cli run \\
        --generation-model flux \\
        --num-images 50 \\
        --action retrain \\
        --classification-model resnet50 \\
        --num-epochs 15
    """
    logger.info("Starting pipeline execution...")
    logger.info(f"Generation model: {generation_model}")
    logger.info(f"Number of images: {num_images}")
    logger.info(f"Action: {action}")
    logger.info(f"Classification model: {classification_model}")
    
    # Get project path
    project_path = Path(__file__).parent.parent.parent.parent
    
    try:
        # Bootstrap and configure Kedro project
        metadata = bootstrap_project(project_path)
        configure_project(metadata.package_name)
        
        # Prepare parameter overrides
        params_override = {
            "image_generation": {
                "model": generation_model.lower(),
                "num_images": num_images,
                "image_size": image_size,
                "batch_size": batch_size,
                "seed": 42
            },
            "classification": {
                "model": classification_model.lower(),
                "batch_size": 32,
                "num_classes": 2
            },
            "training": {
                "model": classification_model.lower(),
                "batch_size": 32,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "use_synthetic_data": True,
                "synthetic_data_ratio": 0.3
            }
        }
        
        # Create session with parameter overrides
        with KedroSession.create(project_path=project_path, extra_params=params_override) as session:
            # Run data generation pipeline
            logger.info("Running data generation pipeline...")
            session.run(pipeline_name="data_generation")
            
            if action == "classify":
                # Run classification pipeline
                logger.info("Running classification pipeline...")
                session.run(pipeline_name="classification")
                
                # Load results from catalog
                context = session.load_context()
                
                try:
                    classification_results = context.catalog.load("classification_results")
                    if classification_results is not None:
                        logger.info(f"\nClassification Results:")
                        logger.info(f"Total images classified: {len(classification_results)}")
                        if "confidence" in classification_results.columns:
                            logger.info(f"Average confidence: {classification_results['confidence'].mean():.4f}")
                        if "predicted_class_name" in classification_results.columns:
                            logger.info(f"\nClass distribution:")
                            print(classification_results["predicted_class_name"].value_counts())
                except Exception as e:
                    logger.warning(f"Could not load classification results: {e}")
            
            elif action == "retrain":
                # Run training pipeline
                logger.info("Running training pipeline...")
                session.run(pipeline_name="training")
                
                # Load results from catalog
                context = session.load_context()
                
                try:
                    evaluation_metrics = context.catalog.load("evaluation_metrics")
                    if evaluation_metrics:
                        logger.info(f"\nTraining Results:")
                        logger.info(f"Test Accuracy: {evaluation_metrics.get('test_accuracy', 0):.2f}%")
                        logger.info(f"Test Precision: {evaluation_metrics.get('test_precision', 0):.2f}%")
                        logger.info(f"Test Recall: {evaluation_metrics.get('test_recall', 0):.2f}%")
                        logger.info(f"Test F1: {evaluation_metrics.get('test_f1', 0):.2f}%")
                except Exception as e:
                    logger.warning(f"Could not load evaluation metrics: {e}")
            
            logger.info("\nPipeline execution completed successfully!")
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
def list_models():
    """List all saved retrained models."""
    from synthetic_data_gen.utils.model_storage import ModelStorage
    
    storage = ModelStorage()
    models = storage.list_models()
    
    if not models:
        click.echo("No saved models found.")
        return
    
    click.echo(f"\nFound {len(models)} saved model(s):\n")
    
    for model_info in models:
        click.echo(f"Model: {model_info.get('name', 'Unknown')}")
        click.echo(f"  Path: {model_info.get('path', 'Unknown')}")
        if 'timestamp' in model_info:
            click.echo(f"  Timestamp: {model_info['timestamp']}")
        if 'metrics' in model_info:
            metrics = model_info['metrics']
            click.echo(f"  Metrics:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    click.echo(f"    {key}: {value:.4f}")
        click.echo()


if __name__ == "__main__":
    cli()

