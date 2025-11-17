"""Pipeline definitions for synthetic data generation and classification."""

from synthetic_data_gen.pipelines.data_generation.pipeline import create_pipeline as create_data_generation_pipeline
from synthetic_data_gen.pipelines.classification.pipeline import create_pipeline as create_classification_pipeline
from synthetic_data_gen.pipelines.training.pipeline import create_pipeline as create_training_pipeline

from kedro.pipeline import Pipeline


def register_pipelines() -> dict:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a Pipeline object.
    """
    data_generation_pipeline = create_data_generation_pipeline()
    classification_pipeline = create_classification_pipeline()
    training_pipeline = create_training_pipeline()
    
    return {
        "data_generation": data_generation_pipeline,
        "classification": classification_pipeline,
        "training": training_pipeline,
        "__default__": data_generation_pipeline + classification_pipeline,
    }
