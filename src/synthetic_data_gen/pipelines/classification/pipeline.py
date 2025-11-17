"""Classification pipeline definition."""

from kedro.pipeline import Pipeline, node
from .nodes import classify_images, aggregate_classification_results


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the classification pipeline.
    
    Returns:
        Kedro Pipeline for image classification
    """
    return Pipeline(
        [
            node(
                func=classify_images,
                inputs=["generated_images",
                       "params:classification.model",
                       "params:classification.batch_size",
                       "params:classification.num_classes"],
                outputs="classification_results",
                name="classify_images"
            ),
            node(
                func=aggregate_classification_results,
                inputs="classification_results",
                outputs="classification_summary",
                name="aggregate_classification_results"
            )
        ]
    )

