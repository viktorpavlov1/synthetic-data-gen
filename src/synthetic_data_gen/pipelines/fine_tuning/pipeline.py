"""Fine-tuning pipeline definition."""

from kedro.pipeline import Pipeline, node
from .nodes import (
    prepare_training_dataset,
    fine_tune_stable_diffusion
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the fine-tuning pipeline.
    
    Returns:
        Kedro Pipeline for fine-tuning image generation models
    """
    return Pipeline(
        [
            node(
                func=prepare_training_dataset,
                inputs=["dataset_images",
                       "params:fine_tuning.output_size",
                       "params:fine_tuning.max_images"],
                outputs="prepared_dataset",
                name="prepare_training_dataset"
            ),
            node(
                func=fine_tune_stable_diffusion,
                inputs=["prepared_dataset",
                       "params:fine_tuning.base_model",
                       "params:fine_tuning.output_dir",
                       "params:fine_tuning.num_epochs",
                       "params:fine_tuning.learning_rate",
                       "params:fine_tuning.batch_size"],
                outputs="fine_tuned_model_path",
                name="fine_tune_stable_diffusion"
            )
        ]
    )

