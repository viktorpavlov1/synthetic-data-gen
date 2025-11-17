"""Training pipeline definition."""

from kedro.pipeline import Pipeline, node
from .nodes import (
    prepare_training_data,
    train_model,
    evaluate_model,
    save_trained_model
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the training pipeline.
    
    Returns:
        Kedro Pipeline for model training
    """
    return Pipeline(
        [
            node(
                func=prepare_training_data,
                inputs=["generated_images",
                       "params:training.use_synthetic_data",
                       "params:training.synthetic_data_ratio"],
                outputs=["train_dataset", "val_dataset", "test_dataset"],
                name="prepare_training_data"
            ),
            node(
                func=train_model,
                inputs=["params:training.model",
                       "train_dataset",
                       "val_dataset",
                       "params:classification.num_classes",
                       "params:training.batch_size",
                       "params:training.learning_rate",
                       "params:training.num_epochs"],
                outputs=["trained_model", "training_history"],
                name="train_model"
            ),
            node(
                func=evaluate_model,
                inputs=["trained_model",
                       "test_dataset",
                       "params:training.batch_size"],
                outputs="evaluation_metrics",
                name="evaluate_model"
            ),
            node(
                func=save_trained_model,
                inputs=["trained_model",
                       "params:training.model",
                       "evaluation_metrics",
                       "training_history",
                       "params:training",
                       "params:model_storage.base_path"],
                outputs="saved_model_path",
                name="save_trained_model"
            )
        ]
    )

