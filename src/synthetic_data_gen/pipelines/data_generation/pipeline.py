"""Data generation pipeline definition."""

from kedro.pipeline import Pipeline, node
from .nodes import generate_images, create_prompts


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the data generation pipeline.
    
    Returns:
        Kedro Pipeline for image generation
    """
    return Pipeline(
        [
            node(
                func=create_prompts,
                inputs=["params:image_generation.lesion_types",
                       "params:image_generation.prompt_template",
                       "params:image_generation.num_images",
                       "params:ham10000.dataset_path",
                       "params:ham10000.use_ham10000_prompts"],
                outputs="prompts",
                name="create_prompts"
            ),
            node(
                func=generate_images,
                inputs=["params:image_generation.model",
                       "prompts",
                       "params:image_generation.num_images",
                       "params:image_generation.image_size",
                       "params:image_generation.batch_size",
                       "params:image_generation.seed"],
                outputs="generated_images",
                name="generate_images"
            )
        ]
    )

