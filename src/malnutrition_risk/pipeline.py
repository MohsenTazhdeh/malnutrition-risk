from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline

def build_pipeline(cfg: DictConfig) -> Pipeline:

    # Instantiate feature_engineering & preprocessor and force conversion from ListConfig to standard Python lists
    feature_engineering = instantiate(cfg.feature_engineering, _convert_="all")
    preprocessor = instantiate(cfg.preprocessor, _convert_="all")

    # Sklearn strictly requires a list of tuples for the transformers parameter.
    # Because YAML only creates lists of lists, we cast the inner lists to tuples here.
    if hasattr(feature_engineering, "transformers"):
        feature_engineering.transformers = [tuple(step) for step in feature_engineering.transformers]
    if hasattr(preprocessor, "transformers"):
        preprocessor.transformers = [tuple(step) for step in preprocessor.transformers]

    # instantiate the classifier
    model = instantiate(cfg.model)

    pipeline = Pipeline(steps=[
        ('feature_engineering', feature_engineering),
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.set_output(transform="pandas")

    return pipeline




