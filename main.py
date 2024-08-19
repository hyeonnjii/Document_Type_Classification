import json

import hydra
from omegaconf import OmegaConf, DictConfig

from src.pipelines.pipeline import train, predict, tune


@hydra.main(config_path="configs/", config_name="resnet.yaml")
def main(
    config: DictConfig,
) -> None:
    if config.mode == "train":
        return train(config)
    elif config.mode == "predict":
        return predict(config)
    elif config.mode == "tune":
        return tune(config)
    else:
        raise ValueError(f"Invalid execution mode: {config.mode}")


if __name__ == "__main__":
    main()
