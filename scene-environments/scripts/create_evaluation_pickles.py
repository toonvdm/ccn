import logging

from scene_environments import SceneEnvironment
from tqdm import tqdm
from pathlib import Path

from scene_environments.util import get_data_path
import os

# os.environ["DISPLAY"] = "localhost:11.0"

os.environ["PYOPENGL_PLATFORM"] = "egl"

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    store_dir = get_data_path()
    store_dir /= "ccn-evaluation-pickles-v4"
    store_dir.mkdir(exist_ok=True, parents=True)

    objects = [
        "002_master_chef_can",
        "003_cracker_box",
        "004_sugar_box",
        "005_tomato_soup_can",
        "006_mustard_bottle",
    ]

    logger.info(
        f"Creating evaluation pickles and storing them at {str(store_dir)}"
    )

    for n_objects in range(1, 6):
        for target_object in objects:
            store_path = store_dir / f"{target_object}-{n_objects}"
            store_path.mkdir(exist_ok=True, parents=True)
            seed_idx = 2000 + n_objects * 100
            for seed in tqdm(range(seed_idx, seed_idx + 100)):
                if not (store_path / f"{seed}.pkl").exists():
                    env = SceneEnvironment(
                        objects,
                        n_objects=n_objects,
                        seed=seed,
                        render_table=True,
                        eval=True,
                        resample_objects=True,
                        random_table_color=True,
                        resample_table_color=True,
                        target_object=target_object,
                    )
                    env.store_pkl(store_path / f"{seed}.pkl")
