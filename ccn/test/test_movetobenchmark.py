import unittest
import torch
from torch.testing import assert_close

from ccn.benchmark.moveto import MoveToBenchmark
from ccn.active_inference.diff_agent import DAgent, agent_modes
from ccn.util import get_data_path
from scene_environments.util import get_data_path as scene_data_path


class BenchmarkTestCase(unittest.TestCase):
    def test_single_run(self):

        object_names = [
            "002_master_chef_can",
            "003_cracker_box",
            "004_sugar_box",
            "006_mustard_bottle",
            "005_tomato_soup_can",
        ]

        # Configuration parameters
        data_dir = get_data_path()
        load_dir = (
            data_dir / "input/models/SpatialTransformerCCN/checkpoint-5000"
        )
        mask_load_dir = data_dir / "input/models/ObjectMasks/checkpoint-5000"

        agent = DAgent(
            object_names,
            model_dir=load_dir,
            mask_model_dir=mask_load_dir,
            particle_filter=True,
            teleport=True,
            debug=False,
            mode=agent_modes.EFE,
        )

        sdp = scene_data_path() / "ccn-evaluation-pickles"
        bm = MoveToBenchmark(sdp, agent)
        logs = bm.run_one(0)


if __name__ == "__main__":
    # Set to float64 to avoid rounding errors for testing
    torch.set_default_dtype(torch.float64)
    unittest.main()
