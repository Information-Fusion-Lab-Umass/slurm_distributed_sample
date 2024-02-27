import yaml
import os
import torch
import numpy as np
import random
import argparse
import json
import engines.lrs3_separation as lrs3_engine
import distributed as distributed_util

# lock all random seed to make the experiment replicable
seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

ROOT_PATH = "/project/pi_mfiterau_umass_edu/test"


if __name__ == "__main__":
    """
    main.py is only the entrance to the pipeline
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True)
    args = parser.parse_args()

    distributed_util.init_distributed_mode_torchrun(args)
    distributed_util.init_signal_handler()
    torch.distributed.barrier()

    exp_name = args.exp_name

    # create the directory for the output file
    if args.local_rank == 0:
        if not os.path.exists(os.path.join(ROOT_PATH, "results", exp_name)):
            os.makedirs(os.path.join(ROOT_PATH, "results", exp_name))

    # path to save output files, like losses, scores, figures etc
    report_path = os.path.join(ROOT_PATH, "results", exp_name)

    p = lrs3_engine.DistributedAVLITSeparation(cfg, report_path, args)  # any engine inherited from DistributedAbs.Distributed 
    p.train()

    test_result_path = p.test()

    if args.local_rank == 0:
        test_result = json.load(open(test_result_path, "r"))
        print("test loss: ", test_result["loss"])
        print("test performance:", test_result["performance"])

