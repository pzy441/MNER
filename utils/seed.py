import random

import numpy as np
import torch


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_seed(seed):
    try:
        import tensorflow as tf
        tf.random.set_random_seed(seed)
    except Exception as e:
        print("Set seed failed,details are ", e)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        print("Set seed failed,details are ", e)
        pass
    import numpy as np
    np.random.seed(seed)
    import random as python_random
    python_random.seed(seed)
    # cuda env
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
