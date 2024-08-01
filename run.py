from segmentation.nnUNet_inference import main as infer_main
from measure.main import main as mea_main
import numpy as np
import torch


if __name__ == "__main__":
    infer_main()
    mea_main()