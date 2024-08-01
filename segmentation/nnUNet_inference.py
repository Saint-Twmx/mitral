import os
from typing import NoReturn
import torch
from segmentation.postprocess import postprocess
from measure.tool.readdicom import get_info_with_sitk_nrrd, handle_save_array

def main( ) -> NoReturn:
    iso_pred, infor = get_info_with_sitk_nrrd(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output",
            "output.nrrd"
        )
    )

    iso_pred = postprocess(torch.from_numpy(iso_pred))
    handle_save_array(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output",
            "output.nrrd"
        ),
        iso_pred, infor
    )
