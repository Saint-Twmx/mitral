import logging
import os
import sys
import time
from typing import NoReturn
import nrrd
import numpy as np
import torch
from segmentation.postprocess import postprocess
import SimpleITK as sitk
curdir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, curdir)
from segmentation.nnunet_sub.preprocessing.preprocessing import resize_segmentation_ori
from segmentation.nnunet_sub.utilities.one_hot_encoding import to_one_hot
from segmentation.nnunet_sub.training.model_restore import load_model_and_checkpoint_files
from segmentation.nnunet_sub.inference.segmentation_export import save_segmentation_nifti
from measure.tool.readdicom import get_info_with_sitk_nrrd, handle_save_array

def nnUNet_pred(
    iso_data,
    model_path: str = None,
    folds=None,
    segs_from_prev_stage=None,
    do_tta=True,
    use_gaussian=True,
    mixed_precision=True,
    all_in_gpu=True,
    step_size=0.5,
    checkpoint_name="model_final_checkpoint",
    trt_engine=None,
    do_postprocessing=True,
    ps=[96,  160,  160]
):
    timea = time.time()
    torch.cuda.empty_cache()

    trainer, params = load_model_and_checkpoint_files(
        model_path,
        folds,
        mixed_precision=mixed_precision,
        checkpoint_name=checkpoint_name,
    )
    timeb = time.time()
    print(f"load_model time :{timeb - timea}")


    trainer.data_aug_params["mirror_axes"] = (2,)
    if trt_engine is None:
        trainer.patch_size = np.array(ps)
    else:
        trainer.patch_size = trt_engine.input_shape[2:]
        trainer.network.trt_engine = trt_engine

    d, _, dct = trainer.preprocess_patient(iso_data,new_c=True)
    classes = list(range(1, trainer.num_classes))

    timec = time.time()
    print(f"trainer.preprocess time :{timec - timeb}")

    if segs_from_prev_stage is not None:
        # check to see if shapes match
        seg_prev = segs_from_prev_stage.transpose(trainer.plans["transpose_forward"])
        seg_reshaped = resize_segmentation_ori(seg_prev, d.shape[1:], order=1)
        seg_reshaped = to_one_hot(seg_reshaped, classes)
        d = np.vstack((d, seg_reshaped)).astype(np.float32)

    # preallocate the output arrays
    # same dtype as the return value in predict_preprocessed_data_return_seg_and_softmax (saves time)
    all_softmax_outputs = np.zeros(
        (len(params), trainer.num_classes, *d.shape[1:]), dtype=np.float16
    )
    all_seg_outputs = np.zeros((len(params), *d.shape[1:]), dtype=int)

    timed = time.time()
    print(f"all_softmax time :{timed - timec}")

    for i, p in enumerate(params):
        trainer.load_checkpoint_ram(p, False)
        res = trainer.predict_preprocessed_data_return_seg_and_softmax(
            d,
            do_mirroring=do_tta,
            mirror_axes=trainer.data_aug_params["mirror_axes"],
            use_sliding_window=True,
            step_size=step_size,
            use_gaussian=use_gaussian,
            all_in_gpu=all_in_gpu,
            verbose=False,
            mixed_precision=mixed_precision,
        )
        if len(params) > 1:
            # otherwise we dont need this and we can save ourselves the time it takes to copy that
            all_softmax_outputs[i] = res[1]
        all_seg_outputs[i] = res[0]

    timee = time.time()
    print(f"segmentation costs {timee - timed} seconds")

    if hasattr(trainer, "regions_class_order"):
        region_class_order = trainer.regions_class_order
    else:
        region_class_order = None
    assert region_class_order is None, (
        "predict_cases_fastest can only work with regular softmax predictions "
        "and is therefore unable to handle trainer classes with region_class_order"
    )

    if len(params) > 1:
        softmax_mean = np.mean(all_softmax_outputs, 0)
        seg = softmax_mean.argmax(0)
    else:
        seg = all_seg_outputs[0]

    transpose_forward = trainer.plans.get("transpose_forward")
    if transpose_forward is not None:
        transpose_backward = trainer.plans.get("transpose_backward")
        seg = seg.transpose([i for i in transpose_backward])

    results = save_segmentation_nifti(seg, None, dct, 0, None)
    print("分割 END ", type(results), f"耗时:{time.time() - timea}")
    return torch.from_numpy(results)

def main( ) -> NoReturn:
    if os.path.exists(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output",
            "output.nrrd"
        )
    ):
        print(
            "The segmentation has been completed, skip this step, "
            "inference code details please see: https://github.com/MIC-DKFZ/MedNeXt")
        return


    ori_raw, infor = get_info_with_sitk_nrrd(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input",
            "TAVI_00401_00014_00001_0000.nii.gz"
        )
    )

    #
    # handle_save_array(
    #     os.path.join(
    #         os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input",
    #         "input.nii.gz"
    #     ),
    #     ori_raw[40:420,70:460,30:], infor
    # )


    fullres3d_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model",)

    iso_pred = nnUNet_pred(
        ori_raw,
        model_path=fullres3d_model_path,
        folds=0,
        use_gaussian=True,
        do_tta=True,
        step_size=1,
        checkpoint_name="model_final_checkpoint",
        trt_engine=None,
        ps=[80, 160, 160]
    )

    iso_pred = postprocess(iso_pred)
    for i in range(1, 3):
        t_points = torch.nonzero(iso_pred == i)
        print(i, t_points.shape)

    handle_save_array(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output",
            "output.nrrd"
        ),
        iso_pred, infor
    )



if __name__ == "__main__":
    torch.cuda.set_device(1)
    test = main()