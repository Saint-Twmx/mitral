from measure.tool.readdicom import get_info_with_sitk_nrrd
from measure.mitral_centerline import mit_centerline
from measure.mitral_bestplane import mit_bestplane_new
from measure.mitral_cc_ap import mit_cc_ap
from measure.mitral_annulus import mit_annulus_perimeter_area
from measure.mitrial_analysis import numerical_calculation
from measure.mitral_tt import mit_tt
from measure.mitral_leaflet import mit_leaflets_length
import numpy as np
import os
from measure.tool.show import get_some_nrrd
from measure.post_processing_measure import post_processing_measure
def main():
    ori_pred, head = get_info_with_sitk_nrrd(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output",
            "output.nrrd"
        )
    )
    if not isinstance(ori_pred, np.ndarray):
        ori_pred = np.array(ori_pred)

    measure = dict()

    centerline = mit_centerline(ori_pred, simple=True)

    threeD_plane, best_plane = mit_bestplane_new(ori_pred, centerline, measure)

    mit_annulus_perimeter_area(ori_pred, head, threeD_plane, best_plane, measure)

    mit_cc_ap(ori_pred, head, measure) # cc ap 只能是types = 2

    mit_tt(ori_pred, head, best_plane, measure)  # cc ap  types 可为 1 可为 2

    mit_leaflets_length(ori_pred, head, best_plane, measure)

    numerical_calculation(measure, ori_pred, head)  # 指标 数值计算

    get_some_nrrd(measure, ori_pred,head)

    post_processing_measure(head,measure)






if __name__ == "__main__":

    paths = r"/mitral/demo2"
    import os
    import pandas as pd
    data_excel = pd.DataFrame(columns=[])
    for files in sorted(os.listdir(paths)):
        files = r"4_6_2.seg.nrrd"
        print('-------------------------------------------',files)
        ori_pred, head = get_info_with_sitk_nrrd(os.path.join(paths, files))

        measure = main(ori_pred, head, name = files.split(".")[0], debug=True)

        L = len(data_excel)
        data_excel.loc[L, "mitral_cc"] = measure["mitral_cc"]
        data_excel.loc[L, "mitral_cc_proj"] = measure["mitral_cc_proj"]
        data_excel.loc[L, "mitral_cc_real"] = measure["mitral_cc_real"]
        data_excel.loc[L, "mitral_cc_real_proj"] = measure["mitral_cc_real_proj"]
        data_excel.loc[L, "mitral_ap"] = measure["mitral_ap"]
        data_excel.loc[L, "mitral_ap_proj"] = measure["mitral_ap_proj"]
        data_excel.loc[L, "area"] = sum([m["proj_area"] for m in measure["orifice_proj"]])
        data_excel.loc[L, "mitral_perimeter"] = measure["mitral_perimeter"]
        data_excel.loc[L, "mitral_proj_perimeter"] = measure["mitral_proj_perimeter"]
        data_excel.loc[L, "mitral_area"] = measure["mitral_area"]
        data_excel.loc[L, "mitral_proj_area"] = measure["mitral_proj_area"]
        data_excel.loc[L, "mitral_hight"] = measure["mitral_hight"]-1

    data_excel.to_excel(r"/mitral/demo2_json/TJ.xlsx")

    # 4-4      4-6
    # 3-10     3-1
    # 5-3      5-13
    # 14-1     14-2
    # 40-1     40-5
    # 41-1     41-5
    # 52-3     52-10