import torch
import numpy as np
import json
import math
import os
def transf(head, data: torch.tensor):
    scale = torch.tensor(head["spacing"])
    shift = torch.tensor(head["origin"][::-1])
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    data = data.cpu()
    data = data * scale + shift
    return np.array(data[:, [2, 1, 0]]).tolist()

def post_processing_measure(
    head,
    measure_result: dict,
):
    # 朴实无华的处理每个key
    from measure.measure_template import json as mj

    parameters = ['left_ventricular_volume', 'left_ventricular_myocardial_volume','papillary_muscle_volume',
                  'anterior_papillary_muscle_to_mitral_valve_annular_distance', 'posterior_papillary_muscle_to_mitral_valve_annular_distance',
                  'interatrial_septum_annluls_angle', 'non_planarity', 'leaflet_to_annulus_ratio', 'coaptation_index',
                  'AHCWR', 'aortic_valve_annulus_and_mitral_valve_annulus_angle']
    for p in parameters:
        mj["mitralCalculation"]["parameters"].append(
            {p:measure_result.get(p,None)}
        )

    mj["annularPlane"]["markups"][0]["controlPoints"] = transf(head, measure_result["mitral_hull_point"])
    for i in mj["annularPlane"]["markups"][0]["measurements"]:
        if i.get("name") == "length":
            i["value"] = float(measure_result["mitral_perimeter"])
        elif i.get("name") == "area":
            i["value"] = float(measure_result["mitral_area"])
    for i in mj["annularPlane"]["markups"][0]["measurements"]:
        if i.get("name") == "length":
            mj["annularPlane"]["markups"][0]["measurements"].append({
                            "name": "length_diameter",
                            "value": i["value"] / math.pi,
                        })
        elif i.get("name") == "area":
            mj["annularPlane"]["markups"][0]["measurements"].append({
                "name": "area_diameter",
                "value": math.sqrt(i["value"] / math.pi) * 2,
            })

    mj["annularPlaneProj"]["markups"][0]["controlPoints"] = transf(head, measure_result["hull_point_3d"])
    for i in mj["annularPlaneProj"]["markups"][0]["measurements"]:
        if i.get("name") == "length":
            i["value"] = float(measure_result["mitral_proj_perimeter"])
        elif i.get("name") == "area":
            i["value"] = float(measure_result["mitral_proj_area"])
    for i in mj["annularPlaneProj"]["markups"][0]["measurements"]:
        if i.get("name") == "length":
            mj["annularPlaneProj"]["markups"][0]["measurements"].append({
                            "name": "length_diameter",
                            "value": i["value"] / math.pi,
                        })
        elif i.get("name") == "area":
            mj["annularPlaneProj"]["markups"][0]["measurements"].append({
                "name": "area_diameter",
                "value": math.sqrt(i["value"] / math.pi) * 2,
            })

    mj["annularHeight"]["markups"][0]["controlPoints"] = transf(head, measure_result["mitral_annulus_hight_points"])
    mj["annularHeight"]["markups"][0]["measurements"][0]["value"] = float(measure_result["mitral_annulus_hight"])


    for j in ["mitral_cc","mitral_cc_proj","mitral_cc_real","mitral_cc_real_proj","mitral_ap","mitral_ap_proj","mitral_tt"]:
        jj = {"mitral_cc":"cc","mitral_cc_proj":"ccProj","mitral_cc_real":"ccReal",
              "mitral_cc_real_proj":"ccRealProj","mitral_ap":"ap","mitral_ap_proj":"apProj",
              "mitral_tt":"tt"}.get(j)
        mj[jj]["markups"][0]["controlPoints"] = transf(head, measure_result[f"{j}_points"])
        mj[jj]["markups"][0]["measurements"][0]["value"] = float(measure_result[j])

    for j in ["A1","A2","A3","P1","P2","P3"]:
        jj = {"A1":"a1","A2":"a2","A3":"a3","P1":"p1","P2":"p2","P3":"p3"}.get(j)
        mj[jj]["markups"][0]["controlPoints"] = transf(head, measure_result[f"{j}_points"])
        for k in mj[jj]["markups"][0]["measurements"]:
            if k.get("name") == "line_lenght":
                k["value"] = float(measure_result[f"{j}_points_line_dis"])
            elif k.get("name") == "curve_lenght":
                k["value"] = float(measure_result[f"{j}_points_curve_dis"])


    save_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output",
            "measurement.json"
        )
    with open(save_path, 'w') as file:
        json.dump(mj, file)