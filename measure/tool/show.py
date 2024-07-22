import numpy as np
import os
from measure.tool.readdicom import handle_save_array

def get_some_nrrd(measure, ori_pred, head):
    test = np.zeros(ori_pred.shape)

    for i in measure["mitral_hull_point"]:
        j = [int(i[0]),int(i[1]),int(i[2]),]
        test[j[0]-1:j[0]+2,j[1]-1:j[1]+2,j[2]-1:j[2]+2,] = 1

    for i in measure["mitral_cc_real_points"]:
        j = [int(i[0]), int(i[1]), int(i[2]), ]
        test[j[0] - 2:j[0] + 2, j[1] - 2:j[1] + 2, j[2] - 2:j[2] + 2, ] =2

    for i in measure["mitral_cc_points"]:
        j = [int(i[0]), int(i[1]), int(i[2]), ]
        test[j[0] - 2:j[0] + 2, j[1] - 2:j[1] + 2, j[2] - 2:j[2] + 2, ] =3

    for i in measure["mitral_ap_points"]:
        j = [int(i[0]), int(i[1]), int(i[2]), ]
        test[j[0] - 2:j[0] + 2, j[1] - 2:j[1] + 2, j[2] - 2:j[2] + 2, ] =4

    for i in measure["mitral_tt_points"]:
        j = [int(i[0]), int(i[1]), int(i[2]), ]
        test[j[0] - 2:j[0] + 2, j[1] - 2:j[1] + 2, j[2] - 2:j[2] + 2, ] =5

    nnn = 6
    for ta in ["A1_points","A2_points","A3_points","P1_points","P2_points","P3_points"]:
        for i in measure[ta]:
            j = [int(i[0]), int(i[1]), int(i[2]), ]
            test[j[0] - 1:j[0] + 2, j[1] - 1:j[1] + 2, j[2] - 1:j[2] + 2, ] = nnn
        nnn+=1

    save_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "output",
        "check.seg.nrrd"
    )


    handle_save_array(save_path,test,head)

