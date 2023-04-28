import argparse
import os, tqdm
import shutil
import time
from os.path import join

import cv2
import numpy as np

from datasets.coordinate_transformation import convert_x1y1x2y2_to_tlwh
from tracking.DeepFusionMOT import DeepFusionMOT
from utils.config import Config
from evaluation.KITTI.evaluation_HOTA.scripts.run_kitti import eval_kitti
from utils.combine_trk_cat import combine_category_result
from datasets.data_fusion import data_fusion
from utils.save_results import save_results


def tracking(cfg):
    spilt = cfg.spilt
    seq_list = cfg.tracking_seqs
    total_time, total_frames = 0, 0

    for category in cfg.cat_list:
        for seq_id in tqdm.trange(len(seq_list)):
            # ----------------------------- Initialize tracker -------------------------
            tracker = DeepFusionMOT(cfg, category)
            seq_name = str(seq_id).zfill(4)

            dets_path_3d = os.path.join(cfg.dets_path_3d, cfg.detector_3d, spilt, category) + "/" + str(seq_id).zfill(4) + '.txt'
            dets_path_2d = os.path.join(cfg.dets_path_2d, cfg.detector_2d, spilt, category) + "/" + str(seq_id).zfill(4) + '.txt'
            image_02_path = os.path.join(cfg.dataset_path, spilt, 'image_02') + "/" + str(seq_id).zfill(4)
            image_filenames = [join(image_02_path, x) for x in os.listdir(image_02_path)]
            dets_3d = np.loadtxt(dets_path_3d, delimiter=',')  # load 3D detections, N x 15
            dets_2d = np.loadtxt(dets_path_2d, delimiter=',')

            #----------------- Remove 3D detections of low confidence -------------------
            # det_scores = seq_dets_3d[:, 6]
            # mask = det_scores > cfg.input_score
            # seq_dets_3d = seq_dets_3d[mask]

            #----------------- Remove 2D detections of low confidence -------------------
            # if dets_2d.any():
            #     det_scores_2d = dets_2d[:, 5]
            #     mask_2d = det_scores_2d > 0.4
            #     dets_2d = dets_2d[mask_2d]

            min_frame, max_frame = 0, len(image_filenames)

            for frame in tqdm.trange(max_frame):
                img0_path = image_filenames[frame]
                img_0 = cv2.imread(img0_path)
                dets_3d_camera = dets_3d[dets_3d[:, 0] == frame, 7:14]  # 3D bounding box(h,w,l,x,y,z,theta)

                ori_array = dets_3d[dets_3d[:, 0] == frame, -1].reshape((-1, 1))
                other_array = dets_3d[dets_3d[:, 0] == frame, 1:7]
                additional_info = np.concatenate((ori_array, other_array), axis=1)

                dets_3dto2d_image = dets_3d[dets_3d[:, 0] == frame, 2:6]

                if len(dets_2d):
                    dets_2d_frame = dets_2d[dets_2d[:, 0] == frame, 1:5]  # 2D bounding box(x1,y1,x2,y2)
                else:
                    dets_2d_frame = []
                # -------------------- The fusion of 3D detections and 2D detections -------------
                dets_3d_fusion, dets_3d_only, dets_2d_only = \
                    data_fusion(dets_3d_camera, dets_2d_frame, dets_3dto2d_image, additional_info)

                dets_2d_only_tlwh = np.array([convert_x1y1x2y2_to_tlwh(i) for i in dets_2d_only])

                start_time = time.time()
                trackers = tracker.update(dets_3d_fusion,
                                          dets_2d_only_tlwh,
                                          dets_3d_only,
                                          cfg,
                                          frame,
                                          seq_id
                                          )
                cycle_time = time.time() - start_time
                total_time += cycle_time
                total_frames += 1
                save_results(trackers, cfg, seq_name, frame, category, img_0)

    print('--------------The total time is {}s --------------'.format(total_time))
    print('--------------FPS = {} --------------'.format(total_frames / total_time))


if __name__ == '__main__':
    file_path = 'results'
    try:
        shutil.rmtree(file_path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    parser = argparse.ArgumentParser(description='DeepFusionMOT')
    parser.add_argument('--cfg', type=str, default='./config/kitti.yaml', help='data')
    args = parser.parse_args()
    cfg, _ = Config(args.cfg)

    tracking(cfg)
    combine_category_result(cfg)

    # print("--------------Starting Evaluation-------------")
    results = eval_kitti()