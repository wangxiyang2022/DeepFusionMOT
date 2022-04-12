# -*-coding:utf-8-*
# author: wangxy
from __future__ import print_function
import os, numpy as np, time, cv2, torch
from os import listdir
from os.path import join
from file_operation.file import load_list_from_folder, mkdir_if_inexistence, fileparts
from detection.detection import Detection_2D, Detection_3D_only, Detection_3D_Fusion
from tracking.tracker import Tracker
from datasets.datafusion import datafusion2Dand3D
from datasets.coordinate_transformation import convert_3dbox_to_8corner, convert_x1y1x2y2_to_tlwh
from visualization.visualization_3d import show_image_with_boxes
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


class DeepFusion(object):
    def __init__(self, max_age, min_hits):
        '''
        :param max_age:  The maximum frames in which an object disappears.
        :param min_hits: The minimum frames in which an object becomes a trajectory in succession.
        '''
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracker = Tracker(max_age,min_hits)
        self.reorder = [3, 4, 5, 6, 2, 1, 0]
        self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
        self.frame_count = 0

    def update(self,detection_3D_fusion,detection_2D_only,detection_3D_only,detection_3Dto2D_only,
               additional_info, calib_file):

        dets_3d_fusion = np.array(detection_3D_fusion['dets_3d_fusion'])
        dets_3d_fusion_info = np.array(detection_3D_fusion['dets_3d_fusion_info'])
        dets_3d_only = np.array(detection_3D_only['dets_3d_only'])
        dets_3d_only_info = np.array(detection_3D_only['dets_3d_only_info'])

        if len(dets_3d_fusion) == 0:
            dets_3d_fusion = dets_3d_fusion
        else:
            dets_3d_fusion = dets_3d_fusion[:,self.reorder]  # convert [h,w,l,x,y,z,rot_y] to [x,y,z,rot_y，l,w,h]
        if len(dets_3d_only) == 0:
            dets_3d_only = dets_3d_only
        else:
            dets_3d_only = dets_3d_only[:, self.reorder]

        detection_3D_fusion = [Detection_3D_Fusion(det_fusion, dets_3d_fusion_info[i]) for i, det_fusion in enumerate(dets_3d_fusion)]
        detection_3D_only = [Detection_3D_only(det_only, dets_3d_only_info[i]) for i, det_only in enumerate(dets_3d_only)]
        detection_2D_only = [Detection_2D(det_fusion) for i, det_fusion in enumerate(detection_2D_only)]

        self.tracker.predict_2d()
        self.tracker.predict_3d()
        self.tracker.update(detection_3D_fusion, detection_3D_only, detection_3Dto2D_only, detection_2D_only, calib_file, iou_threshold=0.5)

        self.frame_count += 1
        outputs = []
        for track in self.tracker.tracks_3d:
            if track.is_confirmed():
                bbox = np.array(track.pose[self.reorder_back])
                outputs.append(np.concatenate(([track.track_id_3d], bbox, track.additional_info)).reshape(1, -1))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):  # Convert the coordinate format of the bbox box from center x, y, w, h to upper left x, upper left y, w, h
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _tlwh_to_xyxy(self, bbox_tlwh):
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), 0)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), 0)
        return x1, y1, x2, y2

    def _tlwh_to_x1y1x2y2(self, bbox_tlwh):
        x, y, w, h = bbox_tlwh
        x1 = x
        x2 = x + w
        y1 = y
        y2 = y + h
        return x1, y1, x2, y2


if __name__ == '__main__':
    # Define the file name
    data_root = 'datasets/kitti/train'
    detections_name_3D = '3D_pointrcnn_Car_val'
    detections_name_2D = '2D_rrc_Car_val'

    # Define the file path
    calib_root = os.path.join(data_root, 'calib_train')
    dataset_dir = os.path.join(data_root,'image_02_train')
    detections_root_3D = os.path.join(data_root, detections_name_3D)
    detections_root_2D = os.path.join(data_root, detections_name_2D)

    # Define the file path of results.
    save_root = 'results/train'   # The root directory where the result is saved
    txt_path_0 = os.path.join(save_root, 'data'); mkdir_if_inexistence(txt_path_0)
    image_path_0 = os.path.join(save_root, 'image'); mkdir_if_inexistence(image_path_0)
    # Open file to save in list.
    det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
    calib_files = os.listdir(calib_root)
    detections_files_3D = os.listdir(detections_root_3D)
    detections_files_2D = os.listdir(detections_root_2D)
    image_files = os.listdir(dataset_dir)
    detection_file_list_3D, num_seq_3D = load_list_from_folder(detections_files_3D, detections_root_3D)
    detection_file_list_2D, num_seq_2D = load_list_from_folder(detections_files_2D, detections_root_2D)
    image_file_list, _ = load_list_from_folder(image_files, dataset_dir)

    total_time, total_frames, i = 0.0, 0, 0  # Tracker runtime, total frames and Serial number of the dataset
    tracker = DeepFusion(max_age=25, min_hits=3)  # Tracker initialization

    # Iterate through each data set
    for seq_file_3D, image_filename in zip(detection_file_list_3D, image_files):
        print('--------------Start processing the {} dataset--------------'.format(image_filename))
        total_image = 0  # Record the total frames in this dataset
        seq_file_2D = detection_file_list_2D[i]
        seq_name, datasets_name, _ = fileparts(seq_file_3D)
        txt_path = txt_path_0 + "\\" + image_filename + '.txt'
        image_path = image_path_0 + '\\' + image_filename; mkdir_if_inexistence(image_path)

        calib_file = [calib_file for calib_file in calib_files if calib_file==seq_name ]
        calib_file_seq = os.path.join(calib_root, ''.join(calib_file))
        image_dir = os.path.join(dataset_dir, image_filename)
        image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        seq_dets_3D = np.loadtxt(seq_file_3D, delimiter=',')  # load 3D detections, N x 15
        seq_dets_2D = np.loadtxt(seq_file_2D, delimiter=',')  # load 2D detections, N x 6

        min_frame, max_frame = int(seq_dets_3D[:, 0].min()), len(image_filenames)

        for frame, img0_path in zip(range(min_frame, max_frame + 1), image_filenames):
            img_0 = cv2.imread(img0_path)
            _, img0_name, _ = fileparts(img0_path)
            dets_3D_camera = seq_dets_3D[seq_dets_3D[:, 0] == frame, 7:14]  # 3D bounding box(h,w,l,x,y,z,theta)
            dets_8corners = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets_3D_camera]

            ori_array = seq_dets_3D[seq_dets_3D[:, 0] == frame, -1].reshape((-1, 1))
            other_array = seq_dets_3D[seq_dets_3D[:, 0] == frame, 1:7]
            additional_info = np.concatenate((ori_array, other_array), axis=1)

            dets_3Dto2D_image = seq_dets_3D[seq_dets_3D[:, 0] == frame, 2:6]
            dets_2D = seq_dets_2D[seq_dets_2D[:, 0] == frame, 1:5]   # 2D bounding box(x1,y1,x2,y2)

            # Data Fusion(3D and 2D detections)
            detection_2D_fusion, detection_3Dto2D_fusion, detection_3D_fusion, detection_2D_only, detection_3Dto2D_only, detection_3D_only = \
                datafusion2Dand3D(dets_3D_camera, dets_2D, dets_3Dto2D_image, additional_info)

            detection_2D_only_tlwh = np.array([convert_x1y1x2y2_to_tlwh(i) for i in detection_2D_only]) # (x1,y1,x2,y2) to (x,y,center_x,center_y)

            start_time = time.time()
            trackers = tracker.update(detection_3D_fusion, detection_2D_only_tlwh, detection_3D_only, detection_3Dto2D_only,
                                      additional_info, calib_file_seq)
            cycle_time = time.time() - start_time
            total_time += cycle_time

            # Outputs
            total_frames += 1 # Total frames for all datasets
            total_image += 1 # Total frames for a dataset
            if total_image % 50 == 0:
                print("Now start processing the {} image of the {} dataset".format(total_image, image_filename))

            if len(trackers) > 0:
                for d in trackers:
                    bbox3d = d.flatten()
                    bbox3d_tmp = bbox3d[1:8]  # 3D bounding box(h,w,l,x,y,z,theta)
                    id_tmp = int(bbox3d[0])
                    ori_tmp = bbox3d[8]
                    type_tmp = det_id2str[bbox3d[9]]
                    bbox2d_tmp_trk = bbox3d[10:14]
                    conf_tmp = bbox3d[14]
                    color = compute_color_for_id(id_tmp)
                    label = f'{id_tmp} {"car"}'
                    image_save_path = os.path.join(image_path, '%06d.jpg' % (int(img0_name)))
                    with open(txt_path, 'a') as f:
                        str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (frame, id_tmp,type_tmp, ori_tmp,bbox2d_tmp_trk[0],
                                bbox2d_tmp_trk[1],bbox2d_tmp_trk[2],bbox2d_tmp_trk[3],bbox3d_tmp[0], bbox3d_tmp[1],bbox3d_tmp[2], bbox3d_tmp[3],
                                bbox3d_tmp[4], bbox3d_tmp[5],bbox3d_tmp[6],conf_tmp)
                        f.write(str_to_srite)
                        # show_image_with_boxes(img_0, bbox3d_tmp, image_path, color, img0_name, label, calib_file_seq,line_thickness=1)
        i += 1
        print('--------------The time it takes to process all datasets are {}s --------------'.format(total_time))
    print('--------------FPS = {} --------------'.format(total_frames/total_time))
