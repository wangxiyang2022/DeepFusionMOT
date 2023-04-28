import os

from utils.file_operation.file import mkdir_if_inexistence
from utils.visualization.visualization_3d import show_image_with_boxes_3d
from utils.visualization.visualization_2d import show_image_with_boxes_2d

def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def save_results(trackers, cfg, seq_name, frame, category, image):
    save_path = os.path.join(cfg.save_path, category, "data"); mkdir_if_inexistence(save_path)
    save_name = os.path.join(save_path, seq_name + '.txt')
    f = open(save_name, 'a')

    save_image_dir = os.path.join(cfg.save_path, category, "image", seq_name); mkdir_if_inexistence(save_image_dir)

    # -----------------下面的代码是为了用AB3DMOT的3D测评输出的------------
    save_path = os.path.dirname(save_path)
    save_trk_dir = os.path.join(save_path, 'trk_withid', seq_name);  mkdir_if_inexistence(save_trk_dir)
    save_trk_file = os.path.join(save_trk_dir, '%06d.txt' % frame); save_trk_file = open(save_trk_file, 'w')

    if len(trackers) > 0:
        for d in trackers:
            bbox3d = d.flatten()
            bbox3d_tmp = bbox3d[1:8]  # 3D bounding box(h,w,l,x,y,z,theta)
            id_tmp = int(bbox3d[0])
            ori_tmp = bbox3d[8]
            type_tmp = category
            bbox2d_tmp_trk = bbox3d[10:14]
            conf_tmp = bbox3d[14]
            color = compute_color_for_id(id_tmp)
            label = f'{id_tmp} {"car"}'
            # with open(save_name, 'a') as f:
            str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % \
                           (frame, id_tmp, type_tmp,
                            ori_tmp,
                            bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3],
                            bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3],
                            bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[6], conf_tmp)
            f.write(str_to_srite)
            img_id = str(frame).zfill(6)
            # show_image_with_boxes_3d(img_0, bbox3d_tmp, image_path, color, img0_name, label, calib_file_seq, line_thickness=1)
            # show_image_with_boxes_2d(bbox3d, image, save_image_dir, color, img_id, label, line_thickness=2)
            # save in detection format with track ID, can be used for dection evaluation and tracking visualization

            # -----------------下面的代码是为了用AB3DMOT的3D测评输出的------------
            str_to_srite_3d = '%s -1 -1 %f %f %f %f %f %f %f %f %f %f %f %f %f %d\n' % (type_tmp, ori_tmp,
                                                                                     bbox2d_tmp_trk[0],
                                                                                     bbox2d_tmp_trk[1],
                                                                                     bbox2d_tmp_trk[2],
                                                                                     bbox2d_tmp_trk[3],
                                                                                     bbox3d_tmp[0], bbox3d_tmp[1],
                                                                                     bbox3d_tmp[2], bbox3d_tmp[3],
                                                                                     bbox3d_tmp[4], bbox3d_tmp[5],
                                                                                     bbox3d_tmp[6], conf_tmp, id_tmp)
            save_trk_file.write(str_to_srite_3d)
        save_trk_file.close()