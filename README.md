# DeepFusionMOT

![https://img.shields.io/badge/%20State%20of%20the%20Art%20-Multiple%20Object%20Tracking%20on%20KITTI%20Dataset-Lime]

This is the offical implementation of paper "[DeepFusionMOT: A 3D Multi-Object Tracking Framework Based on Camera-LiDAR Fusion with Deep Association](https://arxiv.org/abs/2202.12100) "

Contact: [wangxiyang@cqu.edu.cn](mailto:zhouxy@cs.utexas.edu). Any questions or discussion are welcome!

![329ee4972b77c1931b3f33f5e5e94ce3_raw 00_00_00-00_00_30](https://user-images.githubusercontent.com/71493146/155647887-68f42724-1f7e-417f-bf80-1e05d1c0d536.gif)

In the recent literature, on the one hand, many 3D multi-object tracking (MOT) works have focused on tracking accuracy and neglected computation speed, commonly by designing rather complex cost functions and feature extractors. On the other hand, some methods have focused too much on computation speed at the expense of tracking accuracy. In view of these issues, this paper proposes a robust and fast camera-LiDAR fusion-based MOT method that achieves a good trade-off between accuracy and speed. Relying on the characteristics of camera and LiDAR sensors, an effective deep association mechanism is designed and embedded in the proposed MOT method. This association mechanism realizes tracking of an object in a 2D domain when the object is far away and only detected by the camera, and updating of the 2D trajectory with 3D information obtained when the object appears in the LiDAR field of view to achieve a smooth fusion of 2D and 3D trajectories. Extensive experiments based on the KITTI dataset indicate that our proposed method presents obvious advantages over the state-of-the-art MOT methods in terms of both tracking accuracy and processing speed.

![框架](https://user-images.githubusercontent.com/71493146/155648073-e0d9b364-f869-421e-9280-937651d805e9.jpg)


### Video examples on benchmarks dataset

![my_0001 00_00_00-00_00_30](https://user-images.githubusercontent.com/71493146/155648648-3951f69e-7f93-4c73-ad31-4ab1bccf438e.gif)


## Dependencies

* scikit-image
* scikit-learn
* scipy
* pillow
* pandas
* pillow
* numba
* numpy
* opencv-python
* filterpy
* torchvision

## Getting Started

#### *1. Clone the github repository.*

```
git clone https://github.com/wangxiyang2022/DeepFusionMOT
```

#### *2. Dataset preparation*

 Please download the official KITTI [object tracking dataset](http://www.cvlibs.net/datasets/kitti/eval_tracking.php).

The final dataset organization should be like this:

```
DeepFusionMOT
├── datasets
    ├── kitti
        ├── train
		│   ├──calib_train
		│   ├──image_02_train
        ├── test
		    ├──calib_test
		    ├──image_02_test
```

#### *3. Install dependency*

```
cd your_path/DeepFusionMOT
pip install -r requirements.txt
```



#### *4. 3D Object Detections & 2D Object Detections*

Thanks to these researchers for making their code public, in this repository, for convenience, we provide the 3D detections of PointRCNN and 2D detections of RRC on the KITTI MOT dataset for car. Of course you can also use the results of other detectors, but you need to put the files in the following file directories.

```
DeepFusionMOT
├── datasets
    ├── kitti
        ├── train
        │   ├──2D_rrc_Car_train  
        │   ├──3D_pointrcnn_train 
		│   ├──calib_train
		│   ├──image_02_train
        ├── test
            ├──2D_rrc_Car_test
            ├──3D_pointrcnn_test
		    ├──calib_test
		    ├──image_02_test
```



#### *5. Run demo*

```
python main.py
```

#### *6. Visualization*

If you want to visualize the tracking results, first you need to uncomment lines 198  in the main.py. Then run `python main.py`. You can see the results in `results/train/image` . If you want to make a video of the tracking results, you  need to run `visualization/img_to_video.py`.  Of course, you need to modify the corresponding file directory.

#### *7. KITTI MOT Evaluation*

If you want to evaluate the tracking results using the evaluation tool on the KITTI website, you will need to go https://github.com/JonathonLuiten/TrackEval to download the evaluation code and follow the appropriate steps to set.

Using  3D detections of PointRCNN  and 2D detections of RRC,  the following results will be obtained.

|                      |  HOAT  |  MOTA  |
| :------------------: | :----: | :----: |
| **Training dataset** | 77.45% | 87.28% |
| **Testing dataset**  | 75.46% | 84.63% |



### Acknowledgement

A portion  code is borrowed from [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT) and [Deepsort](https://github.com/nwojke/deep_sort).  Many thanks to their wonderful work!
