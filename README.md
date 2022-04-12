# DeepFusionMOT

This is the offical implementation of paper "[DeepFusionMOT: A 3D Multi-Object Tracking Framework Based on Camera-LiDAR Fusion with Deep Association](https://arxiv.org/abs/2202.12100) "

![HOTA-FPS](https://github.com/wangxiyang2022/DeepFusionMOT/raw/master/assets/HOTA_FPS.jpg)

Contact: [wangxiyang@cqu.edu.cn](mailto:zhouxy@cs.utexas.edu). Any questions or discussion are welcome!

## Abstract

In the recent literature, on the one hand, many 3D multi-object tracking (MOT) works have focused on tracking accuracy and neglected computation speed, commonly by designing rather complex cost functions and feature extractors. On the other hand, some methods have focused too much on computation speed at the expense of tracking accuracy. In view of these issues, this paper proposes a robust and fast camera-LiDAR fusion-based MOT method that achieves a good trade-off between accuracy and speed. Relying on the characteristics of camera and LiDAR sensors, an effective deep association mechanism is designed and embedded in the proposed MOT method. This association mechanism realizes tracking of an object in a 2D domain when the object is far away and only detected by the camera, and updating of the 2D trajectory with 3D information obtained when the object appears in the LiDAR field of view to achieve a smooth fusion of 2D and 3D trajectories. Extensive experiments based on the KITTI dataset indicate that our proposed method presents obvious advantages over the state-of-the-art MOT methods in terms of both tracking accuracy and processing speed.

![comparison](https://github.com/wangxiyang2022/DeepFusionMOT/raw/master/assets/comparison.jpg)

## DeepFusionMOT
![Framework](https://github.com/wangxiyang2022/DeepFusionMOT/raw/master/assets/Framework.jpg)


### Video examples on benchmarks dataset

![Video examples](https://github.com/wangxiyang2022/DeepFusionMOT/raw/master/assets/Video_examples.gif)


## Dependencies
* Windows >= 8
* scikit-image
* scikit-learn
* scipy
* pillow
* pandas
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

- **For 3D detections**

| Frame | Type |      2D BBOX (x1, y1, x2, y2)       | Score  |          3D BBOX (h, w, l, x, y, z, rot_y)          |  Alpha  |
| :---: | :--: | :---------------------------------: | :----: | :-------------------------------------------------: | :-----: |
|   0   |  2   | 298.3125,165.1800,458.2292,293.4391 | 8.2981 | 1.9605,1.8137,4.7549,-4.5720,1.8435,13.5308,-2.1125 | -1.7867 |

- **For 2D detections**

| Frame |          2D BBOX (x1, y1, x2, y2)           |  Score   |
| :---: | :-----------------------------------------: | :------: |
|   0   | 296.021000,160.173000,452.297000,288.372000 | 0.529230 |

The format definition can be found in the object development toolkit here: https://github.com/JonathonLuiten/TrackEval/blob/master/docs/KITTI-format.txt

#### *5. Run demo*

```
python main.py
```

#### *6. Visualization*

If you want to visualize the tracking results, first you need to uncomment lines 198  in the main.py. Then run `python main.py`. You can see the results in `results/train/image` . If you want to make a video of the tracking results, you  need to run `visualization/img_to_video.py`.  Of course, you need to modify the corresponding file directory.

#### *7. KITTI MOT Evaluation*

If you want to evaluate the tracking results using the evaluation tool on the KITTI website, you will need to go https://github.com/JonathonLuiten/TrackEval to download the evaluation code and follow the appropriate steps to set.

Using  3D detections of PointRCNN  and 2D detections of RRC,  the following results will be obtained.

|                      | HOAT( **↑)** | **DetA( **↑)**** | **AssA**(**↑)** | IDSW（↓） | MOTP(**↑)** | MOTA(**↑)** | FPS（↑） |
| :------------------: | :----------: | :--------------: | :-------------: | :-------: | :---------: | :---------: | :------: |
| **Training dataset** |    77.45%    |      75.35%      |     79.85%      |    83     |   86.60%    |   87.28%    |   104    |
| **Testing dataset**  |    75.46%    |      71.54%      |     80.05%      |    84     |   85.02%    |   84.63%    |   110    |

### Acknowledgement

A portion  code is borrowed from [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT) and [Deepsort](https://github.com/nwojke/deep_sort), and the visualization code from [3D-Detection-Tracking-Viewer](https://github.com/hailanyi/3D-Detection-Tracking-Viewer).  Many thanks to their wonderful work!
