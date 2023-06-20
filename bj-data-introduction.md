# Data Introduction

The Beijing data is a dataset collected in Beijing cabs in November 2015, including 1018312 trajectories. We obtained the corresponding road network data from OpenStreetMap and preprocessed the trajectory data to get the Beijing trajectory dataset matched to the road network, and we believed that this dataset could promote the development of urban trajectory mining tasks.

[Download](https://pan.baidu.com/s/1TbqhtImm_dWQZ1-9-1XsIQ?pwd=1231)

The statistical information of the data is as follows:

| Dataset     | Beijing              |
| ----------- | -------------------- |
| Time span   | 2015.11.1~2015.11.30 |
| #Trajectory | 1018312              |
| #Usr        | 1677                 |
| #road/geo   | 40306                |
| #edge/rel   | 101023               |

The directory structure and data description are as follows:

- `bj_roadmap_edge/` 
  - `bj_roadmap_edge.geo`: the geo file which stores the road segment information of the road network.
    - `geo_id,type,coordinates,highway,lanes,tunnel,bridge,roundabout,oneway,length,maxspeed,u,v`
  - `bj_roadmap_edge.rel`: the rel file which stores the adjacent information between road segments.
    - `rel_id,type,origin_id,destination_id`
  - The format definition follows the [LibCity library](https://bigscity-libcity-docs.readthedocs.io/en/latest/user_guide/data/atomic_files.html).
- `traj_bj_11.csv` is a semicolon split csv file, each line represents data for one trajectory. Specifically, the meaning of each column is as follows:
  - `id`,  the unique id of the trajectory, which is not consecutive due to data processing.
  - `path`, the road segment ID list, each ID represent a `geo_id` in `bj_roadmap_edge.geo`.
  - `tlist`, the corresponding timestamp (UTC) list of each road ID in `path`.
  - `length`, the routing length of the trajectory, accumulated according to the road length provided by the `geo` file.
  - `speed`, the average speed of the trajectory.
  - `duration`, the total time from the start to the end of the trajectory.
  - `hop`, the number of IDs contained in the `path`, i.e. the number of hops.
  - `usr_id`,  the ID of the driver of the trajectory.
  - `traj_id`, the ID of different trajectories of the same driver, which is not consecutive due to data processing.
  - `vflag`, passenger marker, 0 means empty, 1 means carrying passengers.
  - `start_time`, the start time of the trajectory.

Please ensure that this data is **used for research purposes only**. 

If you use this data, please apply the two papers below, thank you:

```
@inproceedings{START,
  title={Self-supervised Trajectory Representation Learning with Temporal Regularities and Travel Semantics},
  author={Jiawei Jiang and Dayan Pan and Houxing Ren and Xiaohan Jiang and Chao Li and Jingyuan Wang},
  booktitle={2023 IEEE 39th international conference on data engineering (ICDE)},
  year={2023},
  organization={IEEE}
}

@inproceedings{libcity,
  author       = {Jingyuan Wang and
                  Jiawei Jiang and
                  Wenjun Jiang and
                  Chao Li and
                  Wayne Xin Zhao},
  title        = {LibCity: An Open Library for Traffic Prediction},
  booktitle    = {{SIGSPATIAL/GIS}},
  pages        = {145--148},
  publisher    = {{ACM}},
  year         = {2021}
}
```
