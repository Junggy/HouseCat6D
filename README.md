<div align="center">
  <img src="./ImportedPhoto.734104285.343345.jpeg" alt="" width="400"/>
</div>

# HouseCat6D Toolbox
This repo provides some useful tools for using the [HouseCat6D](https://sites.google.com/view/housecat6d) dataset [CVPR 2024 Highlight]. 

```javascript 
conda create -n housecat python=3.8
conda activate housecat
pip install -r requirement.txt
```

Copy the folder `visualization` to the downloaded HouseCat6D folder.

### Object Pose Visualization

One can visualize both rendered mask and 3D bounding box on entire frame given scene name and dataset directory.
Make sure to put `obj_models_small_size_final` folder next to scene folders.

```javascript 
cd path/to/visualization

python vis_obj.py (path/to/dataset) (scene_name)
```
For example if the script is located into dataset folder and want to visualize scene01, simply run 
```javascript 
python vis_obj.py ./ scene01
```

### Grasp Visualization

We support two ways of visualization. 3D will give an instant pyrender visualization, and 2D will save the image with the grasps rendered on the image plane. You can choose whether you want to downsample the visualized grasps.

```javascript 
cd path/to/visualization

python vis_grasp.py --split train --scene 1 --dimentional 3D --ds
```

### Additional Baseline Reference

We provide the reimplementation of [VI-Net](https://github.com/JiehongLin/VI-Net) to show how to use HouseCat6D. 

#### Installation

```javascript
conda create -n vi-net python=3.9
conda activate vi-net
```

We tried with PyTorch 1.9 and CUDA 11.1.

```
cd VI-Net/lib/pointnet2/
pip install .
cd ../sphericalmap_utils/
pip install .
pip install gorilla-core==0.2.6.0
pip install opencv-python
pip install gpustat==1.0.0
pip install --upgrade protobuf
pip install scipy
```

Note that the latest `gorilla-core` would fail. Modify `path/to/HouseCat6D` in the `VI-Net/config/housecat.yaml`.

#### Rotation branch:

Training with RGB-D 

```javascript
cd VI-Net
python train_housecat.py --gpus 0 --dataset housecat --mode r --config config/housecat.yaml
```

Training with RGB+P-D

```javascript
python train_pol.py --gpus 0 --dataset housecat --mode r --config config/housecat.yaml
```

#### Translation branch:

Training with RGB-D 

```javascript
cd VI-Net
python train_housecat.py --gpus 0 --dataset housecat --mode ts --config config/housecat.yaml
```

Training with RGB+P-D

```javascript
cd VI-Net
python train_pol.py --gpus 0 --dataset housecat --mode ts --config config/housecat.yaml
```

### Note

HouseCat6D is released under CC BY 4.0.
