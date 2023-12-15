# HouseCat6D Toolbox
This repo provides some useful tools for visualizing the [HouseCat6D](https://sites.google.com/view/housecat6d) dataset. 

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

### Note

The dataset is released under CC BY 4.0.

