# ai_grand_challenge_kr
<img src="https://github.com/yehengchen/grand_ai_challenge/blob/master/graph/ai_grand.png" width="60%" height="60%">

## Requirements
1. Python 3.5
2. OpenCV
3. PyTorch 0.4

## Detection Example
<img src="https://github.com/yehengchen/Object-Detection-and-Tracking/blob/master/OneStage/yolo/yolo_img/TownCentreXVID_output_ss.gif" width="50%" height="50%">
<img src="https://github.com/yehengchen/ai_grand_challenge_kr/blob/master/t1_video/t1_video_00001/t1_video_00001_00028.jpg" width="50%" height="50%"><img src="https://github.com/yehengchen/ai_grand_challenge_kr/blob/master/img/ai_grand_cg.png" width="20%" height="20%">

### On single or multiple images

Clone, and `cd` into the repo directory. The first thing you need to do is to get the weights file
This time around, for v3, authors has supplied a weightsfile only for COCO [here](https://pjreddie.com/media/files/yolov3.weights), and place 

the weights file into your repo directory. Or, you could just type (if you're on Linux)

```
wget https://pjreddie.com/media/files/yolov3.weights 
python main.py
```


`--images` flag defines the directory to load images from, or a single image file (it will figure it out), and `--det` is the directory
to save images to. Other setting such as batch size (using `--bs` flag) , object threshold confidence can be tweaked with flags that can be looked up with. 

```
python main.py
```
