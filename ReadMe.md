# What is it

Fine tuning a YOLOv9 model.  

Originally a homework assignment to students to get an understanding of the fine tuning process.  (The outline of the code was originally taken from https://learnopencv.com/fine-tuning-yolov9/.)

Now it's a set of code to do the same using the UTA HPC cluster, using technologies like Slurm, etc.

# First time
````conda create --name yolo_fine_tune -f environment.yml````

# Every time
````
conda activate yolo_fine_tune
python yolov9_fine_tune.py

# training images are read from SkyFusion-YOLOv9/train (see DAYA_YML_PATH)
#   the code will download these
# weights are saved/loaded from runs/detect/train3/weights/best.pt (see WT_PATH)
#   one set of wts w/ 3 epochs of training were added.  can unzip
# test images are read from data/test (see TE_IMG_BASE_PATH)
# results are written to results (see RESULT_IMG_BASE_PATH)
````

# When done
````conda deactivate````

# Going further
````conda env export --no-builds > environment.yml````  
Check in your code