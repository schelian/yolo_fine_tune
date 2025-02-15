# What is it

Fine tuning a YOLOv9 model.  

Originally a homework assignment to students to get an understanding of the fine tuning process.  (The outline of the code was originally taken from https://learnopencv.com/fine-tuning-yolov9/.)

Now it's a set of code to do the same using the UTA HPC cluster, using technologies like Slurm, etc.

# First time
````conda create --name yolo_fine_tune -f environment.yml````

# Every time
````conda activate yolo_fine_tune````

# When done
````conda deactivate````

# Going further
````conda env export --no-builds > environment.yml````  
Check in your code