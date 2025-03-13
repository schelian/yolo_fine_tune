# What is it

Fine tuning a YOLOv9 model.  

Originally a homework assignment to students to get an understanding of the fine tuning process.  (The outline of the code was originally taken from https://learnopencv.com/fine-tuning-yolov9/.)

Now it's a set of code to do the same using the UTA HPC cluster, using technologies like Slurm, etc.

# First time
````conda create --name yolo_fine_tune -f environment.yml````

# Every time

## Test naive model
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
Compare results to what's in results/naive_model.  There should be no ship or airplane detected

## Train/test model for 3 epochs

### Training
````
in yolov9_fine_tune.py
set GOT_DATA to False
set DO_TRAIN to True
````

### Testing w/o training
````
# unzip weights
cd runs/detect/train3/weights/
unzip best.pt.zip

Then go back to Test naive model
Results will be written to results

Compare (e.g., via diff) yolo_fine_tune/results and yolo_fine_tune/results/train3/
E.g., 
  cd results
  diff result_ship.json train3/result_ship.json
  There will be some issues w/ numerical precision starting w/ the 3rd digit.
E.g., .717... v .718...
  new
  < "[\n  {\n    \"name\": \"ship\",\n    \"class\": 1,\n    \"confidence\": 0.71787

  old
  > "[\n  {\n    \"name\": \"ship\",\n    \"class\": 1,\n    \"confidence\": 0.71837
````

# UTA HPC
1. Use Ivanti VPN if you are off campus
1. ssh ````<username>@hpcr8o2rnp.uta.edu````
1. ````sbatch FineTune.slurm````

    * You’ll see something like this  
  ````Submitted batch job 14580````
    * When it’s done, you’ll see two new files slurm-14580.err and slurm-14580.out  
    * Those are the output of stderr and stdout.  Examine those and other files for the status of your job, results, etc.

# When done
````conda deactivate````

# Going further
````conda env export --no-builds > environment.yml````  
Check in your code

# Notes
If you need an interactive session (shell prompt) on a compute node, use command ````srun --partition=NAME --pty /bin/bash````, where partition NAME is an available partition.

Partition names and availability can be found with the "sinfo" command. As of this email (Jan 21, 2025), we have NORMAL, LONG, SHORT, LOW and CHECK partitions. Our normal and long partition has the best hardware... NORMAL parition will run job for 8 days. LONG will run for 16 days. We also have SHORT and LOW partition... CHECK will allocate the next available compute node but is limited to 30 minutes of runtime.  
* NB: the partitions are lowercase but they are written in uppercase to show that they are not the usual English words.

See https://go.uta.edu/hpcinfo and "HPC Users Group" on MS Teams for more tips.