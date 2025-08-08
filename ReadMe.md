# What is it

Fine tuning a YOLOv9 model.  

Originally a homework assignment to students to get an understanding of the fine tuning process.  (The outline of the code was originally taken from https://learnopencv.com/fine-tuning-yolov9/.)

Now it's a set of code to do the same using the UTA GPU cluster, and the UTA HPC cluster w/o GPUs (for this using technologies like Slurm, etc.).

# First time
``pyth``conda env create --name yolo_fine_tune -f environment.yml````

## Needed on UTA GPU cluster but not elsewhere
Make sure you have access to ````cn-1e1901.shost.uta.edu````.

````
mkdir datasets
cd datasets
ln -s ../SkyFusion-YOLOv9
# otherwise, it can't find the data

# Get the data on the cluster via sFTP.
# The data is here: https://www.kaggle.com/datasets/pranavdurai/skyfusion-aerial-imagery-object-detection-dataset?resource=download
````

# Every time

## Test naive model
````
conda activate yolo_fine_tune
python yolov9_fine_tune.py

For UTA GPU cluster:
Log into VPN (Ivanti)  
ssh -Y chelians@cn-1e1901.shost.uta.edu  
python yolov9_fine_tune.py --got_data=True

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
#in yolov9_fine_tune.py
#set GOT_DATA to False
#set DO_TRAIN to True

python yolov9_fine_tune.py
````

If the GPU runs out of memory, and there is more than one GPU available, try this:
````
CUDA_AVAILABLE_DEVICES=2; python yolov9_fine_tune.py --gpu_number=2
````

#### Notes on timing
````
Each training epoch

On UTA GPU node, 4 A30's (24 GB of RAM):
  Tensor cores: 224, Ada
  CUDA Cores: 3804
  Peak FP 32 TFLOPS: 5.2
  More here: https://www.pny.com/nvidia-a30
  
On UTA HPC head node -- the job will be killed

On CHECK node
  No GPU, takes > 20 minutes :(

On AIS GPU server, GeForce RTX 4070 (12 GB of RAM)
  TBD, ran out of memory
  Tensor Cores: 184, Lovelace
  CUDA Cores: 5888
  RT Cores: 46
  Peak TFLOPs: 29.15
  More here: https://www.techpowerup.com/gpu-specs/geforce-rtx-4070.c3924

On SJSU Colab, T4 (16 GB of RAM, GDDR6): 1 m, 50 seconds
 Tensor Cores: 320, Turing
 CUDA Cores: 2560
 Peak FP32 TFLOPs: 8.1
 More here: https://www.pny.com/nvidia-tesla-t4

On laptop, GeForce RTX 4060 (8 GB of RAM, GDDR6): 18 minutes
 CUDA cores: 96
 Peak FP32 TFLOPs: 15.11
 More here: https://www.techpowerup.com/gpu-specs/geforce-rtx-4060.c4107
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
# When done
````conda deactivate````

# Going further
````conda env export --no-builds > environment.yml````  
Check in your code


# UTA HPC Steps
1. Use Ivanti VPN if you are off campus
1. ssh ````{username}@hpcr8o2rnp.uta.edu````
1. ````sbatch FineTune_check.slurm````

    * You’ll see something like this:  ````Submitted batch job 14580````
    * When it’s done, you’ll see two new files ````slurm-14580.err```` and ````slurm-14580.out````  
    * Those are the output of ````stderr```` and ````stdout````.  Examine those and other files for the status of your job, results, etc.

## Notes
If you need an interactive session (shell prompt) on a compute node, use command ````srun --partition=NAME --pty /bin/bash````, where partition NAME is an available partition.
* Type ````exit```` when done.  Without doing this, common commands like ````git```` won't be there.

Partition names and availability can be found with the ````sinfo```` command. As of this email (Jan 21, 2025), we have NORMAL, LONG, SHORT, LOW and CHECK partitions. Our normal and long partition has the best hardware... NORMAL parition will run job for 8 days. LONG will run for 16 days. We also have SHORT and LOW partition... CHECK will allocate the next available compute node but is limited to 30 minutes of runtime.  
* NB: the partitions are lowercase but they are written in uppercase to show that they are not the usual English words.

Nodes cannot download data, etc. from the internet.  You have to transfer data onto the cluster via sFTP.  

See https://go.uta.edu/hpcinfo and "HPC Users Group" on MS Teams for more tips.

## Slurm Notes
````sinfo```` shows nodes and their status  

````squeue -p {spartition}```` to see job status  (can add "|grep {username}")

More here https://it.engineering.oregonstate.edu/hpc/slurm-howto

tail the stdout file

strace https://superuser.com/questions/473240/redirect-stdout-while-a-process-is-running-what-is-that-process-sending-to-d