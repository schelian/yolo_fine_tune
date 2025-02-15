# -*- coding: utf-8 -*-
"""yolov9-fine-tuning (CMPE 258, Assn 3).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SMDEZGuqi7Wn7glvW-zJEdG89TSTLWCu
"""

# IMPORTANT.  Make sure Runtime is set to GPU (Runtime->Change runtime type->T4)
# If you have a CPU runtime, the kernel will crash.

!pip install ultralytics

# imports
from ultralytics import YOLO

import os
from urllib.request import urlretrieve
from zipfile import ZipFile
import time
import torch

# constants
NUM_EPOCHS = 50
BASE_PATH = '/content/'
#BASE_PATH = './' # if running locally
DATA_YML_PATH = BASE_PATH + "SkyFusion-YOLOv9/data.yaml"
WT_PATH = BASE_PATH + 'runs/detect/train3/weights/best.pt' # update number to best epoch trained (e.g., train3->train11)

TE_IMG_BASE_PATH = BASE_PATH + 'SkyFusion-YOLOv9/test/images/'
AIRPLANE_IMG_FNAME = '17497_png_jpg.rf.376a103c16ee8a07cd33e6780dbe01c3.jpg'
SHIP_IMG_FNAME = '0a24a4100_png_jpg.rf.7591a740aebb923a7b9b7cddffbe5d1a.jpg'

GOT_DATA = False # first run: False, after that: True (prevents getting the data again)
#GOT_MODEL = True # first run: False, after that: True (prevents getting the model again)

DO_TRAIN = False # first run: True, after that: False (prevents retraining)
DO_TEST = True #

def predict_img( img_fname, result_fname ):
  # --visualize flag can be used to export feature visualization maps for each layer in the YOLOv9-C model.
  #cmd = "yolo predict model=" + WT_PATH + " source=" + img_fname + " --visualize"
  #print( f"cmd: {cmd}")
  #os.system( cmd )

  # https://docs.ultralytics.com/modes/predict/#images
  # Run batched inference on a list of images
  print( f"predicting on {img_fname}")
  results = model( img_fname )  # return a list of Results objects

  # Process results list
  for result in results:
      boxes = result.boxes  # Boxes object for bounding box outputs
      masks = result.masks  # Masks object for segmentation masks outputs
      keypoints = result.keypoints  # Keypoints object for pose outputs
      probs = result.probs  # Probs object for classification outputs
      obb = result.obb  # Oriented boxes object for OBB outputs
      result.show()  # display to screen
      print( f"saving results to {result_fname}")
      result.save(filename=result_fname)  # save to disk

# get data
def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assets....", end="")
    # Downloading zip file using urllib package.
    urlretrieve(url, save_path)
    try:
        # Extracting zip file using the zipfile package.
        with ZipFile(save_path) as z:
            # Extract ZIP file contents in the same directory.
            z.extractall(os.path.split(save_path)[0])
        print("Done")
    except Exception as e:
        print("\nInvalid file.", e)
URL = r"https://www.dropbox.com/scl/fi/36utqtkyfg7piqczxlmb3/SkyFusion-YOLOv9.zip?rlkey=c1801ghd40kzs0uk8d4bnelhg&dl=1"
asset_zip_path = os.path.join(os.getcwd(), "Fine-Tuning-YOLOv9.zip")
print( "cwd: " + os.getcwd() ) # goes to /content

"""## Check if GPU is there"""

if torch.cuda.is_available():
    device = "cuda"
    print("GPU is available")
else:
    device = "cpu"
    print("GPU is not available, CPU will be used")

"""## Get data and define model"""

if ( not GOT_DATA ):
  download_and_unzip(URL, asset_zip_path)
  got_data = True
# files end up in /content

"""## Define model"""

# if you need more RAM, etc. first try the HPC or your computer
# if that doesn't work, go with a smaller model such as 9m, 9s or 9t, and note which model you used in your submission
model = YOLO('yolov9c.pt') # files end up in /content, yolov9c.pt

"""## Baseline Training"""

# each epoch on SJSU Colab, T4 (16 GB of RAM, GDDR6): 1 m, 50 seconds
#  Turing Tensor Cores: 320
#  CUDA Cores: 2560
#  Peak FP32 8.1 TFLOPs
#  More here: https://www.pny.com/nvidia-tesla-t4
# each epoch on laptop w/ NVIDIA GeForce RTX 4060 (8 GB of RAM, GDDR6): 18 minutes
#  CUDA cores: 96
#  Peak FP32 15.11 TFLOPs
#  More here: https://www.techpowerup.com/gpu-specs/geforce-rtx-4060.c4107
if ( True or DO_TRAIN ):
  start_time = time.time()

  if ( device != 'cpu' ):
     device = 0

  results = model.train(data=DATA_YML_PATH, epochs=NUM_EPOCHS, imgsz=640, device=device) # https://docs.ultralytics.com/modes/train/#train-settings
  # After training on a custom dataset,
  #  the best weight is automatically stored in the runs/detect/train/weights directory as best.pt
  #  results are saved in runs/detect/train; results.png shows training curves, etc.
  #
  # TODO look into resume
  end_time = time.time()
  elapsed_time = end_time - start_time
  print(f"Time {elapsed_time:.2f} seconds")

"""## Experiment 1: Freezing the Backbone + Learning Rate at 0.001"""

#if (DO_TRAIN):
#  results = model.train(data=DATA_YML_PATH, epochs=100, imgsz=640, freeze=10, lr0=0.001)

"""## Experiment 2: Freezing Backbone + Learning Rate at 0.01"""

#if (DO_TRAIN):
#  results = model.train(data=DATA_YML_PATH, epochs=NUM_EPOCHS, imgsz=640, freeze=10, lr0=0.01)

"""## Experiment 3: Freezing Backbone + Enlarged Input Image Size + Learning Rate at 0.01"""

#if (DO_TRAIN):
#  results = model.train(data=DATA_YML_PATH, epochs=NUM_EPOCHS, imgsz=1024, freeze=10, lr0=0.01)

"""## Testing"""

if ( DO_TEST ):
  print( f"loading {WT_PATH}" )
  if ( os.path.exists(WT_PATH) ):
    model = YOLO(WT_PATH)
  else:
    print( f"WARNING: fine tuned model does not exist; using default model" )
    model = YOLO('yolov9c.pt')

  # airplane
  img_fname = TE_IMG_BASE_PATH + AIRPLANE_IMG_FNAME
  result_fname = "result_airplane.jpg"
  predict_img( img_fname, result_fname )

  # ship
  img_fname = TE_IMG_BASE_PATH + SHIP_IMG_FNAME
  result_fname = "result_ship.jpg"
  predict_img( img_fname, result_fname )