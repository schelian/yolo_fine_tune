# imports
from ultralytics import YOLO

import os
from urllib.request import urlretrieve
from zipfile import ZipFile
import time
import torch
import supervision as sv
import json

# constants
NUM_EPOCHS = 3
#BASE_PATH = '/content/' # from SJSU, to run on Colab
BASE_PATH = './' # if running locally
DATA_YML_PATH = BASE_PATH + "SkyFusion-YOLOv9/data.yaml"
WT_PATH = BASE_PATH + 'runs/detect/train3/weights/best.pt' # update number to best epoch trained (e.g., train3->train11)

#TE_IMG_BASE_PATH = BASE_PATH + 'SkyFusion-YOLOv9/test/images/' # original location from fresh download
TE_IMG_BASE_PATH = BASE_PATH + "data/test/"
AIRPLANE_IMG_FNAME = '17497_png_jpg.rf.376a103c16ee8a07cd33e6780dbe01c3.jpg'
SHIP_IMG_FNAME = '0a24a4100_png_jpg.rf.7591a740aebb923a7b9b7cddffbe5d1a.jpg'

RESULT_IMG_BASE_PATH = BASE_PATH + "results/"

GOT_DATA = False # first run: False, after that: True (prevents getting the data again)
#GOT_MODEL = True # first run: False, after that: True (prevents getting the model again)

DO_TRAIN = False # first run: True, after that: False (prevents retraining)
DO_TEST = True #
DO_SAVE_RESULTS = True # turn off for more speed (e.g, if things are displayed on a video, etc.)

def predict_img( img_fname, result_fname_prefix, DO_SAVE=True ):
  # --visualize flag can be used to export feature visualization maps for each layer in the YOLOv9-C model.
  #cmd = "yolo predict model=" + WT_PATH + " source=" + img_fname + " --visualize"
  #print( f"cmd: {cmd}")
  #os.system( cmd )

  # https://docs.ultralytics.com/modes/predict/#images
  # Run batched inference on a list of images
  print( f"predicting on {img_fname}")
  results = model( img_fname )  # return a list of Results objects

  # Process results list
  if ( DO_SAVE_RESULTS ):
    results_save = []
    for result in results:
        # save images (from original code base; probably lots of wasted code)
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        #print( "boxes: " )
        #print( boxes )
        #results_save.append( boxes.data )

        result.show()  # display to screen
        print( f"saving results to {result_fname_prefix}"+".png")
        result.save(filename=result_fname_prefix+".png")  # save to disk

        print( f"saving results to {result_fname_prefix}"+".json")
        result_json = result.to_json( "{result_fname_prefix}"+".json")
        #csv_sink = sv.CSVSink( result_fname_prefix + ".csv") # https://supervision.roboflow.com/0.25.0/how_to/save_detections/#save-detections-as-json
        #detections = sv.Detections.from_ultralytics( result )
        #csv_sink.append( detections )
        with open(result_fname_prefix + ".json", "w") as json_file:
          json.dump(result_json, json_file, indent=4)

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
print( "cwd: " + os.getcwd() ) # goes to BASE_PATH

"""## Check if GPU is there"""
# region
if torch.cuda.is_available():
    device = "cuda"
    print("GPU is available")
else:
    device = "cpu"
    print("GPU is not available, CPU will be used")
#endregion

"""## Get data"""
#region
if ( not GOT_DATA ):
  download_and_unzip(URL, asset_zip_path)
  got_data = True
# files end up in BASE_PATH/Fine-Tuning-YOLOv9.zip
#endregion

"""## Define model"""
#region
# if you need more RAM, etc. first try the HPC or your computer
# if that doesn't work, go with a smaller model such as 9m, 9s or 9t, and note which model you used in your submission
model = YOLO('yolov9c.pt') # files end up in BASE_PATH, yolov9c.pt
#endregion

"""## Baseline Training"""
#region

# each epoch on SJSU Colab, T4 (16 GB of RAM, GDDR6): 1 m, 50 seconds
#  Turing Tensor Cores: 320
#  CUDA Cores: 2560
#  Peak FP32 8.1 TFLOPs
#  More here: https://www.pny.com/nvidia-tesla-t4
# each epoch on laptop w/ NVIDIA GeForce RTX 4060 (8 GB of RAM, GDDR6): 18 minutes
#  CUDA cores: 96
#  Peak FP32 15.11 TFLOPs
#  More here: https://www.techpowerup.com/gpu-specs/geforce-rtx-4060.c4107
if ( DO_TRAIN ):
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
#endregion

"""## IGNORE -- Experiment 1: Freezing the Backbone + Learning Rate at 0.001 and Experiment 2: Freezing Backbone + Learning Rate at 0.01"""
#region
# ignore, from original code base; not helpful

#if (DO_TRAIN):
#  results = model.train(data=DATA_YML_PATH, epochs=100, imgsz=640, freeze=10, lr0=0.001)

#if (DO_TRAIN):
#  results = model.train(data=DATA_YML_PATH, epochs=NUM_EPOCHS, imgsz=640, freeze=10, lr0=0.01)
#endregion

"""## Experiment 3: Freezing Backbone + Enlarged Input Image Size + Learning Rate at 0.01"""
#region
#if (DO_TRAIN):
#  results = model.train(data=DATA_YML_PATH, epochs=NUM_EPOCHS, imgsz=1024, freeze=10, lr0=0.01)
#endregion

"""## Testing"""
#region
print( "*"*80 )
print( "Testing")

if ( DO_TEST ):
  print( f"loading {WT_PATH}" )
  if ( os.path.exists(WT_PATH) ):
    model = YOLO(WT_PATH)
  else:
    print( f"WARNING: fine tuned model does not exist; using default model" )
    model = YOLO('yolov9c.pt')

  # airplane
  img_fname = TE_IMG_BASE_PATH + AIRPLANE_IMG_FNAME
  result_fname = RESULT_IMG_BASE_PATH + "result_airplane"
  predict_img( img_fname, result_fname, DO_SAVE_RESULTS )

  # ship
  img_fname = TE_IMG_BASE_PATH + SHIP_IMG_FNAME
  result_fname = RESULT_IMG_BASE_PATH + "result_ship"
  predict_img( img_fname, result_fname, DO_SAVE_RESULTS )
#endregion
