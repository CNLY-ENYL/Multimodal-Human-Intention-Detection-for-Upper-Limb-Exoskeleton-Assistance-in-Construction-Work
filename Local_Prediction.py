if __name__ == "__main__" :
    print("\033cStarting ...\n") # Clear Terminal

# ----   # Modifiable variables   ----
action_to_idx = {
    'Bimanual_Down': 0,
    'Bimanual_Left': 1,
    'Bimanual_Prepare': 2,
    'Bimanual_Right': 3,
    'Bimanual_Up': 4,
    'Unimanual_Down': 5,
    'Unimanual_Left': 6,
    'Unimanual_Prepare': 7,
    'Unimanual_Right': 8,
    'Unimanual_Up': 9
}
root_directory = 'Temporary_Data'                   # Directory where temporary folders are stored
time_for_prediction = 25                            # Time we wait for each prediction
prediction_threshold = 3                            # how much prediction we need to activate
# ------------------------------------

import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try :
    import torch
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from ultralytics import YOLO
except ModuleNotFoundError as Err:
    missing_module = str(Err).replace('No module named ', '')
    missing_module = missing_module.replace("'", '')
    sys.exit(f'No module named {missing_module} try : pip install {missing_module}')

try :
    from Imports.dataloader import HARDataSet
    from Imports.Functions import model_exist
    from Imports.Models.MoViNet.config import _C as config
    from Imports.Models.fusion import FusionModel
except ModuleNotFoundError :
   sys.exit('Missing Import folder, make sure you are in the right directory')

def make_prediction(Dataset):
    Loader = DataLoader(Dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    with torch.no_grad():
        for video_frames, imu_data in Loader:
            video_frames, imu_data = video_frames.to(device), imu_data.to(device)
            predicted = torch.argmax(model(video_frames, imu_data))

            from Imports.Functions import detect_tools_with_fusion_check
            final_tool = detect_tools_with_fusion_check(yolo_model, video_frames, predicted_label)

            print(f"Final detected tool (majority): {final_tool}")
                
    return predicted

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

# If there is no model to load, we stop
if not model_exist() :
    sys.exit("No model to load") # If there is no model to load, we stop


try :
    Done = False
    while not Done :
        try :
            if len(os.listdir(root_directory)) > 1 :
                Done = True
            else : time.sleep (0.1)
        except FileNotFoundError : 
            pass
        print('Waiting for data, launch data collection file')
        time.sleep(0.1)
        print(LINE_UP, end=LINE_CLEAR)
except KeyboardInterrupt :
    sys.exit('\nProgramme Stopped\n')

'''input('Programme Ready, Press Enter to Start')
for i in range(3) :
    print(f'Starting in {3-i}s')
    time.sleep(1)
    print(LINE_UP, end=LINE_CLEAR)'''

Start_Tracking_Time = time.time()

idx_to_action = {v: k for k, v in action_to_idx.items()}    # We invert the dictionary to have the action with the index
tracking = []

transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
dataset = HARDataSet(root_dir=root_directory, transform=transform)

# Fusion model path
ModelToLoad_Path = r"D:\1_DESK\Dissertation_Update\0726\Pre Trained Model\fusion_movinet_final.pt"
ModelName = os.path.basename(ModelToLoad_Path).replace('.pt', '')

# YOLO model path
yolo_model_path = r"D:\1_DESK\Dissertation_Update\0726\Pre Trained Model\best.pt"
yolo_model_name = os.path.basename(yolo_model_path).replace('.pt', '')

if ModelName.endswith('.pt') :
    ModelName = ModelName.replace('.pt','')
else :
    ModelName = ModelName.replace('.pht','')
print(f"Loading Fusion model: {ModelName}")
print(f"Loading YOLO model: {yolo_model_name}")
yolo_model = YOLO(yolo_model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}\n")
LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

model = FusionModel(config.MODEL.MoViNetA0, num_classes=10, lstm_input_size=27, lstm_hidden_size=512, lstm_num_layers=2)
model.load_state_dict(torch.load(ModelToLoad_Path, weights_only = True, map_location=device))
model.to(device)
model.eval()

print("Models loaded successfully. Starting to read data samples...\n")


try : # Main Loop
    print(f'\033cProgramme running   ctrl + C to stop\n\nLoading {ModelName}\nUsing {device}\n')
    old_sample = ''
    first_sample = ''
    for action in action_to_idx:
        tracking.append(0) # We create a variable in the list for each action
    if not os.listdir(root_directory) :
        print('No files in root directory')
        sys.exit(0)
    while True:
        while old_sample == dataset.SampleNumber :
            time.sleep(0.001)
            dataset = HARDataSet(root_dir=root_directory, transform=transform)
        old_sample = dataset.SampleNumber

        try :
            prediction = make_prediction(dataset)
        except Exception as e:
            print(f'Error on {old_sample}')
            continue  
        label = idx_to_action.get(prediction.item(), "Rest")
        tracking[prediction] += 1
        print(f'{old_sample} : {label} at {round(time.time()-Start_Tracking_Time,2)}')
        if first_sample == '' : first_sample = old_sample


except KeyboardInterrupt:
    pass

except FileNotFoundError:
    print("Samples folder got deleted")
    
num_of_predictions = 0
for i in tracking :
    num_of_predictions += i
num_first = int(first_sample.replace('Sample_',''))
num_last = int(old_sample.replace('Sample_',''))

if num_of_predictions > 1 : end_text = 's'
else : end_text = ''
print(f'\nThere were a total of {num_of_predictions} prediction{end_text}, with {(num_last-num_first+1)-num_of_predictions} missed')
for action, i in action_to_idx.items() :
    print(f'{tracking[i]} for {action}')



if __name__ == "__main__" :
    print('\nProgramme Stopped\n')
