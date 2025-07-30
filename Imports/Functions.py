import csv  	# for plot_f1_score
import locale  	# For wifi_is_connected
import os  		# for model_exist
import platform
import subprocess
import sys
import pandas as pd
import math
import numpy as np
from collections import Counter

try :
	import matplotlib.pyplot as plt  	# for plot_confusion_matrix and plot_precision_recall_curve
	import seaborn as sns  				# for plot_confusion_matrix
	from sklearn.metrics import precision_recall_curve  # for plot_precision_recall_curve
	from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report  # for plot_f1_score
except ModuleNotFoundError as Err :
	missing_module = str(Err).replace('No module named ','')
	missing_module = missing_module.replace("'",'')
	if missing_module == 'cv2' :
		sys.exit(f'No module named {missing_module} try : pip install opencv-python')
	if missing_module == 'sklearn' :
		sys.exit(f'No module named {missing_module} try : pip install scikit-learn')
	else :
		print(f'No module named {missing_module} try : pip install {missing_module}')

def connected_wifi() :
	os_name = platform.system()
	region = locale.getlocale()

	if os_name == "Windows" :
		if region[0][:-3] == 'fr' :
			list_current_networks_command = "netsh wlan show interfaces"
			network_output = subprocess.check_output(list_current_networks_command, encoding="437")
			str_n_out = str(network_output)
			start_idx = str_n_out.find('État') + 24

			if not str_n_out[start_idx] == 'c':
				sys.exit('Not connected to Wifi')

			Wifi_SSID = ''
			for i in range(start_idx+38, len(str_n_out)) :
				if str_n_out[i] == "\n" :
					break
				Wifi_SSID += str_n_out[i]

			return True, Wifi_SSID
		
		else : 
			print(f'Windows {region[0][:-3].upper()} is not supported')
			return False, 0

	elif os_name == "Linux" :
		try : 
			list_current_networks_command = 'nmcli d'
			network_output = subprocess.check_output(list_current_networks_command, shell=True, text=True)
			str_n_out = str(network_output)
			start_idx = str_n_out.find('wifi') + 10

			if not str_n_out[start_idx] == 'c':
				sys.exit('Not connected to Wifi')

			Wifi_SSID = ''
			for i in range(start_idx + 24, len(str_n_out)):
				if str_n_out[i] == '\n':
					Wifi_SSID = Wifi_SSID.replace(' ', '')
					break
				Wifi_SSID += str_n_out[i]
			
			return True, Wifi_SSID
		
		except subprocess.CalledProcessError:
			print("Network manager is not running")
			return False, 0
	else : 
		print("Unsupported OS")
		return False, 0


def model_exist () :
	if not os.path.exists('Pre Trained Model') :
		print ('Please put the Pre Trained Model in the folder')
		os.makedirs('Pre Trained Model')
		return False
	elif len(os.listdir('./Pre Trained Model')) < 1 :
		print ('Please put a Pre Trained Model in the folder')
		return False
	else :
		for i, _ in enumerate(os.listdir('./Pre Trained Model')) :
			if not os.listdir('./Pre Trained Model')[i].endswith(('.pt','.pth')) :
				print('Please put a valid file in the folder')
				return False
		return os.listdir('./Pre Trained Model')


def format_time(time_in_s) :
	time_h = time_in_s//3600
	time_m = time_in_s%3600//60
	time_s = time_in_s%60

	if time_h != 0:
		return "{}h {}min {}s".format(round(time_h), round(time_m), round(time_s))
	elif time_m != 0:
		return "{}min {}s".format(round(time_m), round(time_s))
	else:
		return "{}s".format(round(time_s))

def plot_confusion_matrix (ConfusionMatrix, ActionNames, path_to_save, Show=False) :
	plt.close()
	plt.figure(figsize=(6, 6))
	sns.heatmap(ConfusionMatrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=ActionNames.keys(), yticklabels=ActionNames.keys())
	plt.xlabel('Predicted')
	plt.ylabel('Actual')
	plt.title('Confusion Matrix')
	plt.savefig(path_to_save + '/Transparent Graphs/Confusion Matrix_Transparent.png', transparent=True)
	plt.savefig(path_to_save + '/Confusion Matrix.png', transparent=False)
	if Show :
		plt.show()

def plot_precision_recall_curve (all_labels, all_scores, ActionNames, path_to_save, Show = False) :
	plt.close()
	for i, class_name in enumerate(ActionNames.keys()):
		precision, recall, _ = precision_recall_curve(all_labels == i, all_scores[:, i])
		plt.plot(recall, precision, lw=2, label=f'{class_name}')

	plt.title('Precision-Recall Curve')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend(loc='best', shadow=True)  # Use 'best' location for the legend
	plt.grid(True)
	plt.savefig(path_to_save+'/Transparent Graphs/Precision-Recall Curve_Transparent.png', transparent=True)
	plt.savefig(path_to_save+'/Precision-Recall Curve.png', transparent=False)
	if Show :
		plt.show()

def plot_f1_score(all_labels, all_scores, ActionNames, path_to_save, Show=False):
	stats = precision_recall_fscore_support(all_labels, all_scores)
	macro = precision_recall_fscore_support(all_labels, all_scores, average='macro')
	weighted = precision_recall_fscore_support(all_labels, all_scores, average='weighted')
	accuracy = accuracy_score(all_labels, all_scores)
	precision = stats[0]
	recall = stats[1]
	fscore = stats[2]
	support = stats[3]
	total = 0
	for i in range(len(support)):
		total += support[i].item()

	final = [("Action", "Precision", "Recall", "F1-score", "Support")]
	for i in range(len(list(ActionNames.keys()))):
		final.append((list(ActionNames.keys())[i], precision[i].item(), recall[i].item(), fscore[i].item(), support[i].item()))
	final.append(("Accuracy", "", "", accuracy, total))
	final.append(("Macro Avg", macro[0].item(), macro[1].item(), macro[2].item(), total))
	final.append(("Weighted Avg", weighted[0].item(), weighted[1].item(), weighted[2].item(), total))

	# Create csv to store model training performances
	with open(path_to_save+"/F1-score.csv", "w", newline="") as csvfile:
		writer = csv.writer(csvfile)
		writer.writerows(final)

	if Show:
		print(classification_report(all_labels, all_scores, target_names = ActionNames))

def plot_training_performances(LossTrainingTracking, LossValidationTracking, AccuracyTrainingTracking, AccuracyValidationTracking, path_to_save, num_epochs, Show = False):
	if num_epochs == 1 :
		num_epochs = 2

	plt.close()
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
	fig.suptitle("NN Training Performances")

	ax1.plot(LossTrainingTracking, '.-', label='Training Loss')
	ax1.plot(LossValidationTracking, '.-', label='Validation Loss')
	ax1.legend()
	ax1.set_title('Loss over Epochs')
	ax1.set_xlim(0, num_epochs-1)
	ax1.set_ylim(0, 1)
	ax1.set_ylabel("Loss")

	ax2.plot(AccuracyTrainingTracking, '.-', label='Training Accuracy')
	ax2.plot(AccuracyValidationTracking, '.-', label='Validation Accuracy')
	ax2.legend()
	ax2.set_title('Accuracy over Epochs')
	ax2.set_xlim(0, num_epochs-1)
	ax2.set_ylim(0, 100)
	ax2.set_xlabel("Epoch")
	ax2.set_ylabel("Accuracy")

	plt.savefig(path_to_save+'/Transparent Graphs/NN Training Performances_Transparent.png', transparent=True)
	plt.savefig(path_to_save+'/NN Training Performances.png', transparent=False)
	if Show :
		plt.show()



def ask_yn(question='(Y/N)')->bool :
	ask = input(question)
	for _ in range(5) :
		if ask.upper() == "Y" or ask.upper() == "YES" :
			return True
		elif ask.upper() == "N" or ask.upper() == "NO" :
			return False
		else : ask = str(input('Yes or No :'))
	sys.exit(0)


def all_the_same(List) :
    """
    function that's returns a tuple, [0] is bool if every item is the same and if false, [1] is the len of the repeating item and [2] is the item
    """
    try :
        counter_list = []
        best_counter = 0
        best_idx = ''
        for i in range(len(List)) :
            counter_list.append(0)
            for item in List :
                if item == List[i] :
                    counter_list[i] += 1
            if counter_list[i] == len(List) :
                return True, 
            elif counter_list[i] > best_counter :
                best_idx = i
                best_counter = counter_list[i]
        return False, counter_list[best_idx], List[best_idx]
    except TypeError :
        sys.exit('Compared object is not a list')


# ---- Helper function for angle calculation ----
def calculate_angle_between_vectors(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a ** 2 for a in v1))
    norm_v2 = math.sqrt(sum(b ** 2 for b in v2))

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    cos_theta = dot / (norm_v1 * norm_v2)
    cos_theta = min(1.0, max(-1.0, cos_theta))  # Clamp value to avoid domain error
    return math.degrees(math.acos(cos_theta))

def extract_arm_direction_angles(rotm_forearm, rotm_upperarm):
    x_forearm = rotm_forearm[0]
    x_upper = rotm_upperarm[0]
    y_upper = rotm_upperarm[1]
    z_upper = rotm_upperarm[2]

    return {
        'flexion_angle': calculate_angle_between_vectors(x_forearm, x_upper),
        'horizontal_angle': calculate_angle_between_vectors(x_forearm, y_upper),
        'twist_angle': calculate_angle_between_vectors(x_forearm, z_upper)
    }

# ---- Main processing function ----
def process_imu_to_new(folder_path):
    """
    Reads imu.csv from the given Sample_x folder,
    calculates joint angles, and saves imu_new.csv in the same folder.
    """
    csv_path = os.path.join(folder_path, 'imu.csv')
    new_csv_path = os.path.join(folder_path, 'imu_new.csv')

    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        print(f"[SKIP] imu.csv does not exist or is empty: {folder_path}")
        return

    df = pd.read_csv(csv_path, header=None)
    new_rows = []

    for _, row in df.iterrows():
        if len(row) < 75:
            print(f"[WARNING] Row has only {len(row)} columns, skipping folder: {folder_path}")
            continue

        # Extract accelerations
        acc1 = row[4:7].tolist()
        acc2 = row[19:22].tolist()
        acc3 = row[34:37].tolist()
        acc4 = row[49:52].tolist()
        acc5 = row[64:67].tolist()

        def safe_rotm(start, end):
            vals = row[start:end].tolist()
            if len(vals) < 9:
                vals += [0.0] * (9 - len(vals))
            return np.array(vals).reshape(3, 3).tolist()

        # Extract rotation matrices
        rotm1 = safe_rotm(6, 15)
        rotm2 = safe_rotm(18, 27)
        rotm3 = safe_rotm(30, 39)
        rotm4 = safe_rotm(42, 51)
        rotm5 = safe_rotm(54, 63)

        # Calculate angles between segments
        angles_12 = extract_arm_direction_angles(rotm1, rotm2)
        angles_23 = extract_arm_direction_angles(rotm2, rotm3)
        angles_45 = extract_arm_direction_angles(rotm4, rotm5)
        angles_53 = extract_arm_direction_angles(rotm5, rotm3)

        new_row = acc1 + acc2 + acc3 + acc4 + acc5 + [
            angles_12['flexion_angle'], angles_12['horizontal_angle'], angles_12['twist_angle'],
            angles_23['flexion_angle'], angles_23['horizontal_angle'], angles_23['twist_angle'],
            angles_45['flexion_angle'], angles_45['horizontal_angle'], angles_45['twist_angle'],
            angles_53['flexion_angle'], angles_53['horizontal_angle'], angles_53['twist_angle']
        ]
        new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows, columns=[
        'acc1_x','acc1_y','acc1_z','acc2_x','acc2_y','acc2_z',
        'acc3_x','acc3_y','acc3_z','acc4_x','acc4_y','acc4_z',
        'acc5_x','acc5_y','acc5_z',
        'angle_12_xx','angle_12_xy','angle_12_xz',
        'angle_23_xx','angle_23_xy','angle_23_xz',
        'angle_45_xx','angle_45_xy','angle_45_xz',
        'angle_53_xx','angle_53_xy','angle_53_xz'
    ])
    new_df.to_csv(new_csv_path, index=False)
    #print(f" imu_new.csv generated: {folder_path}")

def detect_tools_and_closest(yolo_model, image_tensor):
    """
    Use YOLO model to detect hands and tools in a frame.
    - If no hands detected: return list of tool labels detected.
    - If hands detected: return only the closest tool label.
    """
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    results = yolo_model.predict(source=img_np, verbose=False)

    hand_boxes = []
    tool_boxes = []
    tool_labels = []

    for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
        cls_name = results[0].names[int(cls)]
        if "hand" in cls_name.lower():
            hand_boxes.append(box.cpu().numpy())
        else:
            tool_boxes.append((cls_name, box.cpu().numpy()))
            tool_labels.append(cls_name)

    # No hands → return all tool labels
    if not hand_boxes:
        return {
            "mode": "no_hand",
            "tools": tool_labels if tool_labels else ["None"]
        }

    # Hands detected → find closest tool
    selected_tool = None
    min_dist = float("inf")
    for hand_box in hand_boxes:
        hx = (hand_box[0] + hand_box[2]) / 2
        hy = (hand_box[1] + hand_box[3]) / 2
        for tool_name, tool_box in tool_boxes:
            tx = (tool_box[0] + tool_box[2]) / 2
            ty = (tool_box[1] + tool_box[3]) / 2
            dist = ((hx - tx)**2 + (hy - ty)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                selected_tool = tool_name

    return {
        "mode": "hand_detected",
        "tools": [selected_tool] if selected_tool else ["None"]
    }


def detect_tools_with_fusion_check(yolo_model, video_frames, fusion_label):
    """
    Runs YOLO detection on each frame (10 frames) and finds the most frequent tool.
    If Fusion predicts bimanual action but detected tool is not 'long roller',
    prints a torque support message.
    """
    # Ensure shape is [T, C, H, W]
    if video_frames.ndim == 4:  # [C, T, H, W]
        frames = video_frames.permute(1, 0, 2, 3)
    elif video_frames.ndim == 5:  # [B, C, T, H, W]
        frames = video_frames[0].permute(1, 0, 2, 3)
    else:
        raise ValueError("Unexpected video_frames shape")

    detected_tools = []

    for i in range(frames.shape[0]):
        tool_result = detect_tools_and_closest(yolo_model, frames[i])
        if tool_result["tools"]:
            detected_tools.extend(tool_result["tools"])

    if detected_tools:
        tool_counter = Counter(detected_tools)
        most_common_tool, _ = tool_counter.most_common(1)[0]
    else:
        most_common_tool = "None"

    needs_support = "Bimanual" in fusion_label and most_common_tool != "long roller"
    return most_common_tool, needs_support


if __name__ == '__main__' :
	list_test = [] 
	print(all_the_same(list_test))
