"""
@file utils.py
@brief Video loading and annotation helpers (developmental stage, LAG phase).
"""

import math

import numpy as np
import pandas as pd
import torch
from PIL import ImageFile
from pathlib import Path
from torchvision.io import read_video

import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True


ACQUISITION_FREQUENCY = 1 / 15  # one frame every 15 min
NUMBER_OF_FRAMES = 300           # LAG times are normalised by this number of frames
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

## LOADING FUNCTIONS ##

def load_frames(path, size=224, max_frames=None) -> np.ndarray:
    """Video file or folder of images -> (T, size, size) uint8 grayscale (first channel of the video)."""
    path = Path(path)
    frames = []
    if path.is_dir():
        for f in sorted(f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXT)[:max_frames]:
            frames.append(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE))
    else:
        cap = cv2.VideoCapture(str(path))
        ok, img = cap.read()
        while ok and (max_frames is None or len(frames) < max_frames):
            frames.append(img[:, :, 0])
            ok, img = cap.read()
        cap.release()
    if not frames:
        raise ValueError(f"no frame read from {path}")
    return np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in frames])


def load_video(video_path: str) -> torch.Tensor:
    """
    @brief Load video frames either from video file or image directory
    
    @details This function handles both video files (.mp4, .avi) and directories of images,
    returning a standardized tensor representation in grayscale.
    
    @param video_path (string) Path to video file or directory containing frames
    
    @return frames (torch.Tensor) Video frames tensor of shape (T, H, W)
    
    @note Frames are extracted and normalized to [0,1]
    """
    if video_path.endswith(('.mp4', '.avi')):
    
        frames, _, _ = read_video(video_path, pts_unit='sec')  # (T, H, W, C)
        frames = frames[:, :, :, 0]
        frames = frames.float() / 255.  # [0, 1]

    else:
        frames = load_frames(video_path)
        frames= frames.float()
    return frames

def normalize(x: torch.Tensor) -> torch.Tensor:
    """(B, 1, H, W) in [0, 1] -> (B, 3, H, W) normalised with the ImageNet statistics."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x.repeat(1, 3, 1, 1) - mean) / std

## TOOLS  ##

def time_to_idx(time, t0, frequency=ACQUISITION_FREQUENCY):
    """
    @brief Computes the frame index corresponding to the timestamp

    @param time (float) Current timestamp.
    @param t0 (float) Initial timestamp.

    @return Integer index corresponding to the timestamp.
    """
    return math.floor((int(time) - t0) * frequency)


def add_labels(labels, start_idx, end_idx, label):
    """
    @brief Append labels to a list over a given index range.

    @details Adds a repeated label for each index between start_idx and end_idx
             (end_idx is excluded).

    @param labels (list) List of labels to append to.
    @param start_idx (int) Starting index (inclusive).
    @param end_idx (int) Ending index (exclusive).
    @param label (int) Label to assign to each index in the range.

    @return None.
    """
    labels.extend([label] * (end_idx - start_idx))


## ANNOTATIONS ##
    
def split_cdt(start_times, end_times, max_gap=4):
    """
    @brief Split a cytokinesis period into two sub-cytokinesis periods if the number of frames between two cytokinesis
            is bigger than max_gap
    @details ex: 2-cell | CDT CDT CDT CDT CDT CDT CDT CDT | 4-cell -> 2-cell | CDT CDT 3-cell 3-cell 3-cell 3-cell CDT CDT CDT | 4-cell
                It creates 2 new sub-ecs instead of one ec

    @param start_times (list int) List of the starts of the cytokinesis of the given EC
    @param end_times (list int) List of the ends of the cytokinesis of the given EC
    
    @return sub_cdts (list)
    """
    n = len(start_times)
    assert n == len(end_times)

    sub_ecs = []
    current_starts = [start_times[0]]
    current_ends = [end_times[0]]

    for k in range(1, n):
        gap = start_times[k] - end_times[k-1]

        if gap > max_gap:
            sub_ecs.append({
                "start_times": current_starts,
                "end_times": current_ends,
                "n_div": len(current_starts)
            })
            current_starts = [start_times[k]]
            current_ends = [end_times[k]]
        else:
            current_starts.append(start_times[k])
            current_ends.append(end_times[k])

    sub_ecs.append({
        "start_times": current_starts,
        "end_times": current_ends,
        "n_div": len(current_starts)
    })

    return sub_ecs

def calculate_cell_count(ec_list):
    """
    @brief

    @param

    @return
    Calcule dynamiquement le nombre de cellules en respectant la règle :
    - Si pas d'ECj intercalé → dernier sub-ECi = eci_cells
    - Si ECj intercalé → décompte progressif normal
    """
    sorted_ecs = sorted(ec_list, key=lambda x: x['tmin'])
    
    # Stocker les cells finales par EC original
    ec_final_cells = {}
    for ec in sorted_ecs:
        if ec.get('is_cdt', True):
            ec_final_cells[ec['i']] = ec['cells']
    
    current_cells = 1
    
    for i, ec in enumerate(sorted_ecs):
        if ec.get('is_cdt', True):
            is_last_sub_ec = True
            for next_ec in sorted_ecs[i+1:]:
                if next_ec.get('is_cdt', True) and next_ec['i'] == ec['i']:
                    is_last_sub_ec = False
                    break
            
            # Check whether an ECj (j > i) has been inserted BEFORE this sub-EC
            ecj_intercalated = False
            for prev_ec in sorted_ecs[:i]:
                if (prev_ec.get('is_cdt', True) and 
                    prev_ec['i'] > ec['i'] and 
                    prev_ec['tmax'] < ec['tmin']):
                    ecj_intercalated = True
                    break
            
            if is_last_sub_ec and not ecj_intercalated:
                # CASE 1: No ECj (j > i) inserted → we set the final number of cells of the ECi
                ec['cells_before'] = current_cells
                ec['cells_after'] = ec_final_cells[ec['i']]
                current_cells = ec_final_cells[ec['i']]
            else:
                # CASE 2: ECj inserted OR not the last sub-EC → progressive count depending on the number of cytokinesis
                cells_after = current_cells + ec['n_div']
                ec['cells_before'] = current_cells
                ec['cells_after'] = cells_after
                current_cells = cells_after
        else:
            # Between periods
            ec['cells_before'] = current_cells
            ec['cells_after'] = current_cells
    
    return sorted_ecs

def split_ecs(ec_list, max_gap=4):
    new_list = []
    
    for ec in ec_list:
        if ec.get("is_cdt", True) is False:
            new_list.append(ec)
            continue

        starts = ec['start_times']
        ends = ec['end_times']
        cells = ec['cells']

        sub_ecs = split_cdt(starts, ends, max_gap)

        for sub in sub_ecs:
            t_all = sorted(sub['start_times'] + sub['end_times'])
            new_list.append({
                'i': ec['i'],
                'tmin': min(t_all),
                'tmax': max(t_all),
                'start_times': sub['start_times'],
                'end_times': sub['end_times'],
                'all_times': t_all,
                'n_div': sub['n_div'],
                'cells': cells # "True" value computed after with calculate_cell_count
            })

    new_list = calculate_cell_count(new_list)
    
    return new_list

def swap_ec(ec_list):
    swapped = True
    while swapped:
        swapped = False
        i = 0
        nb_ec = len(ec_list)
        while i < nb_ec - 1:
            curr = ec_list[i]
            next_ = ec_list[i + 1]

            if next_['tmin'] < curr['tmin'] and next_['i'] > curr['i']:
                # Swap EC
                ec_list[i], ec_list[i+1] = next_, curr
                ec_list[i+1]['cells'] = min(9, next_['cells'])
                swapped = True
                # After a swap, restart from the beginning
                break
            i += 1
    return ec_list

def getlabel_cda(data, version: str, video_path: str, start_frame: int, end_frame: int,
                 max_frame: int = 300) -> torch.Tensor:
   
    row = data.loc[video_path]
    t0 = row["t0"]
    
    def get_class(c):
        val = min(int(c), 9)
        return val

    def times_to_idx(times_str):
        return [time_to_idx(int(t), t0) for t in str(times_str).split('-') if t != '#N/D']

    # 1. Detection of the first cell division that does not satisfy the version condition: 
    # either there is no cytokinesis in this cell division, or there is another type of developmental abnormality
    # (ex: direct cleavage, reverse cleavage)
    first_false_ec_index = None
    for i in range(1, 5):
        ok = bool(row.get(f"EC{i}_ok_{version}", False))
        if not ok and first_false_ec_index is None:
            first_false_ec_index = i

    
    # If EC1 does not meet the conditions, we consider that the embryo development is arrested 
    # All the frames are labelled  with 10 = Blocked
    if first_false_ec_index == 1:
        return torch.full((end_frame - start_frame,), 10, dtype=torch.long)

    # 2. Collection of valid ECs (only those prior to the first invalid one)
    ec_list = []
    max_valid_index = first_false_ec_index - 1 if first_false_ec_index is not None else 4
    
    for i in range(1, max_valid_index + 1):
        ok = bool(row.get(f"EC{i}_ok_{version}", False))
        times = row.get(f"EC{i}_times", '#N/D')
        cells = row.get(f"EC{i}_cells", 0)
        
        if ok and times not in ('#N/D', None):
            t_list = times_to_idx(times)
            if t_list:
                start_times = t_list[::2]
                end_times = t_list[1::2]
                
                ec_list.append({
                    'i': i,
                    'tmin': min(t_list),
                    'tmax': max(t_list),
                    'start_times': start_times,
                    'end_times': end_times,
                    'all_times': t_list,
                    'n_div': len(start_times),  # Cytokinesis number
                    'cells': get_class(cells), # Final number of cells at the end of the EC
                })


    ec_list = split_ecs(ec_list, max_gap=4)

    # 3. Overlap handling
    i = 0
    nb_ec = len(ec_list)
    while i < nb_ec - 1:
        ec_list = swap_ec(ec_list)
        curr = ec_list[i]
        next_ = ec_list[i + 1]
        
        if next_['tmin'] < curr['tmax'] and 'is_cdt' not in curr.keys():  # Overlap
            tmax_debut_prime = min([t for t in curr['start_times'] if t > next_['tmax']], default=None)
            
            if tmax_debut_prime is None or curr['tmax'] < next_['tmax']:
                # MERGING ECi and EC(i+1): All ECi t_start are < than EC(i+1) t_max
                curr['tmin'] = min(curr['tmin'], next_['tmin'])
                curr['tmax'] = max(curr['tmax'], next_['tmax'])
                curr['start_times'] = sorted(curr['start_times'] + next_['start_times'])
                curr['end_times'] = sorted(curr['end_times'] + next_['end_times'])
                curr['all_times'] = sorted(curr['start_times'] + curr['end_times'])
                curr['cells'] = next_['cells']
                ec_list[i]= curr
                ec_list.pop(i + 1)
            else:
                # COMPLETE PROCESSING: ECi resumes after EC(i+1)
                tmax_prime = max([t for t in curr['end_times'] if t <= next_['tmin']], default=None)
                tmax_prime = max([t for t in curr['end_times'] if t < next_['tmax']], default=None) if tmax_prime is None else tmax_prime
                curr_original_tmax = curr['tmax']


                if tmax_prime is None:
                    curr['tmin'] = curr['tmin']
                    curr['tmax'] = max(curr['tmax'], next_['tmax'])
                    curr['start_times'] = sorted(curr['start_times'] + next_['start_times'])
                    curr['end_times'] = sorted(curr['end_times'] + next_['end_times'])
                    curr['cells'] = next_['cells'] #+ len([t for t in curr['end_times'] if t <= tmax_prime])  
                    ec_list.pop(i + 1)

                elif tmax_prime >= next_['tmin']:
                    curr_part1 = curr.copy()
                    curr_part1['tmax'] = max(tmax_prime, next_['tmax'])
                    curr_part1['start_times'] = sorted([t for t in curr['start_times'] if t <= tmax_prime] + next_['start_times'])
                    curr_part1['end_times'] = sorted([t for t in curr['end_times'] if t <= tmax_prime] + next_['end_times'])
                    curr_part1['all_times'] = sorted(curr_part1['start_times'] + curr_part1['end_times'])
                    curr_part1['cells'] = next_['cells'] - len([t for t in curr['end_times'] if t > tmax_prime])  
                    ec_list.pop(i + 1)

                    inter_period1 = {
                    'i': 0,
                    'tmin': curr_part1['tmax'],
                    'tmax': tmax_debut_prime, 
                    'start_times': [curr_part1['tmax']],
                    'end_times': [tmax_debut_prime],
                    'all-times': [curr_part1['tmax'], tmax_debut_prime], 
                    'cells': curr_part1['cells'], #+ len([t for t in curr['end_times'] if t <= tmax_prime]),
                    'is_cdt': False
                    } #here t_max might be next_next[tmin] also

                    curr_part2 = curr.copy()
                    curr_part2['tmin'] = tmax_debut_prime
                    curr_part2['tmax'] = curr_original_tmax
                    curr_part2['start_times'] = [t for t in curr['start_times'] if t > tmax_prime]
                    curr_part2['end_times'] = [t for t in curr['end_times'] if t > tmax_prime]
                    curr_part2['all_times'] = sorted(curr_part2['start_times'] + curr_part2['end_times'])
                    curr_part2['cells'] = next_['cells'] 

                    ec_list[i] = curr_part1
                    ec_list.insert(i + 1, inter_period1)
                    ec_list.insert(i + 2, curr_part2)
                    
                
                elif tmax_prime < next_['tmin']:

                    # First part of ECi (up to tmax_prime)
                    curr_part1 = curr.copy()
                    curr_part1['tmax'] = tmax_prime
                    curr_part1['end_times'] = [t for t in curr['end_times'] if t <= tmax_prime]
                    curr_part1['start_times'] = [t for t in curr['start_times'] if t <= tmax_prime]
                    curr_part1['all_times'] = sorted(curr_part1['start_times'] + curr_part1['end_times'])
                    curr_part1['cells'] = next_['cells'] - len([t for t in curr['end_times'] if t > tmax_prime]) - len(next_['end_times'])
                    ec_list.pop(i + 1)
                    
                    # Period between ECi (part 1) and EC(i+1)
                    inter_period1 = {
                        'i': 0,
                        'tmin': tmax_prime,
                        'tmax': next_['tmin'], 
                        'cells': curr_part1['cells'],
                        'start_times': [tmax_prime],
                        'end_times': [next_['tmin']],
                        'all-times': [tmax_prime, next_['tmin']], 
                        'is_cdt': False
                    }

                    
                    # EC(i+1)
                    ec_next = next_.copy()
                    ec_next['cells'] = next_['cells']#+len(curr_part1['end_times'])
                    
                    # Period between EC(i+1) and the restart of ECi
                    inter_period2 = {
                        'i': 0,
                        'tmin': next_['tmax'],
                        'tmax': tmax_debut_prime, #peut etre eussi next_next[tmin] ECi+2
                        'cells': next_['cells']- len([t for t in curr['end_times'] if t > tmax_prime]), #+len(curr_part1['end_times']),
                        'start_times': [next_['tmax']],
                        'end_times': [tmax_debut_prime],
                        'all-times': [next_['tmax'], tmax_debut_prime], 
                        'is_cdt': False
                    }
                    
                    # ECi (part 2)
                    curr_part2 = curr.copy()
                    curr_part2['tmin'] = tmax_debut_prime
                    curr_part2['tmax'] = curr_original_tmax
                    curr_part2['start_times'] = [t for t in curr['start_times'] if t > tmax_prime]
                    curr_part2['end_times'] = [t for t in curr['end_times'] if t > tmax_prime]
                    curr_part2['all_times'] = sorted(curr_part2['start_times'] + curr_part2['end_times'])
                    curr_part2['cells'] = next_['cells'] 

                    ec_list[i] = curr_part1
                    ec_list.insert(i + 1, inter_period1)
                    ec_list.insert(i + 2, ec_next)
                    ec_list.insert(i + 3, inter_period2)
                    ec_list.insert(i + 4, curr_part2)
                    
            nb_ec = len(ec_list)
            continue

        elif next_['tmin'] < curr['tmax']:
            ec_list.pop(i + 1)
            inter_period1 = {
            'i': 0,
            'tmin': curr['tmin'],
            'tmax': next_['tmin'], 
            'start_times': [curr['tmin']],
            'end_times': [next_['tmin']],
            'all-times': [curr['tmin'], next_['tmin']], 
            'cells': curr['cells'], #+ len([t for t in curr['end_times'] if t <= tmax_prime]),
            'is_cdt': False
            } #here t_max might be next_next[tmin] also

            curr_part1 = next_.copy()

            inter_period2 = {
            'i': 0,
            'tmin': next_['tmax'],
            'tmax': curr['tmax'], 
            'start_times': [next_['tmax']],
            'end_times': [curr['tmax']],
            'all-times': [next_['tmax'], curr['tmax']], 
            'cells': next_['cells'], #+ len([t for t in curr['end_times'] if t <= tmax_prime]),
            'is_cdt': False
            }
            

            ec_list[i] = inter_period1 
            ec_list.insert(i + 1, curr_part1)
            ec_list.insert(i + 2, inter_period2)

        i += 1
        nb_ec = len(ec_list)
    ec_list = split_ecs(ec_list, max_gap=4)
    ec_list = calculate_cell_count(ec_list)

    # 4. Construction of the list of labels in chronological order
    labels = []
    
    if not ec_list:
        labels = [10] * max_frame
    else:

        sorted_ecs = sorted(ec_list, key=lambda x: x['tmin'])
        
        current_cells = 1 
        
        for frame in range(max_frame):
            ec_cells = current_cells
            
            for ec in sorted_ecs:
                if ec['tmin'] <= frame < ec['tmax']:
                    ec_cells = 0  # CDT
                    break
                elif frame >= ec['tmax']:
                    ec_cells = ec.get('cells_after', ec['cells'])
                    current_cells = ec_cells
            
            labels.append(get_class(ec_cells))


    if first_false_ec_index is not None:
        # We set 10 (Blocked) starting from the BEGINNING of the first false EC (including tmin)
        start_blocked_frame = 0
        
        if ec_list:  # Valid EC
            start_blocked_frame = ec_list[-1]['tmax']
        else:
            # No valid EC
            start_blocked_frame = 0
        

        for i in range(start_blocked_frame, len(labels)):
                labels[i] = 10


    all_labels = torch.tensor(labels[:max_frame])
 
    return all_labels[start_frame:end_frame]


#SR25
def getlabel_ciray(data: pd.DataFrame, version: str, video_path: str, 
                        start_frame: int, end_frame: int, max_frame: int = 301) -> torch.Tensor:
    row = data.loc[video_path]
    t0 = row["t0"]
    labels = []

    def get_min_max_from_times(times_str):
            times = [time_to_idx(t, t0) for t in times_str.split('-') if t != '#N/D']
            return (min(times), max(times)) if times else (None, None)
    def process_divisions(t_min, t_max, initial_cells, times_str, max_frame):
        #print(f"t_min {t_min}, t_max {t_max}")
        times = [time_to_idx(t, t0) for t in times_str.split('-') if t != '#N/D']
        end_groups = {}
        for i in range(0, len(times), 2):
            if i + 1 < len(times):
                end_time = times[i + 1]
                start_time = times[i]
                if end_time not in end_groups:
                    end_groups[end_time] = []
                end_groups[end_time].append(start_time)

        sorted_ends = sorted(end_groups.keys())
        current_cells = initial_cells
        current_time = min(t_min, max_frame)

        for end_time in sorted_ends:
            if current_time >= max_frame:
                break
            if current_time < end_time:
                add_labels(labels, current_time, min(end_time, max_frame), min(current_cells - 1, 8))
            current_cells += len(end_groups[end_time])
            current_time = end_time

        if current_time < t_max:
            add_labels(labels, current_time, min(t_max, max_frame), min(current_cells - 1, 8))

        return current_cells

    # EC1
    if row["EC1_tmin"] != '#N/D' and not pd.isna(row["EC1_tmin"]) and len(labels) < max_frame:
        ec1_tmin = time_to_idx(row["EC1_tmin"], t0)
        add_labels(labels, 1, min(ec1_tmin, max_frame), 0)

    if bool(row["EC1_ok_" + version]) and len(labels) < max_frame:
        ec1_tmax = time_to_idx(row["EC1_tmax"], t0)
        add_labels(labels, ec1_tmin, min(ec1_tmax, max_frame), 0)

    # EC2
    if bool(row["EC1_ok_" + version]) and row["EC2_times"] != '#N/D' and not pd.isna(row["EC2_times"]) and len(labels) < max_frame:
        ec2_tmin, ec2_tmax = get_min_max_from_times(row["EC2_times"])
        ec1_cells = int(row["EC1_cells"])
        add_labels(labels, ec1_tmax, min(ec2_tmin, max_frame), min(ec1_cells - 1, 8))

    if bool(row["EC2_ok_" + version]) and row["EC2_times"] != '#N/D' and not pd.isna(row["EC2_times"]) and len(labels) < max_frame:
        ec2_tmin, ec2_tmax = get_min_max_from_times(row["EC2_times"])
        ec2_cells = process_divisions(ec2_tmin, min(ec2_tmax, max_frame), int(row["EC1_cells"]), row["EC2_times"], max_frame)
        if ec2_cells != int(row["EC2_cells"]):
            pass
            #print(f'EC2 cell count mismatch: expected {row["EC2_cells"]}, got {ec2_cells}')

    # EC3
    if bool(row["EC2_ok_" + version]) and row["EC3_times"] != '#N/D' and not pd.isna(row["EC3_times"]) and len(labels) < max_frame:
        ec3_tmin, ec3_tmax = get_min_max_from_times(row["EC3_times"])
        add_labels(labels, ec2_tmax, min(ec3_tmin, max_frame), min(ec2_cells - 1, 8))

    if bool(row["EC3_ok_" + version]) and row["EC3_times"] != '#N/D' and not pd.isna(row["EC3_times"]) and len(labels) < max_frame:
        ec3_tmin, ec3_tmax = get_min_max_from_times(row["EC3_times"])
        ec3_cells = process_divisions(ec3_tmin, min(ec3_tmax, max_frame), ec2_cells, row["EC3_times"], max_frame)
        if ec3_cells != int(row["EC3_cells"]):
            pass
            #print(f'EC3 cell count mismatch: expected {row["EC3_cells"]}, got {ec3_cells}')

    # EC4
    if bool(row["EC3_ok_" + version]) and row["EC4_times"] != '#N/D' and not pd.isna(row["EC4_times"]) and len(labels) < max_frame:
        ec4_tmin, ec4_tmax = get_min_max_from_times(row["EC4_times"])
    
        add_labels(labels, ec3_tmax, min(ec4_tmin, max_frame), min(ec3_cells - 1, 8))

    if bool(row["EC4_ok_" + version]) and row["EC4_times"] != '#N/D' and not pd.isna(row["EC4_times"]) and len(labels) < max_frame:
        #times_ec4 = [time_to_idx(t, t0) for t in row["EC4_times"].split('-') if t != '#N/D']
        ec4_tmin, ec4_tmax = get_min_max_from_times(row["EC4_times"])

        ec4_cells = process_divisions(ec4_tmin, min(ec4_tmax, max_frame), ec3_cells, row["EC4_times"], max_frame)
        if ec4_tmax < max_frame:
            add_labels(labels, ec4_tmax, max_frame, min(ec4_cells - 1, 8))
        if ec4_cells != int(row["EC4_cells"]):
            pass
            #print(f'EC4 cell count mismatch: expected {row["EC4_cells"]}, got {ec4_cells}')
    

    return torch.tensor(labels)[start_frame:end_frame]



def get_LAG_phase(data: pd.DataFrame, video_path: str, T: int = NUMBER_OF_FRAMES,
                  max_segments: int = 3) -> torch.Tensor:
    """
    @brief Build the LAG phase annotation of a video from the CSV.

    @details The "LAG" column contains "t_s1-t_e1-t_s2-t_e2-t_s3-t_e3" (timestamps with start t_s and end t_e).
             Times are converted to frame indices and normalised by T.
             Valid segments (end > start) are compacted in chronological order.

    @return torch.Tensor of shape (10,):
            [has_seg1, s1, e1, has_seg2, s2, e2, has_seg3, s3, e3, n_segment]
    """
    lag_tensor = torch.zeros(3 * max_segments + 1, dtype=torch.float32)

    raw = data.loc[video_path, "LAG"]
    phase_str = str(raw).strip()
    if pd.isna(raw) or phase_str in ["#N/D", "None", ""]:
        return lag_tensor

    try:
        times = [float(p) for p in phase_str.split("-")]
    except ValueError:
        return lag_tensor

    t0 = data.loc[video_path, "t0"]
    frame_idxs = [time_to_idx(int(t), t0) for t in times]

    # (start, end) pairs, keep only valid ones, in chronological order
    pairs = [(frame_idxs[i], frame_idxs[i + 1]) for i in range(0, len(frame_idxs) - 1, 2)]
    pairs = sorted((s, e) for s, e in pairs if e > s)[:max_segments]

    for k, (s, e) in enumerate(pairs):
        lag_tensor[3 * k: 3 * k + 3] = torch.tensor([1.0, s / float(T), e / float(T)])
    lag_tensor[-1] = len(pairs)

    return lag_tensor
