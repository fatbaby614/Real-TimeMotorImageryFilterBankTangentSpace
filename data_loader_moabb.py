import numpy as np
import pandas as pd
from moabb.datasets import BNCI2014_001, PhysionetMI, Schirrmeister2017
from config.algorithms_config import RANDOM_STATE, LOW_FREQ, HIGH_FREQ, TMIN, TMAX
from scipy import signal


def load_bci_iv_2a_moabb(subjects=None, use_test_data=False):
    """
    使用 MOABB 加载 BCI IV 2A 数据集
    直接使用dataset.get_data()而不使用paradigm，避免pipeline拟合问题
    """
    if subjects is None:
        subjects = list(range(1, 10))
    
    print(f"Loading BCI IV 2A dataset using MOABB...")
    print(f"Subjects: {subjects}")
    print(f"Data type: {'Test' if use_test_data else 'Training'}")
    print(f"Preprocessing: Bandpass filter {LOW_FREQ}-{HIGH_FREQ}Hz, Time window {TMIN}-{TMAX}s")
    
    try:
        # 初始化数据集
        dataset = BNCI2014_001()
        
        # 直接获取原始数据
        print("Getting raw data from MOABB dataset...")
        raw_data = dataset.get_data(subjects=subjects)
        
        all_X = []
        all_y = []
        all_meta = []
        
        # 处理每个受试者的数据
        for subject_id in subjects:
            if subject_id not in raw_data:
                print(f"Subject {subject_id} not found in data")
                continue
                
            subject_data = raw_data[subject_id]
            print(f"Processing subject {subject_id}: {list(subject_data.keys())}")
            
            # 选择session
            if use_test_data:
                session_name = '1test'
            else:
                session_name = '0train'
            
            if session_name not in subject_data:
                print(f"Session {session_name} not found for subject {subject_id}")
                continue
            
            sessions = subject_data[session_name]
            print(f"  Processing {session_name}: {list(sessions.keys())}")
            
            # 处理每个run
            for run_id, raw in sessions.items():
                print(f"    Processing run {run_id}")
                
                # 检查是否有annotations
                if hasattr(raw, 'annotations') and len(raw.annotations) > 0:
                    # 获取annotations
                    annotations = raw.annotations
                    print(f"    Annotations found: {len(annotations)}")
                    
                    # 标签映射
                    label_map = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3}
                    
                    # 处理每个annotation
                    for i, annot in enumerate(annotations):
                        description = str(annot['description'])
                        
                        # 检查是否是我们感兴趣的标签
                        if description in label_map:
                            # 获取事件时间
                            sfreq = raw.info['sfreq']
                            event_sample = int(annot['onset'] * sfreq)
                            label = label_map[description]
                            
                            # 计算时间窗口
                            start_sample = event_sample + int(TMIN * sfreq)
                            end_sample = event_sample + int(TMAX * sfreq)
                            
                            # 检查边界
                            if end_sample > raw.n_times:
                                continue
                            
                            # 提取数据 - 修复get_data调用
                            try:
                                data = raw.get_data(picks=np.arange(22), 
                                                       start=start_sample, 
                                                       stop=end_sample)
                                # get_data返回的是(通道, 时间)形状的数组
                                if isinstance(data, tuple):
                                    data = data[0]
                                elif len(data.shape) == 3:
                                    data = data[0]
                                
                                # 应用带通滤波
                                sfreq = raw.info['sfreq']
                                b, a = signal.butter(4, [LOW_FREQ, HIGH_FREQ], btype='band', fs=sfreq)
                                data = signal.filtfilt(b, a, data, axis=1)
                                
                                all_X.append(data)
                                all_y.append(label)
                                all_meta.append({
                                    'subject': subject_id,
                                    'session': session_name,
                                    'run': run_id,
                                    'event_time': annot['onset']
                                })
                            except Exception as e:
                                continue
                else:
                    print(f"    No annotations found in run {run_id}")
        
        if not all_X:
            raise ValueError("No valid trials found in MOABB data")
        
        X = np.array(all_X)
        y = np.array(all_y)
        meta = pd.DataFrame(all_meta)
        
        print(f"\nData loaded successfully:")
        print(f"  Total samples: {len(X)}")
        print(f"  Channels: {X.shape[1]}")
        print(f"  Time points: {X.shape[2]}")
        print(f"  Classes: {len(np.unique(y))}")
        print(f"  Unique labels: {np.unique(y)}")
        
        return X, y, meta
        
    except Exception as e:
        print(f"Error loading data with MOABB: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to local data loader...")
        # 回退到本地数据加载器
        import data_loader
        return data_loader.load_bci_iv_2a(subjects=subjects, use_test_data=use_test_data)


def load_physionet_mi_moabb(subjects=None, use_test_data=False):
    """
    使用 MOABB 加载 PhysionetMI 数据集
    """
    if subjects is None:
        subjects = list(range(1, 110))  # PhysionetMI 有 109 个受试者
    
    print(f"Loading PhysionetMI dataset using MOABB...")
    print(f"Subjects: {subjects}")
    print(f"Data type: {'Test' if use_test_data else 'Training'}")
    print(f"Preprocessing: Bandpass filter {LOW_FREQ}-{HIGH_FREQ}Hz, Time window {TMIN}-{TMAX}s")
    
    try:
        # 初始化数据集
        dataset = PhysionetMI()
        
        # 直接获取原始数据
        print("Getting raw data from MOABB dataset...")
        raw_data = dataset.get_data(subjects=subjects)
        
        all_X = []
        all_y = []
        all_meta = []
        
        # 处理每个受试者的数据
        for subject_id in subjects:
            if subject_id not in raw_data:
                print(f"Subject {subject_id} not found in data")
                continue
                
            subject_data = raw_data[subject_id]
            print(f"Processing subject {subject_id}: {list(subject_data.keys())}")
            
            # PhysionetMI cross-session protocol: use first session for train, second for test
            # Sort sessions to ensure consistent ordering
            sorted_session_names = sorted(subject_data.keys())
            print(f"  Available sessions: {sorted_session_names}")
            
            # Collect subject data first (for single-session case, load ALL then split per class)
            subject_X = []
            subject_y = []
            subject_meta = []
            
            # Select sessions based on use_test_data flag
            if len(sorted_session_names) >= 2:
                # Multiple sessions available: use separate sessions for train/test
                if use_test_data:
                    target_sessions = sorted_session_names[1:2]
                    print(f"  Selecting TEST sessions: {target_sessions}")
                else:
                    target_sessions = sorted_session_names[0:1]
                    print(f"  Selecting TRAIN sessions: {target_sessions}")
            else:
                # Only 1 session available (e.g., PhysionetMI):
                # Load ALL data first, then do stratified per-class split
                target_sessions = [sorted_session_names[0]]
                print(f"  Single session detected: will load all runs then stratified split per class")
            
            # 处理每个选定的session
            for session_name in target_sessions:
                sessions = subject_data[session_name]
                print(f"  Processing session: {session_name}")
                
                # 处理每个run
                for run_id, raw in sessions.items():
                    print(f"    Processing run {run_id}")
                    
                    # 检查是否有annotations
                    if hasattr(raw, 'annotations') and len(raw.annotations) > 0:
                        # 获取annotations
                        annotations = raw.annotations
                        print(f"    Annotations found: {len(annotations)}")
                        
                        # 标签映射 (PhysionetMI 有4个类别)
                        label_map = {'left_hand': 0, 'right_hand': 1, 'hands': 2, 'feet': 3}
                        
                        # 处理每个annotation
                        for i, annot in enumerate(annotations):
                            description = str(annot['description'])
                            
                            # 检查是否是我们感兴趣的标签
                            if description in label_map:
                                # 获取事件时间
                                sfreq = raw.info['sfreq']
                                event_sample = int(annot['onset'] * sfreq)
                                label = label_map[description]
                                
                                # 计算时间窗口
                                start_sample = event_sample + int(TMIN * sfreq)
                                end_sample = event_sample + int(TMAX * sfreq)
                                
                                # 检查边界
                                if end_sample > raw.n_times:
                                    continue
                                
                                # 提取数据
                                try:
                                    # PhysionetMI 通常有 64 个通道，我们使用所有可用通道
                                    n_channels = min(64, len(raw.info['ch_names']))
                                    data = raw.get_data(picks=np.arange(n_channels), 
                                                           start=start_sample, 
                                                           stop=end_sample)
                                    
                                    # 应用带通滤波
                                    sfreq = raw.info['sfreq']
                                    b, a = signal.butter(4, [LOW_FREQ, HIGH_FREQ], btype='band', fs=sfreq)
                                    data = signal.filtfilt(b, a, data, axis=1)
                                    
                                    subject_X.append(data)
                                    subject_y.append(label)
                                    subject_meta.append({
                                        'subject': subject_id,
                                        'session': session_name,
                                        'run': run_id,
                                        'event_time': annot['onset']
                                    })
                                except Exception as e:
                                    continue
                    else:
                        print(f"    No annotations found in run {run_id}")
            
            # For single-session datasets, do stratified per-class split
            if len(sorted_session_names) < 2:
                subject_X_arr = np.array(subject_X)
                subject_y_arr = np.array(subject_y)
                subject_meta_df = pd.DataFrame(subject_meta)
                
                print(f"  Stratified split: {len(subject_X_arr)} total trials")
                unique_classes = np.unique(subject_y_arr)
                print(f"    Classes: {unique_classes}")
                
                selected_X = []
                selected_y = []
                selected_meta = []
                
                for cls in unique_classes:
                    cls_indices = np.where(subject_y_arr == cls)[0]
                    # Shuffle indices deterministically for reproducible split
                    rng = np.random.RandomState(42)
                    rng.shuffle(cls_indices)
                    
                    split_point = len(cls_indices) // 2
                    if not use_test_data:
                        cls_selected = cls_indices[:split_point]
                        print(f"    Class {cls}: {len(cls_indices)} trials -> TRAIN: {len(cls_selected)}")
                    else:
                        cls_selected = cls_indices[split_point:]
                        print(f"    Class {cls}: {len(cls_indices)} trials -> TEST: {len(cls_selected)}")
                    
                    for idx in cls_selected:
                        selected_X.append(subject_X_arr[idx])
                        selected_y.append(subject_y_arr[idx])
                        selected_meta.append(subject_meta_df.iloc[idx].to_dict())
                
                all_X.extend(selected_X)
                all_y.extend(selected_y)
                all_meta.extend(selected_meta)
                
                print(f"  Subject {subject_id}: {len(selected_X)} selected trials, classes={np.unique(selected_y)}")
            else:
                # Multiple sessions: add all collected data directly
                all_X.extend(subject_X)
                all_y.extend(subject_y)
                all_meta.extend(subject_meta)
        
        if not all_X:
            raise ValueError("No valid trials found in PhysionetMI data")
        
        X = np.array(all_X)
        y = np.array(all_y)
        meta = pd.DataFrame(all_meta)
        
        print(f"\nData loaded successfully:")
        print(f"  Total samples: {len(X)}")
        print(f"  Channels: {X.shape[1]}")
        print(f"  Time points: {X.shape[2]}")
        print(f"  Classes: {len(np.unique(y))}")
        print(f"  Unique labels: {np.unique(y)}")
        
        return X, y, meta
        
    except Exception as e:
        print(f"Error loading data with MOABB: {e}")
        import traceback
        traceback.print_exc()
        raise


def load_schirrmeister2017_moabb(subjects=None, use_test_data=False):
    """
    使用 MOABB 加载 Schirrmeister2017 数据集（High-gamma 数据集）
    """
    if subjects is None:
        subjects = list(range(1, 15))  # Schirrmeister2017 有 14 个受试者
    
    print(f"Loading Schirrmeister2017 (High-gamma) dataset using MOABB...")
    print(f"Subjects: {subjects}")
    print(f"Data type: {'Test' if use_test_data else 'Training'}")
    print(f"Preprocessing: Bandpass filter {LOW_FREQ}-{HIGH_FREQ}Hz, Time window {TMIN}-{TMAX}s")
    
    try:
        # 初始化数据集
        dataset = Schirrmeister2017()
        
        # 直接获取原始数据
        print("Getting raw data from MOABB dataset...")
        raw_data = dataset.get_data(subjects=subjects)
        
        all_X = []
        all_y = []
        all_meta = []
        
        # 处理每个受试者的数据
        for subject_id in subjects:
            if subject_id not in raw_data:
                print(f"Subject {subject_id} not found in data")
                continue
                
            subject_data = raw_data[subject_id]
            print(f"Processing subject {subject_id}: {list(subject_data.keys())}")
            
            # Schirrmeister2017 cross-session protocol: use first session for train, second for test
            # Sort sessions to ensure consistent ordering
            sorted_session_names = sorted(subject_data.keys())
            print(f"  Available sessions: {sorted_session_names}")
            
            # Collect subject data first (for single-session case, load ALL then split per class)
            subject_X = []
            subject_y = []
            subject_meta = []
            
            # Select sessions based on use_test_data flag
            if len(sorted_session_names) >= 2:
                # Multiple sessions available: use separate sessions for train/test
                if use_test_data:
                    target_sessions = sorted_session_names[1:2]
                    print(f"  Selecting TEST sessions: {target_sessions}")
                else:
                    target_sessions = sorted_session_names[0:1]
                    print(f"  Selecting TRAIN sessions: {target_sessions}")
            else:
                # Only 1 session available: load ALL data first, then stratified per-class split
                target_sessions = [sorted_session_names[0]]
                print(f"  Single session detected: will load all runs then stratified split per class")
            
            # 处理每个选定的session
            for session_name in target_sessions:
                sessions = subject_data[session_name]
                print(f"  Processing session: {session_name}")
                
                # 处理每个run
                for run_id, raw in sessions.items():
                    print(f"    Processing run {run_id}")
                    
                    # 检查是否有annotations
                    if hasattr(raw, 'annotations') and len(raw.annotations) > 0:
                        # 获取annotations
                        annotations = raw.annotations
                        print(f"    Annotations found: {len(annotations)}")
                        
                        # 标签映射 (Schirrmeister2017 有4个类别)
                        label_map = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3}
                        
                        # 处理每个annotation
                        for i, annot in enumerate(annotations):
                            description = str(annot['description'])
                            
                            # 检查是否是我们感兴趣的标签
                            if description in label_map:
                                # 获取事件时间
                                sfreq = raw.info['sfreq']
                                event_sample = int(annot['onset'] * sfreq)
                                label = label_map[description]
                                
                                # 计算时间窗口
                                start_sample = event_sample + int(TMIN * sfreq)
                                end_sample = event_sample + int(TMAX * sfreq)
                                
                                # 检查边界
                                if end_sample > raw.n_times:
                                    continue
                                
                                # 提取数据
                                try:
                                    # Schirrmeister2017 通常有 128 个通道，我们使用所有可用通道
                                    n_channels = min(128, len(raw.info['ch_names']))
                                    data = raw.get_data(picks=np.arange(n_channels), 
                                                           start=start_sample, 
                                                           stop=end_sample)
                                    
                                    # 应用带通滤波
                                    sfreq = raw.info['sfreq']
                                    b, a = signal.butter(4, [LOW_FREQ, HIGH_FREQ], btype='band', fs=sfreq)
                                    data = signal.filtfilt(b, a, data, axis=1)
                                    
                                    subject_X.append(data)
                                    subject_y.append(label)
                                    subject_meta.append({
                                        'subject': subject_id,
                                        'session': session_name,
                                        'run': run_id,
                                        'event_time': annot['onset']
                                    })
                                except Exception as e:
                                    continue
                    else:
                        print(f"    No annotations found in run {run_id}")
            
            # For single-session datasets, do stratified per-class split
            if len(sorted_session_names) < 2:
                subject_X_arr = np.array(subject_X)
                subject_y_arr = np.array(subject_y)
                subject_meta_df = pd.DataFrame(subject_meta)
                
                print(f"  Stratified split: {len(subject_X_arr)} total trials")
                unique_classes = np.unique(subject_y_arr)
                print(f"    Classes: {unique_classes}")
                
                selected_X = []
                selected_y = []
                selected_meta = []
                
                for cls in unique_classes:
                    cls_indices = np.where(subject_y_arr == cls)[0]
                    rng = np.random.RandomState(42)
                    rng.shuffle(cls_indices)
                    
                    split_point = len(cls_indices) // 2
                    if not use_test_data:
                        cls_selected = cls_indices[:split_point]
                        print(f"    Class {cls}: {len(cls_indices)} trials -> TRAIN: {len(cls_selected)}")
                    else:
                        cls_selected = cls_indices[split_point:]
                        print(f"    Class {cls}: {len(cls_indices)} trials -> TEST: {len(cls_selected)}")
                    
                    for idx in cls_selected:
                        selected_X.append(subject_X_arr[idx])
                        selected_y.append(subject_y_arr[idx])
                        selected_meta.append(subject_meta_df.iloc[idx].to_dict())
                
                all_X.extend(selected_X)
                all_y.extend(selected_y)
                all_meta.extend(selected_meta)
                
                print(f"  Subject {subject_id}: {len(selected_X)} selected trials, classes={np.unique(selected_y)}")
            else:
                all_X.extend(subject_X)
                all_y.extend(subject_y)
                all_meta.extend(subject_meta)
        
        if not all_X:
            raise ValueError("No valid trials found in Schirrmeister2017 data")
        
        X = np.array(all_X)
        y = np.array(all_y)
        meta = pd.DataFrame(all_meta)
        
        print(f"\nData loaded successfully:")
        print(f"  Total samples: {len(X)}")
        print(f"  Channels: {X.shape[1]}")
        print(f"  Time points: {X.shape[2]}")
        print(f"  Classes: {len(np.unique(y))}")
        print(f"  Unique labels: {np.unique(y)}")
        
        return X, y, meta
        
    except Exception as e:
        print(f"Error loading data with MOABB: {e}")
        import traceback
        traceback.print_exc()
        raise


def load_single_subject_moabb(subject_id, use_test_data=False, dataset='BCI_IV_2A'):
    """
    使用 MOABB 加载单个受试者的数据
    """
    if dataset == 'BCI_IV_2A':
        return load_bci_iv_2a_moabb(subjects=[subject_id], use_test_data=use_test_data)
    elif dataset == 'PhysionetMI':
        return load_physionet_mi_moabb(subjects=[subject_id], use_test_data=use_test_data)
    elif dataset == 'Schirrmeister2017':
        return load_schirrmeister2017_moabb(subjects=[subject_id], use_test_data=use_test_data)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_bci_iv_2a(subjects=None, use_test_data=False):
    """
    兼容旧接口的函数
    """
    return load_bci_iv_2a_moabb(subjects=subjects, use_test_data=use_test_data)
