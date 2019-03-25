import json
import os
import numpy as np
from skimage import io, transform
import pickle
import time

DATASET_ROOT = '/disk/scott/data/ucsd_ped'  # UCSD ped data set. including two folders: 'data' and 'label'

DATA_DIR = os.path.join(DATASET_ROOT, 'data')  # including 'ped1' and 'ped2'
LABEL_DIR = os.path.join(DATASET_ROOT, 'label')  # label data (e.g. 'UCSDped1_testing_frames_Test001.txt')

STORE_DIR = '../data'  # store serialized data


def load_ped(reprocess=False, resize=False, shape=None):
    '''
    get UCSD ped1&2 train and test data.

    :param reprocess: reprocess or not
    :param resize: resize or not
    :param shape: target shape. e.g. (28, 28)
    :return: (X_train, Y_train), (X_test, Y_test)
    dtype: ndarry
    '''
    print('load train data...')
    start_time = time.time()
    X_train, Y_train = load_ped_train(reprocess, resize, shape)
    print('[train data] time cost: %.6fs' % (time.time() - start_time))
    print('[OK]')
    print('load test data...')
    start_time = time.time()
    X_test, frame_map = load_ped_test(reprocess, resize, shape)
    print('[test data] time cost: %.6fs' % (time.time() - start_time))
    print('[OK]')

    return X_train, Y_train, X_test, frame_map


def load_ped_train(reprocess=False, resize=False, shape=None):
    '''
    get UCSD ped1&2 train data.

    :param reprocess: reprocess or not
    :param resize: resize or not
    :param shape: target shape. e.g. (28, 28)
    :return: X_train, Y_train
    dtype: ndarry
    '''
    train_path = os.path.join(STORE_DIR, 'train.npz')
    # check serialized data
    if not reprocess and not resize and os.path.exists(train_path):
        data = np.load(train_path)
        return data['data'], data['label']
    
    # process now
    file_dir1 = DATA_DIR
    pkgs = ['ped1', 'ped2']
    train_data = None
    for pkg in pkgs:
        pkg_data = None
        file_dir2 = os.path.join(file_dir1, pkg, 'train')
        for video in os.listdir(file_dir2):
            file_dir3 = os.path.join(file_dir2, video, 'box_img')
            video_data = None
            for box in os.listdir(file_dir3):
                path = os.path.join(file_dir3, box)
                img = io.imread(path)
                # resize or not
                if resize and len(shape) == 2:
                    img = transform.resize(img, shape)
                img = np.expand_dims(img, axis=0)
                if video_data is None:
                    video_data = img
                else:
                    video_data = np.vstack((video_data, img))
            if pkg_data is None:
                pkg_data = video_data
            else:
                pkg_data = np.vstack((pkg_data, video_data))
        if train_data is None:
            train_data = pkg_data
        else:
            train_data = np.vstack((train_data, pkg_data))
    num_samples = train_data.shape[0]
    label = np.zeros(num_samples)
    return train_data, label


def load_ped_test(reprocess=False, resize=False, shape=None):
    '''
    get UCSD ped1&2 test data.

    :param reprocess: reprocess or not
    :param resize: resize or not
    :param shape: target shape. e.g. (28, 28)
    :return: X_test(dtype: ndarry), frame_map(dtype: dict)

    '''
    test_path = os.path.join(STORE_DIR, 'test.npz')
    frame_map_path = os.path.join(STORE_DIR, 'frame_map.json')
    # check serialized data
    if not reprocess and not resize and os.path.exists(test_path) and os.path.exists(frame_map_path):
        data = np.load(test_path)
        frame_map = json.load(open(frame_map_path, 'r'))
        return data['data'], frame_map

    # process now
    file_dir1 = DATA_DIR
    label_dir = LABEL_DIR
    pkgs = ['ped1', 'ped2']
    test_data = None
    frame_map = dict()
    box_idx, frame_idx = 0, 0
    for pkg in pkgs:
        pkg_data = None
        file_dir2 = os.path.join(file_dir1, pkg, 'test')
        for video in os.listdir(file_dir2):
            file_dir3 = os.path.join(file_dir2, video, 'box_img')
            video_data = None
            # get frame level label. prepared for box level label
            label_file_name = 'UCSD%s_testing_frames_Test0%s.txt' % (pkg, video[5:7])
            label_file_path = os.path.join(label_dir, label_file_name)
            if os.path.exists(label_file_path):
                video_label = np.loadtxt(label_file_path)
            else:
                video_label = np.zeros(200)
            for box in os.listdir(file_dir3):
                path = os.path.join(file_dir3, box)
                img = io.imread(path)
                # resize or not
                if resize and len(shape) == 2:
                    img = transform.resize(img, shape)
                img = np.expand_dims(img, axis=0)
                if video_data is None:
                    video_data = img
                else:
                    video_data = np.vstack((video_data, img))
                # get index list and label
                frame_id = ('%s_%s' % (pkg, box[:16]))  # e.g. ped1_video01_frame199
                if frame_id not in frame_map.keys():
                    frame_map[frame_id] = dict()
                    frame_map[frame_id]['box_index'] = [box_idx]
                    frame_map[frame_id]['label'] = video_label[int(box_idx[13:16])]
                    frame_idx += 1
                else:
                    frame_map[frame_id]['box_index'].append(box_idx)
                box_idx += 1

            if pkg_data is None:
                pkg_data = video_data
            else:
                pkg_data = np.vstack((pkg_data, video_data))
            
        if test_data is None:
            test_data = pkg_data
        else:
            test_data = np.vstack((test_data, pkg_data))

    print('box_cnt:', box_idx, 'frame_cnt:', frame_idx)
    return test_data, frame_map


if __name__ == '__main__':
    print('load ucsd ped...')
    X_train, Y_train, X_test, frame_map = load_ped(reprocess=True, resize=True, shape=(28, 28))
    np.savez(os.path.join(STORE_DIR, 'train.npz'), data=X_train, label=Y_train)
    json.dump(open(os.path.join(STORE_DIR, 'frame_map.json'), 'w'), frame_map)
    print('X_train.shape:', X_train.shape)
    print('Y_train.shape:', Y_train.shape)
    print('X_test.shape:', X_test.shape)
    print('frame_map shape:', len(frame_map))
    print('save [OK}')
