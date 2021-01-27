import os
import numpy as np
import torch

import nibabel as nib
from scipy import stats

def load_AAL(root='AAL3v1_1mm.nii.txt'):
        f = open(root, 'r')
        lines = f.readlines()
        BA = {}
        for line in lines:
                names = line.split(' ')
                name = names[1]
                ID = int(names[0])
                BA[name] = ID
        return BA

def ttest(v1, v2):
    if len(v1) == 0 or len(v2) == 0:
        return 1
    t, p = stats.ttest_ind(v1, v2)
    return p

def get_ours_paths(root='np_ours_split_test_final2'):
    asdn_npy = []
    asdp_npy = []
    for r, dirs, names in os.walk(root):
        for name in names:
            if '.npy' in name:
                if 'ASDP' in name:
                    asdp_npy.append(os.path.join(r, name))
                elif 'ASDN' in name:
                    asdn_npy.append(os.path.join(r, name))
    return asdn_npy, asdp_npy

def get_abide_paths(root='np_abide_test_final_std', label='dataset/abide/labels/test7.txt'):
    f = open(label, 'r')
    lines = f.readlines()
    asdn_npy, asdp_npy = [], []
    asdn_names = []; asdp_names = []
    for line in lines:
        conts = line.strip().split('\t')
        name = conts[0].split('/')[-2]
        gt = int(conts[2])
        if gt == 1:
            asdp_names.append(name)
        elif gt == 0:
            asdn_names.append(name)
    #print(len(asdn_names), len(asdp_names))

    for r, dirs, npys in os.walk(root):
        for npy_name in npys:
            if '.npy' in npy_name:
                name = npy_name[:-4]
                if name in asdn_names:
                    asdn_npy.append(os.path.join(r, npy_name))
                elif name in asdp_names:
                    asdp_npy.append(os.path.join(r, npy_name))
    #print(len(asdn_npy), len(asdp_npy))
    return asdn_npy, asdp_npy

def get_array(paths):
    arrays = []
    for p in paths:
        npy = np.load(p)
        arrays.append(npy)
    return arrays

def get_mean(arrays):
    # element wise mean of the np arrays
    m = None 
    for array in arrays:
        if m is None:
            m = array
        else:
            m += array
    #print(len(arrays))
    return m / len(arrays)

def save_csv(csv_path, BA, AAL_data, asdn_npy, asdp_npy):
    f = open(csv_path, 'w')
    # write header
    heads = ['names']
    for k, v in BA.items():
        heads.append(k)
    headers = ','.join(heads)
    f.write(headers+'\n')
    for path in asdn_npy:
        name = path[:-4]
        line = [name]
        npy = np.load(path)
        #print(npy.shape, AAL_data.shape)
        for k, v in BA.items():
            idx = (AAL_data == v)
            d = npy[idx]
            if len(d) == 0:
                m = 0
                #print(d, v)
            else:
                m = np.mean(d)
            line.append('%.4f'%(m))
        mes = ','.join(line)
        f.write(mes + '\n')
    for path in asdp_npy:
        name = path[:-4]
        line = [name]
        npy = np.load(path)
        for k, v in BA.items():
            idx = (AAL_data == v)
            d = npy[idx]
            if len(d) == 0:
                m = 0
            else:
                m = np.mean(d)
            line.append('%.4f'%(m))
        mes = ','.join(line)
        f.write(mes + '\n')
    f.close()   

def deal_ours(root='./', csv_path='csv/ours_test.csv'):
    # 0. load Brain aera
    BA = load_AAL()
    # load template
    template = nib.load('MNI152_T1_1mm.nii.gz')
    AAL_template = nib.load('AAL3v1_1mm.nii.gz')
    AAL_data = AAL_template.get_data()
    #rint(AAL_data[AAL_data==81])
    #exit(0)
    # convert the 181*217*181 aal template to 182*218*182 template in mni152
    # add one more row in each axis
    a = np.zeros((1, 217, 181))
    AAL_data = np.concatenate((AAL_data, a), axis=0)
    a = np.zeros((182,1,181))
    AAL_data = np.concatenate((AAL_data, a), axis=1)
    a = np.zeros((182,218,1))
    AAL_data = np.concatenate((AAL_data, a), axis=2)
    #print(AAL_data.shape)

    t_data = template.get_data()
    brain_area = np.zeros(t_data.shape)
    for k, v in BA.items():
        idx = (AAL_data == v)
        brain_area[idx] = 1.0
    ba = nib.Nifti1Image(brain_area, template.affine, header=template.header)
    #nib.save(ba, 'brain_areas.nii.gz')
    # 1. load the features path
    asdn_npy, asdp_npy = get_ours_paths(root=root)


    # # 2. calculate the voxel wise p-values
    asdn = get_array(asdn_npy)
    asdp = get_array(asdp_npy)
    p_value = ttest(asdn, asdp)
    # save_pcsv(p_value)
    # print(p_value[p_value < 0.05])
    # print(p_value[p_value < 0.01])
    
    # asdn_mean = get_mean(asdn)
    # asdp_mean = get_mean(asdp)

    # data005 = np.zeros(t_data.shape)
    # data001 = np.zeros(t_data.shape)
    # idx1 = p_value < 0.05
    # idx2 = p_value < 0.01
    # # 3. highlight the T1 template
    # # highlight p-value <0.05
    # idx = idx1 & ((asdn_mean > (np.max(asdn_mean) * 0.8))| (asdp_mean > (np.max(asdp_mean) * 0.8)))
    # data005[idx] = 1
    # data005 = data005 * brain_area
    # # highlight p-value < 0.01
    # idx = idx2 & ((asdn_mean > (np.max(asdn_mean) * 0.8))| (asdp_mean > (np.max(asdp_mean) * 0.8)))
    # data001[idx] = 1
    # data001 = data001 * brain_area
    # # save as nii.gz
    # p005 = nib.Nifti1Image(data005, template.affine, header=template.header)
    # p001 = nib.Nifti1Image(data001, template.affine, header=template.header)
    # nib.save(p005, 'ours_p005.nii.gz')
    # nib.save(p001, 'ours_p001.nii.gz')

    # 4. save the features in csv
    save_csv(csv_path, BA, AAL_data, asdn_npy, asdp_npy)


def deal_abide(root='./', label='dataset/abide/labels/test7.txt', csv_path='csv/abide_test.csv'):
    # 0. load Brain aera
    BA = load_AAL()
    # load template
    template = nib.load('MNI152_T1_1mm.nii.gz')
    AAL_template = nib.load('AAL3v1_1mm.nii.gz')
    AAL_data = AAL_template.get_data()
    # convert the 181*217*181 aal template to 182*218*182 template in mni152
    # add one more row in each axis
    a = np.zeros((1, 217, 181))
    AAL_data = np.concatenate((AAL_data, a), axis=0)
    a = np.zeros((182,1,181))
    AAL_data = np.concatenate((AAL_data, a), axis=1)
    a = np.zeros((182,218,1))
    AAL_data = np.concatenate((AAL_data, a), axis=2)
    #print(AAL_data.shape)

    t_data = template.get_data()
    brain_area = np.zeros(t_data.shape)
    for k, v in BA.items():
        idx = (AAL_data == v)
        brain_area[idx] = 1.0
    ba = nib.Nifti1Image(brain_area, template.affine, header=template.header)
    #nib.save(ba, 'brain_areas.nii.gz')
    # 1. load the features path
    asdn_npy, asdp_npy = get_abide_paths(root=root, label=label)


    # # 2. calculate the voxel wise p-values
    asdn = get_array(asdn_npy)
    asdp = get_array(asdp_npy)

    p_value = ttest(asdn, asdp)
    # save_pcsv p_value 
    # print(p_value[p_value < 0.05])
    # print(p_value[p_value < 0.01])
    
    # asdn_mean = get_mean(asdn)
    # asdp_mean = get_mean(asdp)

    # data005 = np.zeros(t_data.shape)
    # data001 = np.zeros(t_data.shape)
    # idx1 = p_value < 0.05
    # idx2 = p_value < 0.01
    # # 3. highlight the T1 template
    # # highlight p-value <0.05
    # idx = idx1 & ((asdn_mean > (np.max(asdn_mean) * 0.9))| (asdp_mean > (np.max(asdp_mean) * 0.9)))
    # data005[idx] = 1
    # data005 = data005 * brain_area
    # # highlight p-value < 0.01
    # idx = idx2 & ((asdn_mean > (np.max(asdn_mean) * 0.9))| (asdp_mean > (np.max(asdp_mean) * 0.9)))
    # data001[idx] = 1
    # data001 = data001 * brain_area
    # # save as nii.gz
    # p005 = nib.Nifti1Image(data005, template.affine, header=template.header)
    # p001 = nib.Nifti1Image(data001, template.affine, header=template.header)
    # nib.save(p005, 'abide_p005.nii.gz')
    # nib.save(p001, 'abide_p001.nii.gz')

    # 4. save the features in csv
    save_csv(csv_path, BA, AAL_data, asdn_npy, asdp_npy)


if __name__ == '__main__':
    deal_ours(root='np_preschool_test', csv_path = 'csv/preschooler_test.csv')
    # deal_abide(root='np_abide_test', label='dataset/abide/labels/test7.txt', csv_path='csv/abide_test.csv')
    # deal_ours(root='np_preschool_train', csv_path='csv/preschooler_train.csv')
    # deal_abide(root='np_abide_train', label='dataset/abide/labels/train7.txt', csv_path='csv/abide_train.csv')
