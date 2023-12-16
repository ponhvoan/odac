import glob
import os
import numpy as np

directory = '/home/voan/orca/datasets/visda/'
folders =  ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']

IMG_EXTENSIONS = ['jpg', 'jpeg', 'JPG', 'JPEG', 'png']

if __name__ == '__main__':
    np.random.seed(0)
    cls_num = 6
    ratio = 0.5
    fout_train_label = open('visda_label_%d_%.1f.txt'%(cls_num, ratio), 'w')
    fout_train_unlabel = open('visda_unlabel_%d_%.1f.txt'%(cls_num, ratio), 'w')

    for i, folder_name in enumerate(folders):
        files_src = []
        files_tgt = []
        for extension in IMG_EXTENSIONS:
            files_src.extend(glob.glob(os.path.join(directory, 'train', folder_name,'*' + extension)))
            files_tgt.extend(glob.glob(os.path.join(directory, 'validation', folder_name,'*' + extension)))
        for j in range(min(len(files_src), (len(files_tgt)))):
            if i < cls_num and np.random.rand() < ratio:
                fout_train_label.write('%s %d\n'%(files_src[j][len(directory+'train/'):], i))
            else:
                fout_train_unlabel.write('%s %d\n'%(files_tgt[j][len(directory+'validation/'):], i))      

    fout_train_label.close()
    fout_train_unlabel.close()