import os
import torch
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad
import numpy as np
from models.layers.mesh import Mesh
from models.layers.CustomMesh import CustomMesh
import meshio
import re
from numpy.core.multiarray import _reconstruct

class SimplificationData(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root = opt.dataroot
        self.val = opt.valroot
        self.paths, self.target = self.make_dataset(self.root)
        self.val_path, self.val_target = self.make_dataset(self.val)
        self.size = len(self.paths)
        self.get_mean_std()
        self.get_val_mean_std()
        opt.input_nc = 7


    def __getitem__(self, index):
        path = self.paths[index]
        target_path = self.target[index]
        mesh = Mesh(file=path, opt=self.opt, hold_history=True, export_folder=self.opt.export_folder)
        meta = {}
        meta['mesh'] = mesh
        original_mesh = CustomMesh.from_vtk(path)
        original_mesh.edges = mesh.edges
        meta['original_mesh'] = original_mesh
        # meta['mean'] = self.mean
        # meta['std'] = self.std
        meta['target'] = torch.load(target_path, weights_only=True).unsqueeze(0)
        edge_features = mesh.extract_features()
        meta['edge_features'] = pad(edge_features, self.opt.ninput_edges)
        # meta['edge_features'] = (edge_features - self.mean) / self.std

        val_path = self.val_path[index]
        val_target_path = self.val_target[index]
        val_mesh = Mesh(file=val_path, opt=self.opt, hold_history=True, export_folder=self.opt.export_folder)
        meta['val_mesh'] = val_mesh
        val_original_mesh = CustomMesh.from_vtk(val_path)
        val_original_mesh.edges = val_mesh.edges
        meta['val_original_mesh'] = val_original_mesh
        # meta['val_mean'] = self.val_mean
        # meta['val_std'] = self.val_std
        meta['val_target'] = torch.load(val_target_path, weights_only=True).unsqueeze(0)
        val_edge_features = val_mesh.extract_features()
        meta['val_edge_features'] = pad(val_edge_features, self.opt.ninput_edges)
        # meta['val_edge_features'] = (val_edge_features - self.val_mean) / self.val_std

        return meta

    def __len__(self):
        return self.size

    @staticmethod
    def get_seg_files(paths, seg_dir, seg_ext='.seg'):
        segs = []
        for path in paths:
            segfile = os.path.join(seg_dir, os.path.splitext(os.path.basename(path))[0] + seg_ext)
            assert(os.path.isfile(segfile))
            segs.append(segfile)
        return segs

    @staticmethod
    def get_n_segs(classes_file, seg_files):
        if not os.path.isfile(classes_file):
            all_segs = np.array([], dtype='float64')
            for seg in seg_files:
                all_segs = np.concatenate((all_segs, read_seg(seg)))
            segnames = np.unique(all_segs)
            np.savetxt(classes_file, segnames, fmt='%d')
        classes = np.loadtxt(classes_file)
        offset = classes[0]
        classes = classes - offset
        return classes, offset
    
    @staticmethod
    def make_dataset(path):
        meshes = []
        target = []
        assert os.path.isdir(path), '%s is not a valid directory' % path

        for root, _, fnames in sorted(os.walk(path)):
            for fname in fnames:
                if 'target' in fname:
                    continue
                if is_mesh_file(fname):
                    path = os.path.join(root, fname)
                    meshes.append(path)
                elif fname.endswith('.pt'):
                    path = os.path.join(root, fname)
                    target.append(path)

        # def get_sort_key(filename):
        #     parts = filename.split("_")
        #     num1 = int(parts[-2])          # 提取倒数第二个部分作为 num1
        #     num2 = int(parts[-1].split(".")[0])  # 提取最后部分并移除 .vtk 作为 num2
        #     return (num1, num2)

        def get_sort_key(filename):
            # 使用正则表达式匹配文件名末尾的 _num1_num2.扩展名 模式
            match = re.search(r'_(\d+)_(\d+)\.\w+$', filename)
            if match:
                num1 = int(match.group(1))
                num2 = int(match.group(2))
                return (num1, num2)
        
        print(len(meshes), len(target))

        meshes = sorted(meshes, key=get_sort_key)
        target = sorted(target, key=get_sort_key)

         
        return meshes, target


def read_seg(seg):
    seg_labels = np.loadtxt(open(seg, 'r'), dtype='float64')
    return seg_labels


def read_sseg(sseg_file):
    sseg_labels = read_seg(sseg_file)
    sseg_labels = np.array(sseg_labels > 0, dtype=np.int32)
    return sseg_labels