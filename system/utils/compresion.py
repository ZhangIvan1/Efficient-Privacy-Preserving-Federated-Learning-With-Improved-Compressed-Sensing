import numpy as np
import sys
import copy
import scipy.fftpack as spfft
import math
import torch
from seal import *


def dct_2d(x):
    return torch.tensor(spfft.dct(spfft.dct(x.numpy().T, norm='ortho').T, norm='ortho'))


def idct_2d(x):
    return torch.tensor(spfft.idct(spfft.idct(x.numpy().T, norm='ortho').T, norm='ortho'))


def get_u_transpose(shape):
    u_transpose = np.zeros((shape[0]**2, shape[1]**2))
    n = shape[0]
    k = 0
    i = 0
    for row in u_transpose:
        row[k+i] = 1
        k += n
        if k >= n*n:
            k = 0
            i += 1

    return u_transpose


def get_transposed_diagonals(u_transposed):
    transposed_diagonals = np.zeros(u_transposed.shape)
    for i in range(u_transposed.shape[0]):
        a = np.diagonal(u_transposed, offset=i)
        b = np.diagonal(u_transposed, offset=u_transposed.shape[0]-i)
        transposed_diagonals[i] = np.concatenate([a, b])

    return transposed_diagonals


class Packages(object):
    def __init__(self, shape=(200, 200)):
        self.Packages_shape = shape  # 包裹形状
        self.Packages_size = shape[0] * shape[1]  # 包裹容量
        self.Packed_item = None  # 打包好的数据
        #        self.Additional_item = None                 #不符合打包条件的数据
        self.is_Compressed = False  # 包裹是否被压缩
        self.Volume_of_Raw_Item = None  # 包裹原始大小
        self.Volume_of_Compressed_Item = None  # 包裹压缩后的大小
        self.Index_of_Item = {}  # 打包索引
        self.Packed_item_en = None
        self.en_shape = None

    # def __add__(self, other):
    #     self.Packed_item += other.Packed_item
    #     return self

    # def __sub__(self, other):
    #     self.Packed_item -= other.Packed_item
    #     return self

    # def __truediv__(self, other):
    #     self.Packed_item /= other
    #     return self

    # def __mul__(self, other):
    #     self.Packed_item *= other
    #     return self

    def pack_up(self, client_model):
        self.Volume_of_Raw_Item = 0

        Item_left = 0  # 左指针
        Item_right = 0  # 右指针
        Package_pt = 0  # 包内偏移
        Package_idx = 0  # 包裹编号
        package = torch.zeros(self.Packages_shape, dtype=torch.float32)

        for name, param in client_model.named_parameters():
            Item_left = 0
            data_shape = param.shape
            data = copy.deepcopy(param.data.view(-1))
            size = data.size()[0]
            self.Volume_of_Raw_Item += size
            Item_right = Item_left + size

            if self.Packages_size - Package_pt == 0:
                self.package_push(package)
                Package_idx += 1
                Package_pt = 0
                package = torch.zeros(self.Packages_shape, dtype=torch.float32)

            self.Index_of_Item[name] = (
                Package_idx, Package_pt, size, data_shape)  # 包裹编号，包内偏移量，数据长度,原数据格式

            if (size <= self.Packages_size - Package_pt):  # 包裹是否能装下
                package.view(-1)[Package_pt:Package_pt + size] = data
                Package_pt += size
            else:
                while (Item_right is not Item_left):
                    a_size = Item_right - Item_left  # 还需打包的数据长度
                    b_size = self.Packages_size - Package_pt  # 包裹剩余容量
                    sub_size = a_size if a_size <= b_size else b_size  # 本轮打包容量
                    package.view(-1)[Package_pt:Package_pt +
                                     sub_size] = data[Item_left:Item_left + sub_size]
                    Item_left += sub_size
                    Package_pt += sub_size

                    if self.Packages_size - Package_pt == 0:
                        self.package_push(package)
                        Package_idx += 1
                        Package_pt = 0
                        package = torch.zeros(
                            self.Packages_shape, dtype=torch.float32)
                    if a_size == 0:
                        break

        self.package_push(package)

    def package_push(self, package):
        if self.Packed_item is None:
            self.Packed_item = copy.deepcopy(package)
        else:
            if self.Packed_item.dim() != 3:
                self.Packed_item = self.Packed_item.unsqueeze(0)
            self.Packed_item = torch.cat(
                (self.Packed_item, copy.deepcopy(package).unsqueeze(0)))
        return self.Packed_item.shape[0]  # 返回下包裹编号

    def unpack(self, global_model, dev):
        if self.is_Compressed:
            print("请先对包裹进行重建后再解压！！！")
            return 0
        # unpacked_item = {}
        for name, param in global_model.named_parameters():
            # for key in self.Index_of_Item:
            data = torch.tensor([], dtype=torch.float32)
            Package_idx, Package_pt, size, data_shape = self.Index_of_Item[name]

            # if size <= self.Packages_size-Package_pt:
            #     data = self.Packed_item[Package_idx][Package_pt:Package_pt+size]
            # else:
            while size != 0:
                if size <= self.Packages_size - Package_pt:
                    data = torch.cat(
                        (data, self.Packed_item[Package_idx].view(-1)[Package_pt:Package_pt + size]), 0)
                    size = 0
                else:
                    data = torch.cat(
                        (data, self.Packed_item[Package_idx].view(-1)[Package_pt:]), 0)
                    size -= self.Packages_size - Package_pt
                    Package_pt = 0
                    Package_idx += 1

            param.data += data.reshape(data_shape).to(dev)
        return global_model

    def package_compresion(self, r):
        temp = []
        self.Volume_of_Compressed_Item = 0
        for idx in range(self.Packed_item.shape[0]):
            if idx == 0:
                temp = dct_2d(
                    self.Packed_item[idx]).view(-1)[:math.ceil(self.Packages_size * r)]
            else:
                if temp.dim() == 1:
                    temp = temp.unsqueeze(0)
                temp = torch.cat(
                    (temp, dct_2d(
                        self.Packed_item[idx]).view(-1)[:math.ceil(self.Packages_size * r)].unsqueeze(0)))
            self.Volume_of_Compressed_Item += math.ceil(self.Packages_size * r)

        self.is_Compressed = True
        # temp = torch.tensor(temp).numpy()

        self.Packed_item = temp
        # print("1",self.Volume_of_Raw_Item)
        # print("2",self.Volume_of_Compressed_Item)

    def package_decompresion(self, r):
        res = []
        for idx in range(self.Packed_item.shape[0]):
            temp = torch.zeros(self.Packages_shape, dtype=torch.float32)
            # print(idx,math.ceil(self.Packages_size * r))
            temp.view(-1)[:math.ceil(self.Packages_size * r)] = self.Packed_item[idx].view(-1)
            if idx == 0:
                res = idct_2d(temp)
            else:
                if res.dim() != 3:
                    res = res.unsqueeze(0)
                res = torch.cat((res, idct_2d(temp).unsqueeze(0)), dim=0)
        self.is_Compressed = False
        self.Packed_item = res

    def package_en(self, ckks_tools):
        matrix = copy.deepcopy(self.Packed_item.cpu().numpy())
        # matrix = np.arange(1, n*n+1).reshape(n, n)
        # print(self.Packed_item.shape)
        self.en_shape = matrix.shape
        # print(self.Packed_item.shape)
        # u_transposed_diagonals = get_transposed_diagonals(u_transposed)
        # u_transposed_diagonals += 0.00000001  # Prevent is_transparent
        # # ---------------------------------------------------------
        # plain_u_diag = []
        # for row in u_transposed_diagonals:
        #     plain_u_diag.append(
        #         ckks_tools["ckks_encoder"].encode(row, ckks_tools["scale"]))
        plain_matrix = ckks_tools["ckks_encoder"].encode(
            matrix.flatten(), ckks_tools["ckks_scale"])
        cipher_matrix = ckks_tools["encryptor"].encrypt(plain_matrix)
        # ---------------------------------------------------------
        self.Packed_item_en = cipher_matrix
        self.Packed_item = None
        print("3",cipher_matrix.size())

    def package_de(self, ckks_tools):
        p1 = ckks_tools["decryptor"].decrypt(self.Packed_item_en)
        vec = ckks_tools["ckks_encoder"].decode(p1)
        # ---------------------------------------------------------
        self.Packed_item = torch.from_numpy(copy.deepcopy(
            vec[:self.en_shape[0]*self.en_shape[1]].reshape(self.en_shape)))
        # print(self.Packed_item.shape)
        self.Packed_item_en = None
