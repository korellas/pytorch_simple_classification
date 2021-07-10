import os
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import cv2
import random
from copy import deepcopy

import matplotlib.pyplot as plt


def randomChoice(arr, num):
    idxs = np.random.choice(arr.shape[0], num, replace=False)
    return arr[idxs, ...]


def conv2d_np(image, kernel):
    kernel = np.flipud(np.fliplr(kernel)) #XCorrel

    sub_matrices = np.lib.stride_tricks.as_strided(image,
                                                   shape=tuple(np.subtract(
                                                       image.shape, kernel.shape)) + kernel.shape,
                                                   strides=image.strides * 2)

    return np.einsum('ij,klij->kl', kernel, sub_matrices)


def onepad_np(mask, shape):
    res = np.ones(shape[:2])
    x, y = mask.shape
    res[:x, :y] = mask
    return res


def drawbox(x1, y1, w, h, c='r', ax=None):
    x2 = x1 + w
    y2 = y1 + h
    if ax is None:
        ax = plt.gca()
    ax.plot((x1, x1), (y1, y2), c=c)
    ax.plot((x2, x2), (y1, y2), c=c)
    ax.plot((x1, x2), (y1, y1), c=c)
    ax.plot((x1, x2), (y2, y2), c=c)


class DataSet():
    def __init__(self, tv='val'):
        ann_file = f'./skc_film/annotations/instances_{tv}.json'
        self.root = f'./skc_film/{tv}/'
        with open(ann_file, 'r') as f:
            ann_data = json.load(f)
        self.images = ann_data['images']
        self._idx = {x['id']: i for i, x in enumerate(self.images)}
        self.shape = (100, 100)
#         self.width, self.height = 100,100 # 이미지 사이즈. 하드코딩. 범용으로 안쓸거임

        self.bbox_dict = defaultdict(list)
        self.label_dict = defaultdict(list)
        for ann in ann_data['annotations']:
            idx = self._idx[ann['image_id']]
            self.bbox_dict[idx].append(ann['bbox'])
            self.label_dict[idx].append(ann['category_id'])
        for imdata in ann_data['images']:
            idx = self._idx[imdata['id']]
            if self.bbox_dict[idx] == []:
                self.bbox_dict[idx] = [[0.0, 0.0, 0.0, 0.0]]
                self.label_dict[idx] = [0]
        self._classes = {x['id']: x['name'] for x in ann_data['categories']}
        self._classes[0] = 'Pass'
        self._labels = {x['name']: x['id'] for x in ann_data['categories']}
        self._labels['Pass'] = 0
        self._classid_by_imgid = {img_id: class_id[0]
                                  for img_id, class_id in self.label_dict.items()}
#         print(self._classid_by_imgid)
        self._label_by_imgid = {img_id: self._classes[class_id[0]]
                                for img_id, class_id in self.label_dict.items()}
        self._imgid_by_classid = defaultdict(list)
        for img_id, class_id in self._classid_by_imgid.items():
            self._imgid_by_classid[class_id].append(img_id)

    def getImgByClassID(self, x):
        return self._imgid_by_classid[x]

    def getImgByLabel(self, x):
        return self._imgid_by_classid[self._labels[x]]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_dict = self.images[idx]
#         print(img_dict)
        output = dict()
        img_id = img_dict['id']
        idx = self._idx[img_id]
        output['img_id'] = img_id
        output['img_path'] = os.path.join(self.root, img_dict['file_name'])
        output['bbox'] = self.bbox_dict[idx]
        output['class_id'] = self.label_dict[idx]
        output['label'] = [self._classes[class_id] for class_id in output['class_id']]
        return output

    def label(self, idx):
        d = self[idx]
        return d['label']

    def image(self, idx):
        d = self[idx]
        return cv2.imread(d['img_path'])

    def show(self, idx):
        d = self[idx]
        plt.imshow(self.draw(idx))
        plt.title(f"{d['img_path']}\nGT:{d['label']}")

    def drawBox(self, idx, c1, c2):
        x1, y1 = c1
        x2, y2 = c2
        im = self.draw(idx)
        im = cv2.rectangle(im, (int(x1), int(y1)), (x2, y2), (255, 0, 0))
        return im

    def draw(self, idx):
        d = self[idx]
        img = self.image(idx)
        for label, (x1, y1, w, h) in zip(d['label'], d['bbox']):
            x2, y2 = int(x1 + w), int(y1 + h)
            img = cv2.rectangle(img, (int(x1), int(y1)), (x2, y2), (0, 255, 0))
        return img

    def getMask(self, idx):
        d = self[idx]
        mask = np.zeros(self.shape)
        for x1, y1, w, h in d['bbox']:
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(np.ceil(x1 + w))
            y2 = int(np.ceil(y1 + h))
            mask[x1:x2, y1:y2] = 1
#             print(x1,y1,x2,y2)
#         plt.imshow(mask.astype(bool))
#         plt.show()
        return mask.astype(bool)

    def getSpaces(self, idx, input_shape):
        mask = self.getMask(idx).astype(float)
#         k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
#         mask2 = cv2.dilate(mask, k)
        kernel = np.ones(input_shape)
        res = conv2d_np(mask, kernel) > 0
        newmask = np.logical_not(onepad_np(res, self.shape))
        if newmask.sum() == 0:
            return None
        else:
            coords = random.choice(np.array(np.where(newmask > 0)).T)
#             print([(x,x+y) for x,y in zip(coords,input_shape)])
            return tuple((x, x + y) for x, y in zip(coords, input_shape))

# %%


class Patch():
    def __init__(self, image, bbox, mag=(1, 1), pad=(0, 0), blur=7, flip=False, rotation=0):
        '''
        flip과 rotation 은 좌우반전, 180도 하나만 지원. 90도 지원도 가능하지만 흠... 해볼까.
        '''
        self.org_image = image
        self.org_shape = image.shape
        self.org_bbox = bbox #x1,y1,w,h image에 대한 cropbox
        self.mag = mag
        self.flip = flip
        self.rotation = rotation
        self.pad = pad
        self.blur = blur

    @property
    def flip(self):
        return self._flip

    @flip.setter
    def flip(self, x):
        assert isinstance(x, (bool, float, int))
        self._flip = x > 0

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, x):
        assert isinstance(x, (int, float))
        assert x in [0, 90, 180, 270]
        self._rotation = x

    @property
    def rotval(self):
        return int(self._rotation / 90) - 1

    @property
    def blur(self):
        return self._blur

    @blur.setter
    def blur(self, x):
        assert isinstance(x, int)
        assert x % 2 == 1
        assert x > 0
        self._blur = x

    @property
    def pad(self):
        return self._pad

    @pad.setter
    def pad(self, x):
        assert isinstance(x, tuple)
        assert len(x) == 2
        ih, iw = self.org_shape[:2]
        bx, by, bw, bh = self.org_bbox
        lp, rp = bx, iw - bx - bw
        tp, bp = by, ih - by - bh
        max_xp = min(lp, rp)
        max_yp = min(tp, bp)
        assert x[0] <= max_xp
        assert x[1] <= max_yp
        self._pad = x

    def imaug(self, img):
        '''
        flip 이후 rot
        '''
        res = img.copy()
        if self.mag != (1, 1):
            fx, fy = self.mag
            res = cv2.resize(res, dsize=(0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
        if self.flip:
            res = cv2.flip(res, 1)
        if self.rotval != -1:
            res = cv2.rotate(res, self.rotval)
        return res
#     def boxaug(self):
#         x,y,w,h = self.org_bbox

    @property
    def mask(self):
        mask = (np.zeros((self.org_shape[:2])) * 255.).astype(np.uint8)
        px, py = self.pad
        mask[py:mask.shape[0] - py, px:mask.shape[1] - px] = 255
        mask_blurred = cv2.GaussianBlur(mask, (self.blur, self.blur), 0)
        mask_res = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR)
        mask_res = self.imaug(mask_res)
        return mask_res.astype(float) / 255.

    @property
    def image(self):
        '''
        flip 이후 rotate
        '''
        res = self.org_image.copy()
        res = self.imaug(res)
        return res

    @property
    def shape(self):
        return self.image.shape

    @property
    def bbox(self):
        x, y, w, h = self.org_bbox
        ih, iw = self.org_shape[:2]
        r = self.rotval # -1 은 0도.
        if r in [-1, 1]:
            fx, fy = self.mag
        else:
            fy, fx = self.mag
        xs = [x, ih - (h + y), iw - (w + x), y]
        ys = [y, x, ih - (h + y), iw - (w + x)]
        if self.flip:
            xidxs = [1, 0, 3, 2]
            yidxs = [3, 2, 1, 0]
        else:
            xidxs = [1, 2, 3, 0]
            yidxs = [1, 2, 3, 0]

        xidx = xidxs[r]
        yidx = yidxs[r]
        if r in [0, 2]:
            h, w = w, h
        return fx * xs[xidx], fy * ys[yidx], fx * w, fy * h

    @property
    def mag(self):
        return self._mag

    @mag.setter
    def mag(self, x):
        assert isinstance(x, tuple)
        assert len(x) == 2
        self._mag = x

    def overlay(self, src, x, y):
        res = src.copy()
#         x, y = coord
        assert x >= 0
        assert y >= 0
        x1, y1, bw, bh = self.bbox #bbox 의 위치 크기
        h, w = self.shape[:2] #crop image 의 위치 크기
        assert y + h <= src.shape[0]
        assert x + w <= src.shape[1]
#         fx, fy = self.mag
        pat = self.image
        bg_mean = src.mean()
        sx1, sy1 = int(x1), int(y1)
        patch_mean = (pat[:sx1, :].mean() + pat[:, :sy1].mean() +
                      pat[-sx1:, :].mean() + pat[:, -sy1:].mean()) / 4
        pat = np.clip(pat.astype(np.float64) - patch_mean + bg_mean, 0, 255)
        res[y:y + h, x:x + w] = (res[y:y + h, x:x + w] * (1 - self.mask) +
                                 pat * self.mask).astype(np.uint8)
#         print(x,y)
        return res, (x1 + x, y1 + y, bw, bh)

    def quaryMaxMag(self, img, x, y):
        #         x,y = coord
        #         x1,x2,y1,y2 = self.org_bbox
        ph, pw = self.org_shape[:2]
        ih, iw = img.shape[:2]
        return ((iw - x) / pw), ((ih - y) / ph)

    def queryMaxCoord(self, img):
        ph, pw = self.shape[:2]
        ih, iw = img.shape[:2]
        return (iw - pw, ih - ph)

    def show(self, ax=None):
        if ax is None:
            ax = plt.gca()
        im = (self.image * self.mask).astype(np.uint8)
        ax.imshow(im)
        drawbox(*self.bbox, ax=ax)

# %%
