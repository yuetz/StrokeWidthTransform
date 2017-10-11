# -*- encoding: utf-8 -*-
from __future__ import print_function
from __future__ import division
"""
swt 基于笔画宽度检测文字
https://github.com/mypetyak/StrokeWidthTransform

详见论文
"Detecting Text in Natural Scenes with Stroke Width Transform"
缺点：耗内存，速度慢，检测结果不好, 代码待改进
步骤：
1. 使用canny 边缘检测算子提取图像的边缘
2. 计算图像的x向和y向导数，即图像的梯度。每个像素的梯度表明了最大对比度的方向。对于边缘像素，像素的梯度和边缘的法向量是等价的
3. 对于每个边缘像素，沿着梯度θ的方向移动，直到遇到下一个边缘像素。如果对应的边缘像素的梯度指向了相反的方向（θ - π），
我们认为新的边缘大致和原来的边缘平行，我们获得了一个笔画的切片。记录该笔画的宽度，并把我们刚刚遍历的在这条线上的所有像素设为该值。
4. 对于可能属于多条线的像素点，把所有像素的值设为线宽的中值。
5. 使用union-find(互斥集合)数据结构连接重叠的切线。在一个集合里面包括了所有重叠的笔画，每一个集合很可能是一个字母或字符
6. 智能过滤上述线段集合。消除太小的宽或高的字符。消除太长或者太粗的字符（宽高比），消除太稀疏的字符（直径与线宽比）
7. 使用k-d树查找与笔画相似（基于线宽）的字符对，与大小相似（基于宽高）的字符对做交集。计算两个字符的角度。
8. 沿着相似的方向，使用k-d树查找类似的字符。这些字符可能构成一个单词。
9. 生成一个最终图，包括结构单词
"""


import math
import time
from collections import defaultdict

import cv2
import numpy as np
import scipy.sparse
import scipy.spatial

t0 = time.clock()

diagnostics = True


class SWTScrubber(object):
    @classmethod
    def scrub(cls, filepath):
        """
        Apply Stroke-Width Transform to image.

        :param filepath: relative or absolute filepath to source image
        :return: numpy array representing result of transform
        """
        canny, sobelx, sobely, theta = cls._create_derivative(filepath)
        swt = cls._swt(theta, canny, sobelx, sobely)
        """步骤5"""
        shapes = cls._connect_components(swt)
        """步骤6"""
        swts, heights, widths, topleft_pts, images = cls._find_letters(swt, shapes)
        word_images = cls._find_words(swts, heights, widths, topleft_pts, images)

        final_mask = np.zeros(swt.shape)
        i = 0
        for word in word_images:
            cv2.imwrite('final' + str(i) + '.jpg', word * 255)
            final_mask += word
            i +=1
        return final_mask

    @classmethod
    def _create_derivative(cls, filepath):
        img = cv2.imread(filepath, 0)
        edges = cv2.Canny(img, 175, 320, apertureSize=3)
        # Create gradient map using Sobel
        sobelx64f = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
        sobely64f = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)

        theta = np.arctan2(sobely64f, sobelx64f)
        if diagnostics:
            cv2.imwrite('edges.jpg', edges)
            cv2.imwrite('sobelx64f.jpg', np.absolute(sobelx64f))
            cv2.imwrite('sobely64f.jpg', np.absolute(sobely64f))
            # amplify theta for visual inspection
            theta_visible = (theta + np.pi) * 255 / (2 * np.pi)
            cv2.imwrite('theta.jpg', theta_visible)
        return edges, sobelx64f, sobely64f, theta

    @classmethod
    def _swt(self, theta, edges, sobelx64f, sobely64f):
        # create empty image, initialized to infinity
        swt = np.empty(theta.shape)
        swt[:] = np.Infinity
        rays = []

        print(time.clock() - t0)

        # now iterate over pixels in image, checking Canny to see if we're on an edge.
        # if we are, follow a normal a ray to either the next edge or image border
        # edgesSparse = scipy.sparse.coo_matrix(edges)
        # δx
        step_x_g = -1 * sobelx64f
        # δy
        step_y_g = -1 * sobely64f
        mag_g = np.sqrt(step_x_g * step_x_g + step_y_g * step_y_g)
        # cos(θ)
        grad_x_g = step_x_g / mag_g
        # sin(θ)
        grad_y_g = step_y_g / mag_g

        """ 步骤3 """
        for x in xrange(edges.shape[1]):
            for y in xrange(edges.shape[0]):
                if edges[y, x] > 0:
                    # step_x = step_x_g[y, x]
                    # step_y = step_y_g[y, x]
                    # mag = mag_g[y, x]
                    grad_x = grad_x_g[y, x]
                    grad_y = grad_y_g[y, x]
                    ray = [(x, y)]
                    prev_x, prev_y, i = x, y, 0
                    while True:
                        i += 1
                        cur_x = math.floor(x + grad_x * i)
                        cur_y = math.floor(y + grad_y * i)

                        if cur_x != prev_x or cur_y != prev_y:
                            # we have moved to the next pixel!
                            try:
                                if edges[cur_y, cur_x] > 0:
                                    # found edge,
                                    ray.append((cur_x, cur_y))
                                    # theta_point = theta[y, x]
                                    # alpha = theta[cur_y, cur_x]
                                    # 确保两个点的梯度夹角大于pi/2，小于等于pi，即梯度方向相反
                                    if math.acos(grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[
                                        cur_y, cur_x]) < np.pi / 2.0:
                                        # stroke width
                                        thickness = math.sqrt((cur_x - x) * (cur_x - x) + (cur_y - y) * (cur_y - y))
                                        for (rp_x, rp_y) in ray:
                                            swt[rp_y, rp_x] = min(thickness, swt[rp_y, rp_x])
                                        rays.append(ray)
                                    break
                                # this is positioned at end to ensure we don't add a point beyond image boundary
                                ray.append((cur_x, cur_y))
                            except IndexError:
                                # reached image boundary
                                break
                            prev_x = cur_x
                            prev_y = cur_y

        # Compute median SWT
        """ 步骤4 """
        for ray in rays:
            median = np.median([swt[y, x] for (x, y) in ray])
            for (x, y) in ray:
                swt[y, x] = min(median, swt[y, x])
        if diagnostics:
            cv2.imwrite('swt.jpg', swt * 100)

        return swt

    @classmethod
    def _connect_components(cls, swt):
        # STEP: Compute distinct connected components
        # Implementation of disjoint-set
        class Label(object):
            def __init__(self, value):
                self.value = value
                self.parent = self
                self.rank = 0

            def __eq__(self, other):
                if type(other) is type(self):
                    return self.value == other.value
                else:
                    return False

            def __ne__(self, other):
                return not self.__eq__(other)

        ld = {}

        def MakeSet(x):
            try:
                return ld[x]
            except KeyError:
                item = Label(x)
                ld[x] = item
                return item

        def Find(item):
            # item = ld[x]
            if item.parent != item:
                item.parent = Find(item.parent)
            return item.parent

        def Union(x, y):
            """
            :param x:
            :param y:
            :return: root node of new union tree
            """
            x_root = Find(x)
            y_root = Find(y)
            if x_root == y_root:
                return x_root

            if x_root.rank < y_root.rank:
                x_root.parent = y_root
                return y_root
            elif x_root.rank > y_root.rank:
                y_root.parent = x_root
                return x_root
            else:
                y_root.parent = x_root
                x_root.rank += 1
                return x_root

        # apply Connected Component algorithm, comparing SWT values.
        # components with a SWT ratio less extreme than 1:3 are assumed to be
        # connected. Apply twice, once for each ray direction/orientation, to
        # allow for dark-on-light and light-on-dark texts
        trees = {}
        # Assumption: we'll never have more than 65535-1 unique components
        label_map = np.zeros(shape=swt.shape, dtype=np.uint16)
        next_label = 1
        # First Pass, raster scan-style
        swt_ratio_threshold = 3.0
        for y in xrange(swt.shape[0]):
            for x in xrange(swt.shape[1]):
                sw_point = swt[y, x]
                if np.Infinity > sw_point > 0:
                    neighbors = [(y, x - 1),  # west
                                 (y - 1, x - 1),  # northwest
                                 (y - 1, x),  # north
                                 (y - 1, x + 1)]  # northeast
                    connected_neighbors = None
                    neighborvals = []

                    for neighbor in neighbors:
                        # west
                        try:
                            sw_n = swt[neighbor]
                            label_n = label_map[neighbor]
                        except IndexError:
                            continue
                        if label_n > 0 and sw_n / sw_point < swt_ratio_threshold and sw_point / sw_n < swt_ratio_threshold:
                            neighborvals.append(label_n)
                            if connected_neighbors:
                                connected_neighbors = Union(connected_neighbors, MakeSet(label_n))
                            else:
                                connected_neighbors = MakeSet(label_n)

                    if not connected_neighbors:
                        # We don't see any connections to North/West
                        trees[next_label] = (MakeSet(next_label))
                        label_map[y, x] = next_label
                        next_label += 1
                    else:
                        # We have at least one connection to North/West
                        label_map[y, x] = min(neighborvals)
                        # For each neighbor, make note that their respective connected_neighbors are connected
                        # for label in connected_neighbors. @todo: do I need to loop at all neighbor trees?
                        trees[connected_neighbors.value] = Union(trees[connected_neighbors.value], connected_neighbors)

        # Second pass. re-base all labeling with representative label for each connected tree
        layers = {}
        contours = defaultdict(list)
        for x in xrange(swt.shape[1]):
            for y in xrange(swt.shape[0]):
                if label_map[y, x] > 0:
                    item = ld[label_map[y, x]]
                    common_label = Find(item).value
                    label_map[y, x] = common_label
                    contours[common_label].append([x, y])
                    try:
                        layer = layers[common_label]
                    except KeyError:
                        layers[common_label] = np.zeros(shape=swt.shape, dtype=np.uint16)
                        layer = layers[common_label]

                    layer[y, x] = 1
        return layers

    @classmethod
    def _find_letters(cls, swt, shapes):
        # STEP: Discard shapes that are probably not letters
        swts = []
        heights = []
        widths = []
        topleft_pts = []
        images = []

        for label, layer in shapes.iteritems():
            (nz_y, nz_x) = np.nonzero(layer)
            east, west, south, north = max(nz_x), min(nz_x), max(nz_y), min(nz_y)
            width, height = east - west, south - north

            # 过滤掉小于一定宽度及高度的字符
            if width < 8 or height < 8:
                continue
            # 过滤掉过于宽高比异常的字符
            if width / height > 10 or height / width > 10:
                continue

            diameter = math.sqrt(width * width + height * height)
            median_swt = np.median(swt[(nz_y, nz_x)])
            # 过滤掉太稀疏的字符（字符的对角线与线宽中值的比值）
            if diameter / median_swt > 10:
                continue

            # 过滤掉过于高或者过于胖的字符
            if width / layer.shape[1] > 0.4 or height / layer.shape[0] > 0.4:
                continue

            # if diagnostics:
            #     print(" written to image.")
            #     cv2.imwrite('layer' + str(label) + '.jpg', layer * 255)

            # we use log_base_2 so we can do linear distance comparison later using k-d tree
            # ie, if log2(x) - log2(y) > 1, we know that x > 2*y
            # Assumption: we've eliminated anything with median_swt == 1
            swts.append([math.log(median_swt, 2)])
            heights.append([math.log(height, 2)])
            topleft_pts.append(np.asarray([north, west]))
            widths.append(width)
            images.append(layer)

        return swts, heights, widths, topleft_pts, images

    @classmethod
    def _find_words(cls, swts, heights, widths, topleft_pts, images):
        """步骤7"""
        # Find all shape pairs that have similar median stroke widths
        print('SWTS')
        print(swts)
        print('DONESWTS')
        swt_tree = scipy.spatial.KDTree(np.asarray(swts))
        stp = swt_tree.query_pairs(1)

        # Find all shape pairs that have similar heights
        height_tree = scipy.spatial.KDTree(np.asarray(heights))
        htp = height_tree.query_pairs(1)

        # Intersection of valid pairings
        isect = htp.intersection(stp)

        chains = [] # 把字符对串连在一起
        pairs = []  # 字符对
        pair_angles = [] # 字符对的角度
        for pair in isect:
            left = pair[0]
            right = pair[1]
            widest = max(widths[left], widths[right])
            distance = np.linalg.norm(topleft_pts[left] - topleft_pts[right])
            # 过滤掉了距离超过字宽3倍以上的字符对
            if distance < widest * 3:
                delta_yx = topleft_pts[left] - topleft_pts[right]
                angle = np.arctan2(delta_yx[0], delta_yx[1])
                if angle < 0:
                    angle += np.pi

                pairs.append(pair)
                pair_angles.append(np.asarray([angle]))

        """步骤8"""
        angle_tree = scipy.spatial.KDTree(np.asarray(pair_angles))
        atp = angle_tree.query_pairs(np.pi / 12)

        for pair_idx in atp:
            pair_a = pairs[pair_idx[0]]
            pair_b = pairs[pair_idx[1]]
            left_a = pair_a[0]
            right_a = pair_a[1]
            left_b = pair_b[0]
            right_b = pair_b[1]

            # @todo - this is O(n^2) or similar, extremely naive. Use a search tree.
            added = False
            for chain in chains:
                if left_a in chain:
                    chain.add(right_a)
                    added = True
                elif right_a in chain:
                    chain.add(left_a)
                    added = True
            if not added:
                chains.append(set([left_a, right_a]))
            added = False
            for chain in chains:
                if left_b in chain:
                    chain.add(right_b)
                    added = True
                elif right_b in chain:
                    chain.add(left_b)
                    added = True
            if not added:
                chains.append(set([left_b, right_b]))

        word_images = []
        for chain in [c for c in chains if len(c) > 3]:
            for idx in chain:
                word_images.append(images[idx])
                # cv2.imwrite('keeper'+ str(idx) +'.jpg', images[idx] * 255)
                # final += images[idx]

        return word_images


local_filename = "/Users/yuetiezhu/Documents/corpus/hwcs_test/swt2.jpg"
final_mask = SWTScrubber.scrub(local_filename)
# final_mask = cv2.GaussianBlur(final_mask, (1, 3), 0)
# cv2.GaussianBlur(sobelx64f, (3, 3), 0)
cv2.imwrite('final.jpg', final_mask * 255)
print(time.clock() - t0)
