# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 01:31:36 2022

@author: HP
"""

# -*- coding: utf-8 -*-


import numpy as np
import cv2
import math

tzig = np.array([1, 2, 9, 17, 10, 3, 4, 11, 18, 25, 33, 26, 19, 12, 5, 6, 13,
                 20, 27, 34, 41, 49, 42, 35, 28, 21, 14, 7, 8, 15, 22, 29, 36, 43,
                 50, 57, 58, 51, 44, 37, 30, 23, 16, 24, 31, 38, 45, 52, 59, 60, 53,
                 46, 39, 32, 40, 47, 54, 61, 62, 55, 48, 56, 63, 64])


def block_to_zigzag(a):
    arr = np.zeros((64))

    for pointer1 in range(64):
        row = (pointer1 - 1) // 8
        col = pointer1 - 8 * row - 1
        arr[pointer1] = a[row][col]

    return arr


def zigzag_to_block(arr):
    a = np.zeros((8, 8))

    for pointer1 in range(64):
        row = (pointer1 - 1) // 8
        col = pointer1 - 8 * row - 1
        a[row][col] = arr[pointer1]

    return a


def bitstring(y):
    y = int(y)
    x = abs(y)
    if x == 0:
        return 0

    st = bin(x)
    return len(st) - 2


def rle(arr, dci_1):
    n = arr.shape[0]

    x = arr[0] - dci_1

    a = []
    a.append(x)

    cnt = 0

    for pointer1 in range(1, n):
        if (arr[pointer1] != 0):
            a.append((cnt, bitstring(abs(arr[pointer1])), arr[pointer1]))

            cnt = 0
        else:
            cnt = cnt + 1

    a.append((0, 0))

    a = np.array(a, dtype='object')
    return a


import heapq


class node:
    def __init__(self, frequency, identifier, left_child=None, right_child=None):
        self.frequency = frequency

        self.identifier = identifier

        self.left_child = left_child

        self.right_child = right_child

        self.huff = ''

    def __lt__(self, nxt):
        return self.frequency < nxt.frequency


def recursive_huffman(node, temp_dict, coded_string=''):
    new_string_data = coded_string + str(node.huff)

    if (node.right_child):
        recursive_huffman(node.right_child, temp_dict, new_string_data)

    if (node.left_child):
        recursive_huffman(node.left_child, temp_dict, new_string_data)

    if (not node.left_child and not node.right_child):
        temp_dict[node.identifier] = new_string_data


def gen_huffman(node):
    temp_dict = {}
    recursive_huffman(node, temp_dict, coded_string='')

    return temp_dict


def huffman_ac_coeff(zig_vec):
    sym = {}
    temp_size = zig_vec.shape[0]
    for pointer1 in range(temp_size):
        if pointer1 == temp_size - 1:
            break
        key = (zig_vec[pointer1 + 1][0], zig_vec[pointer1 + 1][1])
        if (sym.get(key) == None):
            sym[key] = 1
        else:
            sym[key] += 1

    nodes = []

    for key in sym:
        heapq.heappush(nodes, node(sym[key], key))

    while len(nodes) > 1:
        left_child = heapq.heappop(nodes)
        right_child = heapq.heappop(nodes)

        left_child.huff = 0
        right_child.huff = 1

        newNode = node(left_child.frequency + right_child.frequency, left_child.identifier + right_child.identifier,
                       left_child, right_child)

        heapq.heappush(nodes, newNode)

    # print(nodes[0])

    return nodes[0]


def huffman_dc_coeff(zig_vec):
    sym = {}
    dci_1 = 0
    for pointer1 in range(zig_vec.shape[0]):
        key = bitstring(zig_vec[pointer1] - dci_1)
        if (sym.get(key) == None):
            sym[key] = 1
        else:
            sym[key] += 1
        dci_1 = zig_vec[pointer1]

    nodes = []

    for key in sym:
        heapq.heappush(nodes, node(sym[key], key))

    while len(nodes) > 1:
        left_child = heapq.heappop(nodes)
        right_child = heapq.heappop(nodes)

        left_child.huff = 0
        right_child.huff = 1

        newNode = node(left_child.frequency + right_child.frequency, left_child.identifier + right_child.identifier,
                       left_child, right_child)

        heapq.heappush(nodes, newNode)

    return nodes[0]


def bin_codes_to_rle(parent, string_data, dict_rle, pointer1=0):
    n = len(string_data)

    if ((not parent.left_child and not parent.right_child) or pointer1 == n):
        dict_rle[string_data[0:pointer1]] = parent.identifier
        return parent.identifier, pointer1

    if (dict_rle.get(string_data) == True):
        return dict_rle[string_data], n

    if (string_data[pointer1] == '1'):
        return bin_codes_to_rle(parent.right_child, string_data, dict_rle, pointer1 + 1)

    if (string_data[pointer1] == '0'):
        return bin_codes_to_rle(parent.left_child, string_data, dict_rle, pointer1 + 1)


C = np.array([math.cos(math.pi / 16 * pointer1) for pointer1 in range(8)])
S = np.array([1 / (4 * coded_string) for coded_string in C])
S[0] = 1 / (2 * math.sqrt(2))
A = np.array([
    None,
    C[4],
    C[2] - C[6],
    C[4],
    C[6] + C[2],
    C[6],
])


def transform(vector):
    v0 = vector[0] + vector[7]
    sh = vector[1] + vector[6]
    sh2 = vector[2] + vector[5]
    sh3 = vector[3] + vector[4]
    v4 = vector[3] - vector[4]
    v5 = vector[2] - vector[5]
    v6 = vector[1] - vector[6]
    v7 = vector[0] - vector[7]

    v8 = v0 + sh3
    v9 = sh + sh2
    v10 = sh - sh2
    v11 = v0 - sh3
    v12 = -v4 - v5
    v13 = (v5 + v6) * A[3]
    v14 = v6 + v7

    v15 = v8 + v9
    v16 = v8 - v9
    v17 = (v10 + v11) * A[1]
    v18 = (v12 + v14) * A[5]

    v19 = -v12 * A[2] - v18
    v20 = v14 * A[4] - v18

    v21 = v17 + v11
    v22 = v11 - v17
    v23 = v13 + v7
    v24 = v7 - v13

    v25 = v19 + v24
    v26 = v23 + v20
    v27 = v23 - v20
    v28 = v24 - v19

    return np.array([
        S[0] * v15,
        S[1] * v26,
        S[2] * v21,
        S[3] * v28,
        S[4] * v16,
        S[5] * v25,
        S[6] * v22,
        S[7] * v27,
    ])


def inverse_transform(vector):
    v15 = vector[0] / S[0]
    v26 = vector[1] / S[1]
    v21 = vector[2] / S[2]
    v28 = vector[3] / S[3]
    v16 = vector[4] / S[4]
    v25 = vector[5] / S[5]
    v22 = vector[6] / S[6]
    v27 = vector[7] / S[7]

    v19 = (v25 - v28) / 2
    v20 = (v26 - v27) / 2
    v23 = (v26 + v27) / 2
    v24 = (v25 + v28) / 2

    v7 = (v23 + v24) / 2
    v11 = (v21 + v22) / 2
    v13 = (v23 - v24) / 2
    v17 = (v21 - v22) / 2

    v8 = (v15 + v16) / 2
    v9 = (v15 - v16) / 2

    v18 = (v19 - v20) * A[5]  # Different from original
    v12 = (v19 * A[4] - v18) / (A[2] * A[5] - A[2] * A[4] - A[4] * A[5])
    v14 = (v18 - v20 * A[2]) / (A[2] * A[5] - A[2] * A[4] - A[4] * A[5])

    v6 = v14 - v7
    v5 = v13 / A[3] - v6
    v4 = -v5 - v12
    v10 = v17 / A[1] - v11

    v0 = (v8 + v11) / 2
    sh = (v9 + v10) / 2
    sh2 = (v9 - v10) / 2
    sh3 = (v8 - v11) / 2

    return np.array([
        (v0 + v7) / 2,
        (sh + v6) / 2,
        (sh2 + v5) / 2,
        (sh3 + v4) / 2,
        (sh3 - v4) / 2,
        (sh2 - v5) / 2,
        (sh - v6) / 2,
        (v0 - v7) / 2,
    ])


def dct(arr):
    brr = np.zeros((8, 8))

    for pointer1 in range(8):
        brr[pointer1] = transform(arr[pointer1])

    for pointer2 in range(8):
        zig_vec = np.zeros(8)
        for pointer1 in range(8):
            zig_vec[pointer1] = brr[pointer1][pointer2]

        zig_vec = transform(zig_vec)
        for pointer1 in range(8):
            brr[pointer1][pointer2] = zig_vec[pointer1]

    return brr


def idct(arr):
    brr = np.zeros((8, 8))

    for pointer1 in range(8):
        brr[pointer1] = inverse_transform(arr[pointer1])

    for pointer2 in range(8):
        zig_vec = np.zeros(8)
        for pointer1 in range(8):
            zig_vec[pointer1] = brr[pointer1][pointer2]

        zig_vec = inverse_transform(zig_vec)
        for pointer1 in range(8):
            brr[pointer1][pointer2] = zig_vec[pointer1]

    return brr


def color_1_to_2(image_data):
    n = image_data.shape[0]
    m = image_data.shape[1]

    image = np.zeros((n, m, 3))

    transformation_matrix = np.array([[0.299, 0.587, 0.114], [-0.169, -0.331, -0.5], [0.5, -0.419, -0.081]])
    zig_vec = np.array([0, 128, 128])

    for pointer1 in range(n):
        for pointer2 in range(m):
            random_vector = image_data[pointer1][pointer2]
            t = np.array([random_vector[2], random_vector[1], random_vector[0]])
            image[pointer1][pointer2] = np.dot(transformation_matrix, t) + zig_vec

    return image


def color_2_to_1(image_data):
    n = image_data.shape[0]
    m = image_data.shape[1]

    image = np.zeros((n, m, 3))

    transformation_matrix = np.array([[0.299, 0.587, 0.114], [-0.169, -0.331, -0.5], [0.5, -0.419, -0.081]])
    transformation_matrix = np.linalg.inv(transformation_matrix)

    for pointer1 in range(n):
        for pointer2 in range(m):
            random_vector = image_data[pointer1][pointer2]
            t = np.array([random_vector[0], random_vector[1] - 128, random_vector[2] - 128])
            t1 = np.dot(transformation_matrix, t)

            image[pointer1][pointer2] = np.array([t1[2], t1[1], t1[0]])

    return image


def abs_bin(x):
    x = int(x)
    y = abs(x)
    ans = bin(y)
    ans = ans[2:]
    n = len(ans)

    if n == 1 and ans[0] == '0':
        return ''

    ans1 = ''

    if y != x:
        for pointer1 in range(n):
            if ans[pointer1] == '0':
                ans1 += '1'
            else:
                ans1 += '0'
    else:
        ans1 = ans

    return ans1


def abs_bin_to_int(inp):
    okk = 1

    n = len(inp)
    if (not n):
        return 0
    if inp[0] == '0':
        okk = 0

    inp1 = ''
    if okk == 0:
        for pointer1 in range(n):
            if inp[pointer1] == '1':
                inp1 += '0'
            else:
                inp1 += '1'
    else:
        inp1 = inp

    ans = int(inp1, 2)

    if okk == 0:
        ans = -ans

    return ans


def constant_freq_bits(zig_vec):
    parent = huffman_dc_coeff(zig_vec)
    data_dc = gen_huffman(parent)
    return data_dc, parent


def ac_freq_bits(zig_vec):
    parent = huffman_ac_coeff(zig_vec)
    data_ac = gen_huffman(parent)
    return data_ac, parent


image_data = cv2.imread('Lena.png')
image_data = color_1_to_2(image_data)

[h, w, _] = image_data.shape

height = h
width = w

row_block = math.ceil(h / 8)
row_block = np.int32(row_block)

col_block = math.ceil(w / 8)
col_block = np.int32(col_block)

temp_height = 8 * row_block

temp_width = 8 * col_block

temp_matrix = np.zeros((temp_height, temp_width, 3))

for pointer1 in range(h):
    for pointer2 in range(w):
        temp_matrix[pointer1][pointer2] = image_data[pointer1][pointer2]

qdct_Y = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77],
                   [24, 35, 55, 64, 81, 104, 113, 92], [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]])

qdct_cr = np.array([[17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
                    [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99]])

qdct_cb = np.array([[17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
                    [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99]])

Y1 = np.zeros((temp_height, temp_width))
cr1 = np.zeros((temp_height, temp_width))
cb1 = np.zeros((temp_height, temp_width))

for pointer1 in range(temp_height):
    for pointer2 in range(temp_width):
        Y1[pointer1][pointer2], cr1[pointer1][pointer2], cb1[pointer1][pointer2] = temp_matrix[pointer1][pointer2]


def image_to_zig(image_data, quantization_table):
    n = image_data.shape[0]
    m = image_data.shape[1]
    zig_vec = []
    nh = n // 8
    mh = m // 8

    for pointer1 in range(nh):
        for pointer2 in range(mh):
            block = np.zeros((8, 8))

            for k1 in range(8):
                for k2 in range(8):
                    block[k1][k2] = image_data[8 * pointer1 + k1][8 * pointer2 + k2]

            block = dct(block)
            block = np.divide(block, quantization_table).astype(int)
            out__temp = block_to_zigzag(block)
            zig_vec.append(np.array(out__temp))

    zig_vec = np.array(zig_vec)
    return zig_vec


sh = image_to_zig(Y1, qdct_Y)

sh2 = image_to_zig(cr1, qdct_cr)
sh3 = image_to_zig(cb1, qdct_cb)


def rle_rep(zig_vec):
    dc__vector = []
    ac__vector = []
    n = zig_vec.shape[0]
    dci_1 = 0

    for pointer1 in range(n):
        random_vector = rle(zig_vec[pointer1], dci_1)
        dci_1 = zig_vec[pointer1][0]
        dc__vector.append(random_vector[0])
        for pointer2 in range(1, random_vector.shape[0]):
            ac__vector.append(random_vector[pointer2])

    dc__vector = np.array(dc__vector, dtype='object')
    ac__vector = np.array(ac__vector, dtype='object')

    return dc__vector, ac__vector


def coded_symbols(zig_vec):
    dc__vector, ac__vector = rle_rep(zig_vec)

    data_dc, parents_dc = constant_freq_bits(dc__vector)
    data_ac, parents_ac = ac_freq_bits(ac__vector)

    n = dc__vector.shape[0]
    m = ac__vector.shape[0]

    string_data = ''

    pointer1 = 0
    pointer2 = 0

    okk = 1

    for k in range(n + m):
        if pointer2 < m:
            if okk != 0:

                t = data_dc[bitstring(dc__vector[pointer1])] + abs_bin(dc__vector[pointer1])
                string_data += t
                # print("sym",string_data)
                # print("sym: ",t)
                okk = 0
                pointer1 += 1
            else:

                t = data_ac[(ac__vector[pointer2][0], ac__vector[pointer2][1])]
                if (t != data_ac[(0, 0)]):
                    t += abs_bin(ac__vector[pointer2][2])
                    string_data += t

                    pointer2 += 1
                else:
                    string_data += t

                    pointer2 += 1
                    okk = 1
    # print(string_data)


    return string_data, parents_dc, parents_ac


str1, par_dc1, par_ac1 = coded_symbols(sh)
str2, par_dc2, par_ac2 = coded_symbols(sh2)
str3, par_dc3, par_ac3 = coded_symbols(sh3)

compressed_data = open("Y.txt", "w")

compressed_data.write(str1)

compressed_data.close()

compressed_data = open("Cr.txt", "w")

compressed_data.write(str2)

compressed_data.close()

compressed_data = open("Cb.txt", "w")

compressed_data.write(str3)

compressed_data.close()

new_size = len(str1) + len(str2) + len(str3)
prev_size = 24 * height * width

compression_ratio = new_size / prev_size
print(compression_ratio)

compressed_data = open("Y.txt", "r")

str1 = compressed_data.read()

compressed_data.close()

compressed_data = open("Cr.txt", "r")

str2 = compressed_data.read()

compressed_data.close()

compressed_data = open("Cb.txt", "r")

str3 = compressed_data.read()

compressed_data.close()


def decoder_out(string_data, parents_dc, parents_ac):
    n = len(string_data)
    symbol_dc = {}
    symbol_ac = {}

    dc__vector = []
    ac__vector = []

    pointer1 = 0
    okk = 1

    for k in range(3 * n):
        if pointer1 == n:
            break
        if okk != 0:
            pointer3, pointer2 = bin_codes_to_rle(parents_dc, string_data[pointer1:], symbol_dc)
            pointer2 += pointer1

            x = abs_bin_to_int(string_data[pointer2:pointer2 + pointer3])
            dc__vector.append(x)
            okk = 0
            pointer1 = pointer2 + pointer3
        else:
            pointer3, pointer2 = bin_codes_to_rle(parents_ac, string_data[pointer1:], symbol_dc)
            if ((pointer3[0], pointer3[1]) == (0, 0)):
                okk = 1
                ac__vector.append(pointer3)
                pointer2 += pointer1
                pointer1 = pointer2
            else:
                nh = pointer3[1]
                pointer2 += pointer1
                x = abs_bin_to_int(string_data[pointer2:pointer2 + nh])
                ac__vector.append((pointer3[0], pointer3[1], x))
                pointer1 = pointer2 + nh

    dc__vector = np.array(dc__vector, dtype='object')
    ac__vector = np.array(ac__vector, dtype='object')

    return dc__vector, ac__vector


code_dc_temp1, code_ac_temp1 = decoder_out(str1, par_dc1, par_ac1)
code_dc_temp2, code_ac_temp2 = decoder_out(str2, par_dc2, par_ac2)
code_dc_temp3, code_ac_temp3 = decoder_out(str3, par_dc3, par_ac3)


def merge_fun(dc__vector, ac__vector):
    n = dc__vector.shape[0]
    m = ac__vector.shape[0]

    okk = 1
    zig_vec = []

    out__temp = []
    pointer1 = 0
    pointer2 = 0

    for k in range(5 * n):
        pointer1 = pointer1 + 1
    pointer1 = 0

    for k in range(n + m):
        if pointer2 == m:
            break
        if okk != 0:
            out__temp.append(dc__vector[pointer1])
            pointer1 += 1
            okk = 0
        else:
            out__temp.append(ac__vector[pointer2])
            if (ac__vector[pointer2][0], ac__vector[pointer2][1]) == (0, 0):
                okk = 1
                zig_vec.append(np.array(out__temp, dtype='object'))
                out__temp = []
            pointer2 += 1

    zig_vec = np.array(zig_vec, dtype='object')
    return zig_vec


merge_1 = merge_fun(code_dc_temp1, code_ac_temp1)
merge_2 = merge_fun(code_dc_temp2, code_ac_temp2)
merge_3 = merge_fun(code_dc_temp3, code_ac_temp3)


def rec2(merged_data, dci_1=0):
    n = merged_data.shape[0]
    sh = []

    sh.append(merged_data[0] + dci_1)

    for pointer1 in range(1, n - 1):
        cnt = merged_data[pointer1][0]

        for k in range(cnt):
            sh.append(0)
        sh.append(merged_data[pointer1][2])

    m = len(sh)
    m = 64 - m
    for pointer1 in range(m):
        sh.append(0)
    return sh


def rec(zig_vec):
    n = zig_vec.shape[0]
    sh = np.zeros((n, 64))

    dci_1 = 0
    for pointer1 in range(n):
        sh[pointer1] = rec2(zig_vec[pointer1], dci_1)
        dci_1 = sh[pointer1][0]

    return sh


out_zig1 = rec(merge_1)
out_zig2 = rec(merge_2)
out_zig3 = rec(merge_3)


def zigzag_to_matrix(v_zig, qdct):
    n = temp_height
    m = temp_width
    p = v_zig.shape[0]
    nh = n // 8
    mh = m // 8

    image_data = np.zeros((n, m))
    c = 0

    for pointer1 in range(nh):
        for pointer2 in range(mh):
            out__temp = v_zig[c]
            c += 1
            block = zigzag_to_block(out__temp)
            block = np.multiply(block, qdct)
            block = idct(block)

            for k1 in range(8):
                for k2 in range(8):
                    if 8 * pointer1 + k1 < n and 8 * pointer2 + k2 < m:
                        image_data[8 * pointer1 + k1][8 * pointer2 + k2] = block[k1][k2]

    return image_data


Y1 = zigzag_to_matrix(out_zig1, qdct_Y)
cr1 = zigzag_to_matrix(out_zig2, qdct_cr)
cb1 = zigzag_to_matrix(out_zig3, qdct_cb)

out_matrix_temp = np.zeros((temp_height, temp_width, 3))

for pointer1 in range(temp_height):
    for pointer2 in range(temp_width):
        out_matrix_temp[pointer1][pointer2] = Y1[pointer1][pointer2], cr1[pointer1][pointer2], cb1[pointer1][pointer2]

out_matrix = np.zeros((height, width, 3))

for pointer1 in range(h):
    for pointer2 in range(w):
        out_matrix[pointer1][pointer2] = out_matrix_temp[pointer1][pointer2]

out_image = color_2_to_1(out_matrix)

cv2.imwrite('out_image.jpg', out_image)


