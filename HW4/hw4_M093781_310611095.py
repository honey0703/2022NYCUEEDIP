import cv2
import copy
import numpy as np
from cv2 import dct, idct
from pywt import dwt2, idwt2
from numpy.linalg import svd

class WaterMarkCore:
    def __init__(self, password_img=1, mode='common', processes=None):
        self.block_shape = np.array([4, 4])
        self.password_img = password_img
        self.d1, self.d2 = 80, 20  # Larger d1/d2, more robust, but higher distortion as well.

        # init data
        self.img, self.img_YUV = None, None  # self.img_YUV padding with white, let it be odd number.
        self.ca, self.hvd, = [np.array([])] * 3, [np.array([])] * 3  # The result od dwt.
        self.ca_block = [np.array([])] * 3  # The result of 4 blocking in 3 channel.
        self.ca_part = [np.array([])] * 3  # Let ca % 4 == 0.
        self.wm_size, self.block_num = 0, 0  # The length of watermark and the stored watermark limit size.
        self.pool = AutoPool(mode=mode, processes=processes)
        self.alpha = None  # for transparent image

    def read_img_arr(self, img):
        # if image is transparent
        self.alpha = None
        if img.shape[2] == 4:
            if img[:, :, 3].min() < 255:
                self.alpha = img[:, :, 3]
                img = img[:, :, :3]

        # read img -> YUV -> padding with white -> blocking
        self.img = img.astype(np.float32)
        self.img_shape = self.img.shape[:2]

        # padding with white
        self.img_YUV = cv2.copyMakeBorder(cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV),
                                          0, self.img.shape[0] % 2, 0, self.img.shape[1] % 2,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))

        self.ca_shape = [(i + 1) // 2 for i in self.img_shape]  # the shape of ca

        self.ca_block_shape = (self.ca_shape[0] // self.block_shape[0], self.ca_shape[1] // self.block_shape[1],
                               self.block_shape[0], self.block_shape[1])
        strides = 4 * np.array([self.ca_shape[1] * self.block_shape[0], self.block_shape[1], self.ca_shape[1], 1])

        for channel in range(3):
            self.ca[channel], self.hvd[channel] = dwt2(self.img_YUV[:, :, channel], 'haar')
            # blocking into 4d
            self.ca_block[channel] = np.lib.stride_tricks.as_strided(self.ca[channel].astype(np.float32),
                                                                         self.ca_block_shape, strides)
    def read_wm(self, wm_bit):
        self.wm_bit = wm_bit
        self.wm_size = wm_bit.size

    def init_block_index(self):
        self.block_num = self.ca_block_shape[0] * self.ca_block_shape[1]
        self.part_shape = self.ca_block_shape[:2] * self.block_shape
        self.block_index = [(i, j) for i in range(self.ca_block_shape[0]) for j in range(self.ca_block_shape[1])]

    def block_add_wm(self, arg):
        block, shuffler, i = arg
        wm_1 = self.wm_bit[i % self.wm_size]
        block_dct = dct(block)

        # shuffle
        block_dct_shuffled = block_dct.flatten()[shuffler].reshape(self.block_shape)
        u, s, v = svd(block_dct_shuffled)
        s[0] = (s[0] // self.d1 + 1 / 4 + 1 / 2 * wm_1) * self.d1
        if self.d2:
            s[1] = (s[1] // self.d2 + 1 / 4 + 1 / 2 * wm_1) * self.d2

        block_dct_flatten = np.dot(u, np.dot(np.diag(s), v)).flatten()
        block_dct_flatten[shuffler] = block_dct_flatten.copy()
        return idct(block_dct_flatten.reshape(self.block_shape))

    def embed(self):
        self.init_block_index()

        embed_ca = copy.deepcopy(self.ca)
        embed_YUV = [np.array([])] * 3

        self.idx_shuffle = np.random.RandomState(self.password_img)\
            .random(size=(self.block_num, self.block_shape[0] * self.block_shape[1])).argsort(axis=1)

        for channel in range(3):
            tmp = self.pool.map(self.block_add_wm,
                                [(self.ca_block[channel][self.block_index[i]], self.idx_shuffle[i], i)
                                 for i in range(self.block_num)])

            for i in range(self.block_num):
                self.ca_block[channel][self.block_index[i]] = tmp[i]

            # concatenate 4d to 3d
            self.ca_part[channel] = np.concatenate(np.concatenate(self.ca_block[channel], 1), 1)

            embed_ca[channel][:self.part_shape[0], :self.part_shape[1]] = self.ca_part[channel]
            embed_YUV[channel] = idwt2((embed_ca[channel], self.hvd[channel]), "haar")

        # concatenate 3 channel
        embed_img_YUV = np.stack(embed_YUV, axis=2)
        # delete the extra padding
        embed_img_YUV = embed_img_YUV[:self.img_shape[0], :self.img_shape[1]]
        embed_img = cv2.cvtColor(embed_img_YUV, cv2.COLOR_YUV2BGR)
        embed_img = np.clip(embed_img, a_min=0, a_max=255)

        if self.alpha is not None:
            embed_img = cv2.merge([embed_img.astype(np.uint8), self.alpha])
        return embed_img

    def block_get_wm(self, args):
        block, shuffler = args
        block_dct_shuffled = dct(block).flatten()[shuffler].reshape(self.block_shape)

        u, s, v = svd(block_dct_shuffled)
        wm = (s[0] % self.d1 > self.d1 / 2) * 1
        if self.d2:
            tmp = (s[1] % self.d2 > self.d2 / 2) * 1
            wm = (wm * 3 + tmp * 1) / 4
        return wm

    def extract_raw(self, img):
        self.read_img_arr(img=img)
        self.init_block_index()

        wm_block_bit = np.zeros(shape=(3, self.block_num))

        self.idx_shuffle = np.random.RandomState(self.password_img)\
            .random(size=(self.block_num, self.block_shape[0] * self.block_shape[1])).argsort(axis=1)

        for channel in range(3):
            wm_block_bit[channel, :] = self.pool.map(self.block_get_wm,
                                                     [(self.ca_block[channel][self.block_index[i]], self.idx_shuffle[i])
                                                      for i in range(self.block_num)])
        return wm_block_bit

    def extract_avg(self, wm_block_bit):
        wm_avg = np.zeros(shape=self.wm_size)
        for i in range(self.wm_size):
            wm_avg[i] = wm_block_bit[:, i::self.wm_size].mean()
        return wm_avg

    def extract(self, img, wm_shape):
        self.wm_size = np.array(wm_shape).prod()
        wm_block_bit = self.extract_raw(img=img)
        wm_avg = self.extract_avg(wm_block_bit)
        return wm_avg

class WaterMark:
    def __init__(self, password_wm=1, password_img=1, block_shape=(4, 4), mode='common', processes=None):
        self.bwm_core = WaterMarkCore(password_img=password_img, mode=mode, processes=processes)
        self.password_wm = password_wm
        self.wm_bit = None
        self.wm_size = 0

    def read_img(self, filename=None, img=None):
        if filename is not None:
            img = cv2.imread(filename, flags=cv2.IMREAD_UNCHANGED)
        self.bwm_core.read_img_arr(img=img)
        return img

    def read_wm(self, wm_content, mode='img'):
        wm = cv2.imread(filename=wm_content, flags=cv2.IMREAD_GRAYSCALE)
        self.wm_bit = wm.flatten() > 128
        self.wm_size = self.wm_bit.size
        # shuffle
        np.random.RandomState(self.password_wm).shuffle(self.wm_bit)
        self.bwm_core.read_wm(self.wm_bit)

    def embed(self, filename=None):
        embed_img = self.bwm_core.embed()

        if filename is not None:
            cv2.imwrite(filename, embed_img)
        return embed_img

    def extract(self, filename=None, embed_img=None, wm_shape=None, out_wm_name=None, mode='img'):
        embed_img = cv2.imread(filename, flags=cv2.IMREAD_COLOR)
        self.wm_size = np.array(wm_shape).prod()
        wm_avg = self.bwm_core.extract(img=embed_img, wm_shape=wm_shape)

        # de shuffle：
        wm = self.extract_decrypt(wm_avg=wm_avg)
        # save img：
        wm = 255 * wm.reshape(wm_shape[0], wm_shape[1])
        cv2.imwrite(out_wm_name, wm)
        return wm

    def extract_decrypt(self, wm_avg):
        wm_index = np.arange(self.wm_size)
        np.random.RandomState(self.password_wm).shuffle(wm_index)
        wm_avg[wm_index] = wm_avg.copy()
        return wm_avg

# Pool
class CommonPool(object):
    def map(self, func, args):
        return list(map(func, args))

class AutoPool(object):
    def __init__(self, mode, processes):
        self.mode = mode
        self.processes = processes

        if mode == 'vectorization':
            pass
        elif mode == 'cached':
            pass
        elif mode == 'multithreading':
            from multiprocessing.dummy import Pool as ThreadPool
            self.pool = ThreadPool(processes=processes)
        elif mode == 'multiprocessing':
            from multiprocessing import Pool
            self.pool = Pool(processes=processes)
        else:  # common
            self.pool = CommonPool()

    def map(self, func, args):
        return self.pool.map(func, args)

if __name__ == '__main__':
    # ==================== File path setting ===================== #
    # ---- cover and watermark path -----
    input_img = 'input/lena.bmp'
    input_wm = 'input/watermark.png'
    # ---- other attack, output path ----
    input_wm_resize = 'input/watermark_resize.png'
    output_embedded = 'output/embedded.bmp'
    attack_noise = 'attack/noise.bmp'
    attack_blur = 'attack/blur.bmp'
    attack_compress = 'attack/compress.bmp'
    attack_decompress = 'attack/decompress.bmp'    # resize back to origin image size
    output_wm_noise = 'output/wm_noise.png'
    output_wm_blur = 'output/wm_blur.png'
    output_wm_compress = 'output/wm_compress.png'
    # _fianl is after resize and binarize.
    output_wm_noise_final = 'output/wm_noise_final.png'
    output_wm_blur_final = 'output/wm_blur_final.png'
    output_wm_compress_final = 'output/wm_compress_final.png'
    # ============================================================ #

    bwm = WaterMark(password_wm=1, password_img=1)
    # read cover image
    bwm.read_img(filename=input_img)
    # resize -> binary watermark
    wm = cv2.imread(input_wm)
    wm_resize = cv2.resize(wm, (64, 64), interpolation=cv2.INTER_AREA)
    _, thres = cv2.threshold(wm_resize, 128, 255, cv2.THRESH_BINARY)
    cv2.imwrite(input_wm_resize, thres)
    # read watermark
    bwm.read_wm(input_wm_resize)
    # embedding
    bwm.embed(output_embedded)
    wm_shape = cv2.imread(input_wm_resize, flags=cv2.IMREAD_GRAYSCALE).shape

    # attack (modified from TA)
    # level 1 attack: noise
    def gaussian_noise(img, mean=0, sigma=0.1):
        # int -> float (標準化)
        img = img / 255
        # 隨機生成高斯 noise (float + float)
        noise = np.random.normal(mean, sigma, img.shape)
        # noise + 原圖
        gaussian_out = img + noise
        # 所有值必須介於 0~1 之間，超過1 = 1，小於0 = 0
        gaussian_out = np.clip(gaussian_out, 0, 1)

        # 原圖: float -> int (0~1 -> 0~255)
        gaussian_out = np.uint8(gaussian_out * 255)
        # noise: float -> int (0~1 -> 0~255)
        noise = np.uint8(noise * 255)
        return gaussian_out
    img = cv2.imread(output_embedded)
    img_noise = gaussian_noise(img)
    cv2.imwrite(attack_noise, img_noise)

    # level 2 attack: blurring
    img_blur = cv2.GaussianBlur(img, (7, 7), 2)
    cv2.imwrite(attack_blur, img_blur)

    # level 3 attack: compression
    img_compress = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
    cover_shape = cv2.imread(input_img, flags=cv2.IMREAD_GRAYSCALE).shape
    restore = cv2.resize(img_compress, cover_shape, interpolation=cv2.INTER_AREA)
    cv2.imwrite(attack_compress, img_compress)
    cv2.imwrite(attack_decompress, restore)

    # extract
    bwm1 = WaterMark(password_wm=1, password_img=1)
    extract_list = [(attack_noise, output_wm_noise),
                    (attack_blur, output_wm_blur),
                    (attack_decompress, output_wm_compress)]
    for (_in, _out) in extract_list:
        bwm1.extract(_in, wm_shape=wm_shape, out_wm_name=_out, mode='img')

    # resize watermark
    poprocess_list = [(output_wm_noise,output_wm_noise_final),
                      (output_wm_blur, output_wm_blur_final),
                      (output_wm_compress, output_wm_compress_final)]
    for (_in, _out) in poprocess_list:
        img = cv2.imread(_in)
        tmp = cv2.resize(img, wm_shape, interpolation=cv2.INTER_AREA)
        _, thres = cv2.threshold(tmp, 128, 255, cv2.THRESH_BINARY)
        cv2.imwrite(_out, thres)