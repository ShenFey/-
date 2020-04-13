import cv2 as cv
import numpy as np

def pretreatment(test_image):

    # 灰度化。
    gray_image = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)

    # 高斯模糊，制造边缘模糊效果，方便滤除噪声
    gray_gaussianblur = cv.GaussianBlur(gray_image, (3, 3), 0)
    cv.imwrite('gaussianblur.jpg', gray_gaussianblur)

    # 再进行图片标准化,将图片数组的数值统一到一定范围内。函数的参数
    # 依次是：输入数组，输出数组，最小值，最大值，标准化模式。
    cv.normalize(gray_gaussianblur, gray_gaussianblur, 0, 255, cv.NORM_MINMAX)

    # 使用阈值对图片进行二值化 阈值调试为130
    thresh, result = cv.threshold(gray_gaussianblur, 100, 255, cv.THRESH_BINARY)
    cv.imwrite('binary.jpg', result)
    res_inv = cv.bitwise_not(result)
    cv.imwrite('res_inv.jpg', res_inv)

    # 再进行图片标准化,将图片数组的数值统一到一定范围内。函数的参数
    # 依次是：输入数组，输出数组，最小值，最大值，标准化模式。
    cv.normalize(res_inv, res_inv, 0, 255, cv.NORM_MINMAX)


    # 使用投影算法对图像投影。
    def horizontal_projection(image):
        """水平投影
        :param image:
        :return:
        """
        hProjection = np.zeros(image.shape, np.uint8)
        # 图像高与宽
        h, w = image.shape
        # 长度 = 图像高度的一维数组
        height = [0] * h
        # 循环每一行白色像素的个数 字部分
        for y in range(h):
            for x in range(w):
                if image[y, x] == 255:
                    height[y] += 1
        # 绘制水平投影图像观察
        for y in range(h):
            for x in range(height[y]):
                hProjection[y, x] = 255


        return height,hProjection


    def vertical_projection(image):
        """垂直投影

        :param image:
        :return:
        """
        vProjection = np.zeros(image.shape, np.uint8)
        # 图像高与宽
        (h, w) = image.shape
        # 长度与图像宽度一致的数组
        weight = [0] * w
        # 循环统计每一列白色像素的个数
        for x in range(w):
            for y in range(h):
                if image[y, x] == 255:
                    weight[x] += 1
        # 绘制垂直平投影图像
        for x in range(w):
            for y in range(h - weight[x], h):
                vProjection[y, x] = 255

        return weight,vProjection


    # h_projection 中的黑色部分就是可切割点，最下面一段白色就是身份证号码部分

    h_projection,hProjection = horizontal_projection(res_inv)
    # print('len(h_projection):', len(h_projection))
    # 图像高与宽
    h, w = res_inv.shape


    # print('h:{},w:{}'.format(h, w))


    position = []
    # 切割的起始和结束:遇到白色开始，遇到黑色停止
    hstart = 0
    H_Start = []
    H_End = []
    # 根据水平投影获取垂直分割位置
    for i in range(len(h_projection)):
        # 如果是黑色的，且还未开始切割
        if h_projection[i] > 0 and hstart == 0:
            # 保存身份证号码部分的起始位置：所有第一个白色位置

            H_Start.append(i)
            hstart = 1
        if h_projection[i] <= 0 and hstart == 1:
            # 所有第一个黑色位置
            H_End.append(i)
            hstart = 0
    # print('H_Start[1]:{},H_End[1]:{}'.format(H_Start[-1], H_End[-1]))  # [393, 2072] [1852, 2205]
    # 分割行，分割之后再进行列分割并保存分割位置
    # 获取行图像 (start - end ) * w 的矩形
    cropImg = res_inv[H_Start[-1]:H_End[-1], 0:w]

    # 对行图像进行垂直投影
    w_projection, vProjection= vertical_projection(cropImg)
    wstart = 0
    wend = 0  # 字遍历未结束标志
    w_start = 0
    w_end = 0
    for j in range(len(w_projection)):
        # 如果是白色，就是有字，记录位置后开始继续遍历
        if w_projection[j] > 0 and wstart == 0:
            w_start = j  # 记录字开始的位置
            wstart = 1
            wend = 0
        # 直到黑色，一个字遍历结束
        if w_projection[j] <= 0 and wstart == 1:
            w_end = j  # 记录字结束的位置
            wstart = 0
            wend = 1
        # 如果一个字遍历结束了
        if wend == 1:
            # 记录宽度起始位置，和高度起始位置
            position.append([w_start, H_Start[-1], w_end, H_End[-1]])
            print('w_start:{}, w_end:{}'.format(w_start, w_end))
            wend = 0

    # cv.namedWindow("vProjection", 0)  # flag = 0 ,默认窗口大小可以改变
    # cv.resizeWindow("vProjection", 500, 500)
    # cv.imshow('vProjection', vProjection)
    # cv.namedWindow("hProjection", 0)  # flag = 0 ,默认窗口大小可以改变
    # cv.resizeWindow("hProjection", 500, 500)
    # cv.imshow('hProjection', hProjection)
    # cv.namedWindow("cropImg", 0)  # flag = 0 ,默认窗口大小可以改变
    # cv.resizeWindow("cropImg", w, H_End[-1] - H_Start[-1])
    # cv.imshow('cropImg', cropImg)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return position


