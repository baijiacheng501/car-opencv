import pickle

import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import operator
import collections

def cv_show(name,img):
    # 封装显示函数
    h, w = img.shape[:2]
    max_width = 1800
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, None, fx=scale, fy=scale)
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# step1：过滤掉背景select_rgb_white_yellow()
def select_rgb_white_yellow(image):
    # 定义白色的像素范围，rgb三个通道都在120-255之间
    lower = np.uint8([120,120,120])
    upper = np.uint8([255,255,255])

    # 创建mask：在范围内的像素变成白色（255),否则为黑色（0）
    white_mask = cv2.inRange(image, lower, upper)

    mask_img = cv2.bitwise_and(image,image,mask=white_mask)
    return mask_img,white_mask


# step5:区域标定
def select_region(image):
    rows, cols = image.shape[:2]  # 获取图片的高度和宽度

    pt_1 = [cols * 0.065, rows * 0.92]  # 左下
    pt_2 = [cols * 0.065, rows * 0.67]  # 左上
    pt_3 = [cols * 0.36, rows * 0.47]  # 中左
    pt_4 = [cols * 0.575, rows * 0.11]  # 中右
    pt_5 = [cols * 0.9, rows * 0.12]  # 右上
    pt_6 = [cols * 0.9, rows * 0.92]  # 右下

    # 把点转换为numpy数组
    vertices = np.array([[pt_1, pt_2, pt_3, pt_4, pt_5, pt_6]], dtype=np.int32)

    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255, 255, 255))

    roi_image = cv2.bitwise_and(image, mask)

    point_img = image.copy()
    if len(point_img.shape) == 2:
        point_img = cv2.cvtColor(point_img, cv2.COLOR_GRAY2BGR)

    for point in vertices[0]:
        cv2.circle(point_img, (point[0], point[1]), 10, (0, 0, 255), 4)
    return roi_image, point_img

# 霍夫变换直线
# 得到了一张只有停车场区域内边缘的图片（ roi_image ）。但这些边缘是 零散的像素点 ，
# 我们需要把它们 连接成直线 ，因为停车位的分隔线就是直线。
# 简单理解：
# - 图片中的一条直线 = 很多连续的白色像素点
# - 霍夫变换的作用 = 把这些零散的白色像素点，拼成一条条直线
# - 它会返回每条直线的两个端点坐标 (x1, y1, x2, y2)
def hough_lines(image):
    # 使用概率霍夫变换检测直线：
    # 参数：
    #     image：输入图像（边缘检测后的二值图）
    # 返回：
    #     lines：检测到的直线列表，每条直线是[[x1，y1，x2，y2]]
    lines = cv2.HoughLinesP(
        image,rho=0.1,#距离精度，越小越准确
        theta=np.pi/180,#角度精度（弧度）
        threshold=8,#累加器阈值，越高检测到的线越少
        minLineLength=5,#线的最小长度，比这个短的忽略掉
        maxLineGap=10#两条线段之间的最大间隔，小于这个值的认定为同一条直线
    )
    return lines

# step8:检测每列停车位，画矩形框
def identity_blocks(image,Lines,mask_copy=True):
    # 把检测到的直线聚类成列，每一列生成一个矩形框
    # 参数：
    #     image：原始图片（用于在上面画矩形）
    #     lines：霍夫变换检测到的直线
    # 返回：
    #     new——image：画了矩形的图片
    #     rects：每列的矩形坐标{0：[x1，y1，x2，y2]1：。。。。}
    if mask_copy:
        new_image = image.copy()

    # 1过滤掉直线
    stayed_lines = []
    for line in Lines:
        for x1,y1,x2,y2 in line:
            # 只保留：接近水平+长度在25-55像素之间的线
            # abs（y2-y1）<=1 ->近似水平
            # abs(x2-x1) >=25 不能太短
            # abs(x2 - x1) <= 55 不能太长
            if abs(y2 - y1) <= 3 and abs(x2 - x1) >=25 and abs(x2 - x1) <=55:
                stayed_lines.append((x1,y1,x2,y2))
    print(f'过滤后剩余{len(stayed_lines)}条')

    # 2、排序
    # 把x坐标相近的直线归为同一类，把直线按位置排好，方便后续聚类。
    list1 = sorted(stayed_lines, key=operator.itemgetter(0,1))

    # 3:聚类
    clusters = collections.defaultdict(list)#自动创建列表的字典
    dIndex = 0 #当前是第几列
    clus_dist = 18 #列之间的距离阈值

    for i in range(len(list1) - 1):
        distance = abs(list1[i+1][0] - list1[i][0])
        if distance <= clus_dist:
            # 距离很近，归属于同一列
            clusters[dIndex].append(list1[i])
            clusters[dIndex].append(list1[i+1])
        else:
            # 距离很远 -》新的一列
            dIndex += 1
    print(f'检测到{dIndex+1}列')

    # 4生成矩形
    rects = {}
    i = 0
    for key in clusters:
        all_list = clusters[key]
        cleaned = list(set(all_list))#去重
        # 至少要有5条线才算一列
        if len(cleaned) > 16:
            cleaned = sorted(cleaned,key = lambda tup:tup[1])
            avg_y1 = cleaned[0][1]#最上面的线的y1
            avg_y2 = cleaned[-1][1]#最下面的线的y2
            # 如果高度太小，则跳过
            if abs(avg_y2 - avg_y1) <= 15:
                continue

            avg_x1 = 0
            avg_x2 = 0
            for tup in cleaned:
                avg_x1 += tup[0]
                avg_x2 += tup[2]
            avg_x1 = avg_x1 / len(cleaned)
            avg_x2 = avg_x2 / len(cleaned)

            rects[i] = [avg_x1,avg_y1,avg_x2,avg_y2]
            i +=1

        print(f'有效列数：{len(rects)}')

        # 5:画矩形框
    buff = 4#矩形框向外扩展7个像素
    for key in rects:
        tup_topLeft = (int(rects[key][0] - buff),int(rects[key][1]))
        tup_bottomRight = (int(rects[key][2] + buff),int(rects[key][3]))
        cv2.rectangle(new_image,tup_topLeft,tup_bottomRight,(0,255,0),3)

    return new_image,rects

def rect_finetune(image,rects,make_copy = True):
    """
    手动微调矩形框位置，让框更准确
    参数：
        image：原始图片
        rects：identity——blocks返回的矩形字典
    返回：
        new_image:画了微调后矩形的图片
        new_rects:微调后的矩形字典
    :param image:
    :param rects:
    :param make_copy:
    :return:
    """
    if make_copy:
        new_image = image.copy()
    new_rects = rects.copy()#复制一份，不修改原数据
    # 根据经验手动调整各列坐标
    # 格式：new_rects[列号][坐标索引]+=偏移量
    # 坐标索引：0=x1，1=y1，2=x2，3=y2

    for k in new_rects:
        if k ==0:#第一列
            new_rects[k][1] -= 1#y1向上移动10像素
            new_rects[k][3] += 0#y2向下移动10像素
            new_rects[k][0] -= 0
            new_rects[k][2] += 0
        elif k ==1:
            new_rects[k][1] -= 2  # y1向上移动10像素
            new_rects[k][3] -= 20  # y2向下移动10像素
            new_rects[k][0] -= 10
            new_rects[k][2] += 13
        elif k ==2:
            new_rects[k][1] -= 5  # y1向上移动10像素
            new_rects[k][3] += 5  # y2向下移动10像素
            new_rects[k][0] -= 10
            new_rects[k][2] += 20
        elif k ==3:
            new_rects[k][1] -= 10  # y1向上移动10像素
            new_rects[k][3] -= 10  # y2向下移动10像素
            new_rects[k][0] -= 15#左
            new_rects[k][2] += 10
        elif k ==4:
            new_rects[k][1] -= 45  # y1向上移动10像素
            new_rects[k][3] -= 13  # y2向下移动10像素
            new_rects[k][0] -= 10#左
            new_rects[k][2] += 10
        elif k ==5:
            new_rects[k][1] -= 53  # y1向上移动10像素
            new_rects[k][3] += 3  # y2向下移动10像素
            new_rects[k][0] -= 5#左
            new_rects[k][2] += 10
        elif k ==6:
            new_rects[k][1] -= 8   # y1向上移动10像素
            new_rects[k][3] -= 18  # y2向下移动10像素
            new_rects[k][0] -= 25#左
            new_rects[k][2] += 0
        elif k ==7:
            new_rects[k][1] -= 125  # y1向上移动10像素
            new_rects[k][3] -= 12  # y2向下移动10像素
            new_rects[k][0] -= 12#左
            new_rects[k][2] += 5
        elif k ==8:
            new_rects[k][1] -= 22  # y1向上移动10像素
            new_rects[k][3] += 31  # y2向下移动10像素
            new_rects[k][0] -= 15#左
            new_rects[k][2] += 5
        elif k ==9:
            new_rects[k][1] += 3  # y1向上移动10像素
            new_rects[k][3] += 17  # y2向下移动10像素
            new_rects[k][0] -= 10#左
            new_rects[k][2] += 10
        elif k ==10:
            new_rects[k][1] += 48  # y1向上移动10像素
            new_rects[k][3] -= 3  # y2向下移动10像素
            new_rects[k][0] -= 10#左
            new_rects[k][2] += 10
        elif k == len(new_rects)-1:#最后一列
            new_rects[k][1] -= 170  # y1向上移动10像素
            new_rects[k][3] += 33  # y2向下移动10像素
            new_rects[k][0] -= 5  # 左
            new_rects[k][2] += 3
    # 画出微调后的矩形框
    for key in new_rects:
        tup_topLeft = (int(new_rects[key][0]), int(new_rects[key][1]))
        tup_bottomRight = (int(new_rects[key][2]), int(new_rects[key][3]))
        cv2.rectangle(new_image, tup_topLeft, tup_bottomRight, (255, 0, 0), 3)
    return new_image,new_rects
# step10:生成车位字典
def draw_parking(image,rects,make_copy = True):
    """
    把每列矩形分割成单个停车位，生成坐标字典
    参数:image：原始图片
        recte：微调后的矩形字典
    返回：
        new_IMAGE:画了所有车位框的图片
        spot_dict:车位字典，键是坐标元组，值是车位编号
    """
    if make_copy:
        new_image = image.copy()
    gap = 15.4#每个车位的高度（像素）
    spot_dict = {}#车位字典

    #遍历每一列
    for key in rects:
        x1,y1,x2,y2 = rects[key]

        #计算每一列有多少个车位
        num_splits = int(abs(y2-y1)/gap)

        #分割成单个车位
        for i in range(num_splits+1):
            y = int(y1+i*gap)
            #第一列和最后一列只有一排
            if key ==0 or key ==len(rects)-1:
                # 车位坐标：（x1，y，x2，y+gap）
                spot_dict[(x1,y,x2,y + gap)] = len(spot_dict)+1
                #画出车位框
                cv2.rectangle(new_image,(int(x1),int(y)), (int(x2),y + int(gap)), (0, 255, 0), 2)
                #在车位上标出序号
                cv2.putText(new_image,str(len(spot_dict)),(int(x1) + 5,y + int(gap) // 2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                #中间列各有两派，左右各一排
            else:
                x_mid = int((x1+x2)/2)#中间x坐标
                #左排车位
                spot_dict[(x1,y,x_mid,y + gap)] = len(spot_dict)+1
                cv2.rectangle(new_image,(int(x1),int(y)), (int(x_mid),y + int(gap)), (0, 255, 0), 2)
                cv2.putText(new_image,str(len(spot_dict)),(int(x1) + 5,y + int(gap) // 2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

                #右排车位
                spot_dict[(x_mid,y,x2,y + gap)] = len(spot_dict)+1
                cv2.rectangle(new_image,(x_mid,y),(int(x2),y + int(gap)), (0, 255, 0), 2)
                cv2.putText(new_image,str(len(spot_dict)),(x_mid + 5,y + int(gap) // 2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    print(f'生成了{len(spot_dict)}个车位')
    return new_image,spot_dict

#step 11:过滤掉无效车位：有些检测到的区域不是真正的停车位（比如通道、角落等），需要手动排除。
def filter_invalid_spots(spot_dict):
    """
    手动过滤无效车位
    参数：
        spot_dict: 原始车位字典 {(x1,y1,x2,y2): 编号}
    返回：
        valid_spots_dict: 过滤后的车位字典 {(x1,y1,x2,y2): 新编号}
    """
    #根据经验，这些编号的车位不是真正的停车位
    # invalid_nums = {11,12,34,35,62,63,96,98,136,138,188,190,252,253,326,327,470,471,534,535,573}
    invalid_nums = {2,11,12,34,35,38,39,62,63,96,98,99,136,137,138,140,141,190,192,193,254,255,328,329,332,333,258,259,472,473,536,537,575}
    vali_spots_dict = {}
    cur_idx = 1
    for coords, spot_num in spot_dict.items():
        if spot_num not in invalid_nums:
            vali_spots_dict[coords] = cur_idx
            cur_idx += 1
    print(f'过滤后剩余{len(vali_spots_dict)}个有效车位')
    return vali_spots_dict

#step12:保存车位字典：把车位字典保存到文件，下次使用时直接读取，不需要重新检测。
def save_spot_dict(spot_dict,filename='spot_dict.pickle'):
    """
    保存车位字典到文件
    参数：
        spot_dict：车位字典
        filename：保存的文件名，默认spot_dict.pickle
    """
    with open(filename,'wb') as handle:
        pickle.dump(spot_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'车位字典已保存到{filename}')

#step13：加载车位字典
def load_spot_dict(filename='spot_dict.pickle'):
    """
    从文件加载车位字典
    参数：
        filename：文件名
    返回：
        spot_dict：加载的车位字典
    :param filename:
    :return:
    """
    with open(filename,'rb') as handle:
        spot_dict = pickle.load(handle)
    print(f'车位字典已从{filename}加载，共{len(spot_dict)}个车位')
    return spot_dict

jpg_files = glob.glob('test_images/*.jpg')
for idx,img_path in enumerate(jpg_files):
    image = cv2.imread(img_path)
    cv_show(f"{idx+1}-1.original",image)
    # 背景过滤：select_rgb_white_yellow
    masked_img,mask = select_rgb_white_yellow(image)
    # cv_show(f"{idx+1}-2.mask_img",masked_img)
    # cv_show(f"{idx+1}-3.mask",mask)
    # step3:灰度值转换
    gray_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    cv_show(f"{idx+1}-4.gray",gray_img)
    # step4：canny边缘检测
    edges = cv2.Canny(gray_img, 50, 200, apertureSize = 3)
    cv_show(f"{idx+1}-5.edges",edges)
    roi_image,point_img = select_region(edges)
    cv_show(f"{idx+1}-6.roi",roi_image)
    cv_show(f"{idx+1}-7.roi-1",point_img)

    # step7
    lines = hough_lines(roi_image)

    line_img = image.copy()
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(line_img,(x1,y1),(x2,y2),(0,0,255),2)
    cv_show(f"{idx+1}-8.line",line_img)
    print(f'检测到{len(lines)}')

    # step8: 检测停车位列
    if lines is not None:
        new_image, rects = identity_blocks(image, lines)
        cv_show(f"{idx + 1}-9.rect", new_image)
        print(f'矩形框：{rects}')

        # step9: 矩形框微调
        finetune_image, new_rects = rect_finetune(image, rects)
        cv_show(f"{idx + 1}-10.finetune", finetune_image)
        print(f'微调后矩形框：{new_rects}')

    #step10:生成车位字典
    parking_image,spot_dict = draw_parking(image,new_rects)
    cv_show(f"{idx + 1}-11.parking", parking_image)
    print(f'车位字典{len(spot_dict)}')

    #step 11:过滤掉无效车位
    valid_spots_dict = filter_invalid_spots(spot_dict)
    print(f'有效车位{len(valid_spots_dict)}')

    #step13:保存车位字典
    save_spot_dict(valid_spots_dict)