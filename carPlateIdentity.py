import os
import sys
import cv2
import numpy as np
import tensorflow as tf

char_table = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              '川', '鄂', '赣', '甘', '贵','桂', '黑', '沪', '冀', '津', '京', '吉', '辽', '鲁', '蒙', '闽', '宁', '青', '琼',
              '陕', '苏', '晋','皖', '湘', '新', '豫', '渝', '粤', '云', '藏', '浙']

def hist_image(img):
    assert img.ndim == 2
    hist = [0 for i in range(256)]
    img_h, img_w = img.shape[0], img.shape[1]

    for row in range(img_h):
        for col in range(img_w):
            hist[img[row, col]] += 1
    p = [hist[n] / (img_w * img_h) for n in range(256)]
    p1 = np.cumsum(p)
    for row in range(img_h):
        for col in range(img_w):
            v = img[row, col]
            img[row, col] = p1[v] * 255
    return img


def find_board_area(img):
    assert img.ndim == 2
    img_h, img_w = img.shape[0], img.shape[1]
    top, bottom, left, right = 0, img_h, 0, img_w
    flag = False
    h_proj = [0 for i in range(img_h)]
    v_proj = [0 for i in range(img_w)]

    for row in range(round(img_h * 0.5), round(img_h * 0.8), 3):
        for col in range(img_w):
            if img[row, col] == 255:
                h_proj[row] += 1
        if flag and h_proj[row] > 12:
            flag = True
            top = row
        if flag and row > top + 8 and h_proj[row] < 12:
            bottom = row
            flag = False

    for col in range(round(img_w * 0.3), img_w, 1):
        for row in range(top, bottom, 1):
            if img[row, col] == 255:
                v_proj[col] += 1
        if flag == False and (v_proj[col] > 10 or v_proj[col] - v_proj[col - 1] > 5):
            left = col
            break
    return left, top, 120, bottom - top - 10


def verify_scale(rotate_rect):
    error = 0.4
    aspect = 4
    min_area = 10 * (10 * aspect)
    max_area = 150 * (150 * aspect)
    min_aspect = aspect * (1 - error)
    max_aspect = aspect * (1 + error)
    theta = 45

    if rotate_rect[1][0] == 0 or rotate_rect[1][1] == 0:
        return False

    r = rotate_rect[1][0] / rotate_rect[1][1]
    r = max(r, 1 / r)
    area = rotate_rect[1][0] * rotate_rect[1][1]
    if min_area < area < max_area and min_aspect < r < max_aspect:
        if ((rotate_rect[1][0] < rotate_rect[1][1] and -90 <= rotate_rect[2] < -(90 - theta)) or
                (rotate_rect[1][1] < rotate_rect[1][0] and -theta < rotate_rect[2] <= 0)):
            return True
    return False


def img_Transform(car_rect, image):
    img_h, img_w = image.shape[:2]
    rect_w, rect_h = car_rect[1][0], car_rect[1][1]
    angle = car_rect[2]

    return_flag = False
    if car_rect[2] == 0:
        return_flag = True
    if car_rect[2] == -90 and rect_w < rect_h:
        rect_w, rect_h = rect_h, rect_w
        return_flag = True
    if return_flag:
        car_img = image[int(car_rect[0][1] - rect_h / 2):int(car_rect[0][1] + rect_h / 2),
                  int(car_rect[0][0] - rect_w / 2):int(car_rect[0][0] + rect_w / 2)]
        return car_img

    car_rect = (car_rect[0], (rect_w, rect_h), angle)
    box = cv2.boxPoints(car_rect)

    heigth_point = right_point = [0, 0]
    left_point = low_point = [car_rect[0][0], car_rect[0][1]]
    for point in box:
        if left_point[0] > point[0]:
            left_point = point
        if low_point[1] > point[1]:
            low_point = point
        if heigth_point[1] < point[1]:
            heigth_point = point
        if right_point[0] < point[0]:
            right_point = point

    if left_point[1] <= right_point[1]:
        new_right_point = [right_point[0], heigth_point[1]]
        pts1 = np.float32([left_point, heigth_point, right_point])
        pts2 = np.float32([left_point, heigth_point, new_right_point])
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(image, M, (round(img_w * 2), round(img_h * 2)))
        car_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]

    elif left_point[1] > right_point[1]:  # 负角度
        new_left_point = [left_point[0], heigth_point[1]]
        pts1 = np.float32([left_point, heigth_point, right_point])
        pts2 = np.float32([new_left_point, heigth_point, right_point]) # 字符只是高度需要改变
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(image, M, (round(img_w * 2), round(img_h * 2)))
        car_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]

    return car_img


def pre_process(orig_img):
    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    blur_img = cv2.blur(gray_img, (3, 3))

    sobel_img = cv2.Sobel(blur_img, cv2.CV_16S, 1, 0, ksize=3)
    sobel_img = cv2.convertScaleAbs(sobel_img)

    hsv_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)

    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    blue_img = (((h > 26) & (h < 34)) | ((h > 100) & (h < 124))) & (s > 70) & (v > 70)
    blue_img = blue_img.astype('float32')

    mix_img = np.multiply(sobel_img, blue_img)

    mix_img = mix_img.astype(np.uint8)

    ret, binary_img = cv2.threshold(mix_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    return close_img


def verify_color(rotate_rect, src_image):
    img_h, img_w = src_image.shape[:2]
    mask = np.zeros(shape=[img_h + 2, img_w + 2], dtype=np.uint8)
    connectivity = 4
    loDiff, upDiff = 30, 30
    new_value = 255
    flags = connectivity
    flags |= cv2.FLOODFILL_FIXED_RANGE
    flags |= new_value << 8
    flags |= cv2.FLOODFILL_MASK_ONLY

    rand_seed_num = 5000
    valid_seed_num = 200
    adjust_param = 0.1
    box_points = cv2.boxPoints(rotate_rect)
    box_points_x = [n[0] for n in box_points]
    box_points_x.sort(reverse=False)
    adjust_x = int((box_points_x[2] - box_points_x[1]) * adjust_param)
    col_range = [box_points_x[1] + adjust_x, box_points_x[2] - adjust_x]
    box_points_y = [n[1] for n in box_points]
    box_points_y.sort(reverse=False)
    adjust_y = int((box_points_y[2] - box_points_y[1]) * adjust_param)
    row_range = [box_points_y[1] + adjust_y, box_points_y[2] - adjust_y]
    if (col_range[1] - col_range[0]) / (box_points_x[3] - box_points_x[0]) < 0.4 \
            or (row_range[1] - row_range[0]) / (box_points_y[3] - box_points_y[0]) < 0.4:
        points_row = []
        points_col = []
        for i in range(2):
            pt1, pt2 = box_points[i], box_points[i + 2]
            x_adjust, y_adjust = int(adjust_param * (abs(pt1[0] - pt2[0]))), int(adjust_param * (abs(pt1[1] - pt2[1])))
            if pt1[0] <= pt2[0]:
                pt1[0], pt2[0] = pt1[0] + x_adjust, pt2[0] - x_adjust
            else:
                pt1[0], pt2[0] = pt1[0] - x_adjust, pt2[0] + x_adjust
            if pt1[1] <= pt2[1]:
                pt1[1], pt2[1] = pt1[1] + adjust_y, pt2[1] - adjust_y
            else:
                pt1[1], pt2[1] = pt1[1] - y_adjust, pt2[1] + y_adjust
            temp_list_x = [int(x) for x in np.linspace(pt1[0], pt2[0], int(rand_seed_num / 2))]
            temp_list_y = [int(y) for y in np.linspace(pt1[1], pt2[1], int(rand_seed_num / 2))]
            points_col.extend(temp_list_x)
            points_row.extend(temp_list_y)
    else:
        points_row = np.random.randint(row_range[0], row_range[1], size=rand_seed_num)
        points_col = np.linspace(col_range[0], col_range[1], num=rand_seed_num).astype(np.int)

    points_row = np.array(points_row)
    points_col = np.array(points_col)
    hsv_img = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    flood_img = src_image.copy()
    seed_cnt = 0
    for i in range(rand_seed_num):
        rand_index = np.random.choice(rand_seed_num, 1, replace=False)
        row, col = points_row[rand_index], points_col[rand_index]
        if (((h[row, col] > 26) & (h[row, col] < 34)) | ((h[row, col] > 100) & (h[row, col] < 124))) & (
                s[row, col] > 70) & (v[row, col] > 70):
            cv2.floodFill(src_image, mask, (col, row), (255, 255, 255), (loDiff,) * 3, (upDiff,) * 3, flags)
            cv2.circle(flood_img, center=(col, row), radius=2, color=(0, 0, 255), thickness=2)
            seed_cnt += 1
            if seed_cnt >= valid_seed_num:
                break
    show_seed = np.random.uniform(1, 100, 1).astype(np.uint16)
    mask_points = []
    for row in range(1, img_h + 1):
        for col in range(1, img_w + 1):
            if mask[row, col] != 0:
                mask_points.append((col - 1, row - 1))
    mask_rotateRect = cv2.minAreaRect(np.array(mask_points))
    if verify_scale(mask_rotateRect):
        return True, mask_rotateRect
    else:
        return False, mask_rotateRect


def locate_carPlate(orig_img, pred_image):
    carPlate_list = []
    temp1_orig_img = orig_img.copy()
    temp2_orig_img = orig_img.copy()
    cloneImg, contours, heriachy = cv2.findContours(pred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        cv2.drawContours(temp1_orig_img, contours, i, (0, 255, 255), 2)
        rotate_rect = cv2.minAreaRect(contour)
        if verify_scale(rotate_rect):
            ret, rotate_rect2 = verify_color(rotate_rect, temp2_orig_img)
            if ret == False:
                continue
            car_plate = img_Transform(rotate_rect2, temp2_orig_img)
            car_plate = cv2.resize(car_plate, (car_plate_w, car_plate_h))  # 调整尺寸为后面CNN车牌识别做准备
            box = cv2.boxPoints(rotate_rect2)
            for k in range(4):
                n1, n2 = k % 4, (k + 1) % 4
                cv2.line(temp1_orig_img, (box[n1][0], box[n1][1]), (box[n2][0], box[n2][1]), (255, 0, 0), 2)
            cv2.imshow('opencv_' + str(i), car_plate)
            carPlate_list.append(car_plate)

    return carPlate_list


def horizontal_cut_chars(plate):
    char_addr_list = []
    area_left, area_right, char_left, char_right = 0, 0, 0, 0
    img_w = plate.shape[1]

    def getColSum(img, col):
        sum = 0
        for i in range(img.shape[0]):
            sum += round(img[i, col] / 255)
        return sum;

    sum = 0
    for col in range(img_w):
        sum += getColSum(plate, col)
    col_limit = 0
    charWid_limit = [round(img_w / 12), round(img_w / 5)]
    is_char_flag = False

    for i in range(img_w):
        colValue = getColSum(plate, i)
        if colValue > col_limit:
            if is_char_flag == False:
                area_right = round((i + char_right) / 2)
                area_width = area_right - area_left
                char_width = char_right - char_left
                if (area_width > charWid_limit[0]) and (area_width < charWid_limit[1]):
                    char_addr_list.append((area_left, area_right, char_width))
                char_left = i
                area_left = round((char_left + char_right) / 2)
                is_char_flag = True
        else:
            if is_char_flag:
                char_right = i - 1
                is_char_flag = False
    if area_right < char_left:
        area_right, char_right = img_w, img_w
        area_width = area_right - area_left
        char_width = char_right - char_left
        if (area_width > charWid_limit[0]) and (area_width < charWid_limit[1]):
            char_addr_list.append((area_left, area_right, char_width))
    return char_addr_list


def get_chars(car_plate):
    """

    :param car_plate:
    :return:
    """
    img_h, img_w = car_plate.shape[:2]
    h_proj_list = []  # 水平投影长度列表
    h_temp_len, v_temp_len = 0, 0
    h_startIndex, h_end_index = 0, 0  # 水平投影记索引
    h_proj_limit = [0.2, 0.8]  # 车牌在水平方向得轮廓长度少于20%或多余80%过滤掉
    char_imgs = []

    h_count = [0 for i in range(img_h)]
    for row in range(img_h):
        temp_cnt = 0
        for col in range(img_w):
            if car_plate[row, col] == 255:
                temp_cnt += 1
        h_count[row] = temp_cnt
        if temp_cnt / img_w < h_proj_limit[0] or temp_cnt / img_w > h_proj_limit[1]:
            if h_temp_len != 0:
                h_end_index = row - 1
                h_proj_list.append((h_startIndex, h_end_index))
                h_temp_len = 0
            continue
        if temp_cnt > 0:
            if h_temp_len == 0:
                h_startIndex = row
                h_temp_len = 1
            else:
                h_temp_len += 1
        else:
            if h_temp_len > 0:
                h_end_index = row - 1
                h_proj_list.append((h_startIndex, h_end_index))
                h_temp_len = 0

    if h_temp_len != 0:
        h_end_index = img_h - 1
        h_proj_list.append((h_startIndex, h_end_index))
    h_maxIndex, h_maxHeight = 0, 0
    for i, (start, end) in enumerate(h_proj_list):
        if h_maxHeight < (end - start):
            h_maxHeight = (end - start)
            h_maxIndex = i
    if h_maxHeight / img_h < 0.5:
        return char_imgs
    chars_top, chars_bottom = h_proj_list[h_maxIndex][0], h_proj_list[h_maxIndex][1]

    plates = car_plate[chars_top:chars_bottom + 1, :]
    cv2.imwrite('./carIdentityData/opencv_output/car.jpg', car_plate)
    cv2.imwrite('./carIdentityData/opencv_output/plate.jpg', plates)
    char_addr_list = horizontal_cut_chars(plates)

    for i, addr in enumerate(char_addr_list):
        char_img = car_plate[chars_top:chars_bottom + 1, addr[0]:addr[1]]
        char_img = cv2.resize(char_img, (char_w, char_h))
        char_imgs.append(char_img)
    return char_imgs


def extract_char(car_plate):
    gray_plate = cv2.cvtColor(car_plate, cv2.COLOR_BGR2GRAY)
    ret, binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    char_img_list = get_chars(binary_plate)
    return char_img_list


def cnn_select_carPlate(plate_list, model_path):
    if len(plate_list) == 0:
        return False, plate_list
    g1 = tf.Graph()
    sess1 = tf.Session(graph=g1)
    with sess1.as_default():
        with sess1.graph.as_default():
            model_dir = os.path.dirname(model_path)
            saver = tf.train.import_meta_graph(model_path)
            saver.restore(sess1, tf.train.latest_checkpoint(model_dir))
            graph = tf.get_default_graph()
            net1_x_place = graph.get_tensor_by_name('x_place:0')
            net1_keep_place = graph.get_tensor_by_name('keep_place:0')
            net1_out = graph.get_tensor_by_name('out_put:0')

            input_x = np.array(plate_list)
            net_outs = tf.nn.softmax(net1_out)
            preds = tf.argmax(net_outs, 1)  # 预测结果
            probs = tf.reduce_max(net_outs, reduction_indices=[1])  # 结果概率值
            pred_list, prob_list = sess1.run([preds, probs], feed_dict={net1_x_place: input_x, net1_keep_place: 1.0})
            result_index, result_prob = -1, 0.
            for i, pred in enumerate(pred_list):
                if pred == 1 and prob_list[i] > result_prob:
                    result_index, result_prob = i, prob_list[i]
            if result_index == -1:
                return False, plate_list[0]
            else:
                return True, plate_list[result_index]


def cnn_recongnize_char(img_list, model_path):
    g2 = tf.Graph()
    sess2 = tf.Session(graph=g2)
    text_list = []

    if len(img_list) == 0:
        return text_list
    with sess2.as_default():
        with sess2.graph.as_default():
            model_dir = os.path.dirname(model_path)
            saver = tf.train.import_meta_graph(model_path)
            saver.restore(sess2, tf.train.latest_checkpoint(model_dir))
            graph = tf.get_default_graph()
            net2_x_place = graph.get_tensor_by_name('x_place:0')
            net2_keep_place = graph.get_tensor_by_name('keep_place:0')
            net2_out = graph.get_tensor_by_name('out_put:0')

            data = np.array(img_list)
            net_out = tf.nn.softmax(net2_out)
            preds = tf.argmax(net_out, 1)
            my_preds = sess2.run(preds, feed_dict={net2_x_place: data, net2_keep_place: 1.0})

            for i in my_preds:
                text_list.append(char_table[i])
            return text_list


if __name__ == '__main__':
    cur_dir = sys.path[0]
    car_plate_w, car_plate_h = 136, 36
    char_w, char_h = 20, 20
    plate_model_path = os.path.join(cur_dir, './carIdentityData/model/plate_recongnize/model.ckpt-520.meta')
    char_model_path = os.path.join(cur_dir, './carIdentityData/model/char_recongnize/model.ckpt-520.meta')
    img_from_ROI = cv2.imread('./carIdentityData/images/3.jpg') # 接口

    pred_img = pre_process(img_from_ROI)

    car_plate_list = locate_carPlate(img_from_ROI, pred_img)

    ret, car_plate = cnn_select_carPlate(car_plate_list, plate_model_path)
    if not ret:
        print("未检测到车牌")
        sys.exit(-1)

    char_img_list = extract_char(car_plate)

    text = cnn_recongnize_char(char_img_list, char_model_path)

    print(text)
    cv2.waitKey(0)
