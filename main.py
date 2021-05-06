import cv2
import numpy as np
from ORBFeature import ORBFeature, show_match, orb_match


def read_image(image_name: str, folder: str) -> np.ndarray:
    image = cv2.imread(folder + image_name)
    # cv2.imshow("bird.jpg", image)
    # cv2.waitKey(0)
    return image

def oriented_warp(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    orb1 = ORBFeature(img1, 12, 4, 0.2)
    orb1.detector()
    # orb1.show_corner_points(image_data1)


    orb2 = ORBFeature(img2, 12, 4, 0.2)
    orb2.detector()
    # orb2.show_corner_points(image_data2)

    # show_match(orb1, orb2, image_data1, image_data2)
    matchs = orb_match(orb1, orb2)

    x_y_pairs = list()
    for i1, i2, score in matchs:
        x_y_pairs.append((orb1.corner_points[i1][:2], orb2.corner_points[i2][:2]))

    src_pts = [x for x, y in x_y_pairs]
    dst_pts = [y for x, y in x_y_pairs]

    src_pts = np.array([(x[0], x[1]) for x in src_pts])
    dst_pts = np.array([(x[0], x[1]) for x in dst_pts])

    h_mat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    h, w = image_data1.shape[:2]
    img = cv2.warpPerspective(img1, h_mat, (w, h))
    cv2.imshow("hello", img)
    cv2.waitKey(0)
    return img


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image_data1 = read_image('house1.jpg', '')
    # print(image_data.shape)
    # print(type(image_data))
    gray_image_data1 = cv2.cvtColor(image_data1, cv2.COLOR_BGR2GRAY)
    # fast = OrientedFast(gray_image_data, 9, 0.2)
    # fast.detector()
    # fast.show_interest_points(image_data)
    orb1 = ORBFeature(gray_image_data1, 12, 4, 0.2)
    orb1.detector()
    # orb1.show_corner_points(image_data1)

    image_data2 = read_image('house2.jpg', '')
    gray_image_data2 = cv2.cvtColor(image_data2, cv2.COLOR_BGR2GRAY)
    orb2 = ORBFeature(gray_image_data2, 12, 4, 0.2)
    orb2.detector()
    # orb2.show_corner_points(image_data2)

    # show_match(orb1, orb2, image_data1, image_data2)
    matchs = orb_match(orb1, orb2)

    x_y_pairs = list()
    for i1, i2, score in matchs:
        x_y_pairs.append((orb1.corner_points[i1][:2], orb2.corner_points[i2][:2]))
    
    src_pts = [x for x, y in x_y_pairs]
    dst_pts = [y for x, y in x_y_pairs]

    src_pts = np.array([(int(x[0]), int(x[1])) for x in src_pts])
    dst_pts = np.array([(int(x[0]), int(x[1])) for x in dst_pts])

    h_mat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    h, w = image_data1.shape[:2]
    img = cv2.warpPerspective(image_data1, h_mat, (w, h))
    full_image = np.concatenate((img, image_data2), axis=1)
    cv2.imshow("hello", full_image)
    cv2.waitKey(0)


