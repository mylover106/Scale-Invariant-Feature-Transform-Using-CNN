import cv2
import numpy as np
from ORBFeature import ORBFeature, orb_match
from cnnmatching import cnn_match
from lib.utils import show_match

class Coarse2Fine(object):

    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2
        self.h_mat = np.eye(3)

    
    def compute_homography(self, pairs):
       

        src_pts = [x for x, y in pairs]
        dst_pts = [y for x, y in pairs]

        src_pts = np.array([(x[0], x[1]) for x in src_pts])
        dst_pts = np.array([(x[0], x[1]) for x in dst_pts])

        h_mat, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        
        return h_mat

    
    def coarse_wrap(self):
        img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        orb1 = ORBFeature(img1, 12, 4, 0.2)
        orb1.detector()
        # orb1.show_corner_points(image_data1)

        orb2 = ORBFeature(img2, 12, 4, 0.2)
        orb2.detector()

        matchs = orb_match(orb1, orb2)
        

        x_y_pairs = list()
        for i1, i2, score in matchs:
            x_y_pairs.append((orb1.corner_points[i1][:2], orb2.corner_points[i2][:2]))

        self.h_mat = self.compute_homography(x_y_pairs)

    
    def reverse_warp_points(self, pts):
        reverse_mat = np.linalg.inv(self.h_mat)
        reverse_pts = np.dot(reverse_mat, pts.T).T
        return reverse_pts
    
    def warp_image(self):
        h, w = self.img2.shape[:2]
        return cv2.warpPerspective(self.img1, self.h_mat, (w * 2, h * 2))

    def coarse2fine(self, iter=4):

        for i in range(iter):
            print("iter", i)
            img1 = self.warp_image()
            img2 = self.img2
            src_pts, dst_pts = cnn_match(img1, img2)


            # show_match(src_pts, dst_pts, img1, img2)

            pairs = [(src_pts[i], dst_pts[i]) for i in range(len(src_pts))]
            pairs = [(x, y) for x, y in pairs if sum(img1[int(x[1])][int(x[0])]) != 0]
            h_mat = self.compute_homography(pairs)
            if i != iter - 1:
                self.h_mat = np.dot(h_mat, self.h_mat)
        
            show_match(src_pts, dst_pts, img1, img2)

        




    def oriented_warp(self):
        img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
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

        h_mat, mask = self.compute_homography(x_y_pairs)

        h, w = img1.shape[:2]
        img = cv2.warpPerspective(img1, h_mat, (w, h))
        cv2.imshow("hello", img)
        cv2.waitKey(0)
        return img