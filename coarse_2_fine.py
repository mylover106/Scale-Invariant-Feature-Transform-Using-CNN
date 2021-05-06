import cv2
import numpy as np
from ORBFeature import ORBFeature, orb_match

class Coarse2Fine(object):

    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2
        self.h_mat, self.mask = self.coarse_wrap()

    
    def compute_homography(self, pairs):
       

        src_pts = [x for x, y in pairs]
        dst_pts = [y for x, y in pairs]

        src_pts = np.array([(x[0], x[1]) for x in src_pts])
        dst_pts = np.array([(x[0], x[1]) for x in dst_pts])

        h_mat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        
        return h_mat, mask
    
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

        h_mat, mask = self.compute_homography(x_y_pairs)
        return h_mat, mask
    
    def reverse_warp_points(self, pts):
        reverse_mat = np.linalg.inv(self.h_mat)
        reverse_pts = np.dot(reverse_mat, pts.T).T
        return reverse_pts
    

    def coarse2fine(self, iter=3):
        for i in range(iter):
            pass
        




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