import cv2
import numpy as np
from ORBFeature import ORBFeature, orb_match
from cnnmatching import cnn_match
from lib.utils import show_match

class Coarse2Fine(object):

    def __init__(self, img1, img2, M):
        self.img1 = img1
        self.img2 = img2
        self.h_mat = np.eye(3)
        self.M = M

    
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
        pts = np.array(pts)
        pts = np.concatenate((pts, np.ones(len(pts)).reshape(-1,1)), axis=1)
        reverse_mat = np.linalg.inv(self.h_mat)
        reverse_pts = np.dot(reverse_mat, pts.T).T
        reverse_pts = reverse_pts[:,:2]/np.average(reverse_pts[:,2])

        reverse_pts -= self.M[:,2]
        reverse_pts = np.dot(np.linalg.inv(self.M[:,:2]), reverse_pts.T).T
        return reverse_pts
    
    def warp_image(self):
        h, w = self.img2.shape[:2]
        return cv2.warpPerspective(self.img1, self.h_mat, (int(w * 1.5), int(h * 1.5)))

    def coarse2fine(self, iter=4, save_path='./'):

        for i in range(iter):
            print("iter", i)
            img1 = self.warp_image()
            img2 = self.img2
            src_pts, dst_pts = cnn_match(img1, img2)

            pairs = [(src_pts[i], dst_pts[i]) for i in range(len(src_pts))]
            pairs = [(x, y) for x, y in pairs if sum(img1[int(x[1])][int(x[0])]) != 0]
            acc = self.accuracy(src_pts, dst_pts)
            show_match(src_pts, dst_pts, img1, img2, save_path, str(i) + "_" + str(acc) + '_.png')
            
            h_mat = self.compute_homography(pairs)
            if i != iter - 1:
                self.h_mat = np.dot(h_mat, self.h_mat)

            
    
    def accuracy(self, src_pts, dst_pts):
        src_pts = self.reverse_warp_points(src_pts)
        dlst = list()
        for p1, p2 in zip(src_pts, dst_pts):
            dx = np.sum(np.abs(p1 - p2))
            dlst.append(dx)
        dlst = np.array(dlst)
        return np.count_nonzero(dlst < 20)/len(src_pts)