from coarse_2_fine import Coarse2Fine
import cv2
from lib.utils import rotate_bound


if __name__ == "__main__":
    img1 = cv2.imread('df-ms-data/1/img.jpg')
    img2 = cv2.imread('df-ms-data/1/img.jpg')
    img1 = rotate_bound(img1, 10)
    c2f = Coarse2Fine(img1, img2)
    c2f.coarse_wrap()
    c2f.coarse2fine()