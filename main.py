from coarse_2_fine import Coarse2Fine
import cv2
from lib.utils import rotate_bound
import glob


if __name__ == "__main__":
    files = glob.glob('test_data/*')
    for path in files:
        img1_name = path + '/pair/' + 'img1.png'
        img2_name = path + '/pair/' + 'img2.png'
        img1 = cv2.imread(img1_name)
        img2 = cv2.imread(img2_name)
        img1, M = rotate_bound(img1, 50)
        c2f = Coarse2Fine(img1, img2, M)
        # c2f.coarse_wrap()
        c2f.coarse2fine(iter=1, save_path=path+'/coarse50/')
