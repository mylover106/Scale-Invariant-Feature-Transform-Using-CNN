
"""
circular
                    @  @  @  @
                    @  @  @  @  @  @
                    @  @  @  @  @  @  @  @
                    @  @  @  @  @  @  @  @  @
                    @  @  @  @  @  @  @  @  @  @
                    @  @  @  @  @  @  @  @  @  @  @
                    @  @  @  @  @  @  @  @  @  @  @  @
                    @  @  @  @  @  @  @  @  @  @  @  @  @
                    @  @  @  @  @  @  @  @  @  @  @  @  @
                    @  @  @  @  @  @  @  @  @  @  @  @  @  @
                    @  @  @  @  @  @  @  @  @  @  @  @  @  @
                    @  @  @  @  @  @  @  @  @  @  @  @  @  @
                    @  @  @  @  @  @  @  @  @  @  @  @  @  @  @
                    @  @  @  @  @  @  @  @  @  @  @  @  @  @  @
                    @  @  @  @  @  @  @  @  @  @  @  @  @  @  @
                    @  @  @  @  @  @  @  @  @  @  @  @  @  @  @

"""


import numpy as np
import math
import cv2
from tqdm import tqdm

# brief algorithm samples
brief_pt1 = np.random.randint(-8, 8, (2, 256))
brief_pt2 = np.random.randint(-8, 8, (2, 256))


def blur(image, kernel=(3, 3)):
    image = cv2.blur(image, kernel, 0)
    return image


class OrientedFast(object):
    """Detect Fast Feature points and compute the orientation.

        Attributes:
            image: input image saved as numpy.ndarry type.
                   as default, image should be gray image.

            n_point: if there are at least n connected points on the circle are brighter or darker,
                     then this point is a feature point.

            grayscale_threshold(T): if I_x < I_p - T, I_x is darker,
                                    if I_x > I_p + T, I_x is brighter.
                                    I_x is the point on circle, I_p is the center point.
                                    T = averayge(I) * T

            interest_points: points detected by Fast Feature algorithm
    """

    DARKER = 1
    BRIGHTER = 0

    def __init__(self, image: np.ndarray, n_point=9, threshold=0.2):
        """initialize the OrientedFast Class.

        Args:
            image: input grayscale image
            n_point: threshold for minimum number of continuous points at circle
            threshold: grayscale threshold to judge circle point is darker or brighter
        """
        self.circle_points = [[-1, -3], [0, -3], [1, -3], [2, -2],
                              [3, -1], [3, 0], [3, 1], [2, 2],
                              [1, 3], [0, 3], [-1, 3], [-2, 2],
                              [-3, 1], [-3, 0], [-3, -1], [-2, -2]]
        self.circle_points_x = np.array([p[0] for p in self.circle_points])
        self.circle_points_y = np.array([p[1] for p in self.circle_points])

        self.image = blur(image)
        self.n_point = n_point
        self.grayscale_threshold = np.sum(image)/(image.shape[0] * image.shape[1]) * threshold
        self.interest_points = list()
        return

    def detector(self):
        """detect fast corner points, and save to self.interest_points

        Returns:
            void
        """

        def is_valid(arr):
            cnt = 0
            for rlt in arr + arr:
                cnt = cnt + 1 if rlt else 0
                if cnt > self.n_point:
                    return True
            return False

        self.interest_points = list()
        circle_points_x = self.circle_points_x
        circle_points_y = self.circle_points_y

        for y in tqdm(range(3, self.image.shape[0]-3)):
            for x in range(3, self.image.shape[1]-3):
                pixel = self.image[y, x]
                points_x = circle_points_x + x
                points_y = circle_points_y + y
                circle_pixels = self.image[points_y, points_x]

                brighter_arr = circle_pixels > pixel + self.grayscale_threshold
                darker_arr = circle_pixels < pixel - self.grayscale_threshold

                if is_valid(brighter_arr):
                    self.interest_points.append((x, y, self.BRIGHTER))
                elif is_valid(darker_arr):
                    self.interest_points.append((x, y, self.DARKER))

        self.non_maximum_suppression()
        self.feature_orientation()
        self.brief_descriptor()
        return

    def non_maximum_suppression(self):
        """Delete points with lower scores among adjacent points

        Returns:
            void
        """
        circle_points_x = np.array([p[0] for p in self.circle_points])
        circle_points_y = np.array([p[1] for p in self.circle_points])

        def score(x, y, pixel_type):
            pixel = self.image[y, x]
            points_x = circle_points_x + x
            points_y = circle_points_y + y

            circle_pixels = self.image[points_y, points_x].astype(np.int32)
            circle_pixels -= pixel

            if pixel_type == self.BRIGHTER:
                ret_score = np.sum(circle_pixels[circle_pixels > 0])
                return ret_score

            ret_score = -np.sum(circle_pixels[circle_pixels < 0])
            return ret_score

        pixel2score = dict()

        for x, y, pixel_type in self.interest_points:
            pixel2score[x, y, pixel_type] = score(x, y, pixel_type)

        delete_lst = list()

        def can_deleted(x, y):
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    near_p = x + dx, y + dy, pixel_type
                    if near_p in pixel2score:
                        if pixel2score[near_p] > pixel2score[(x, y, pixel_type)]:
                            return True
            return False

        for x, y, pixel_type in self.interest_points:
            if can_deleted(x, y):
                delete_lst.append((x, y, pixel_type))

        for p in delete_lst:
            del pixel2score[p]

        self.interest_points = list(pixel2score.keys())

        return

    def compute_theta(self, x, y, radius, half_width_max):
        """compute the orientation theta for feature point(x, y)
           using intensity centroid algorithm.

        Args:
            x: point location of x
            y: point location of y
            radius: circle area's radius
            half_width_max: max half width of circle from 0-16 col

        Returns:
            theta, the direction of feature point(x, y)
        """

        height, width = self.image.shape
        m10, m01 = 0, 0
        for col in range(-radius+1, radius):
            cur_width = half_width_max[abs(col)]
            if col + y < 0 or col + y >= height:
                continue
            for row in range(0, cur_width):
                if row + x < width:
                    m10 += row * self.image[col + y, row + x]
                    m01 += col * self.image[col + y, row + x]
                if -row + x >= 0:
                    m10 -= row * self.image[col + y, -row + x]
                    m01 += col * self.image[col + y, -row + x]

        return math.atan2(m01, m10)

    def feature_orientation(self, radius=16):
        """compute the orientation of the feature point

        Args:
            radius: intensity centroid needs to define a circle

        Returns:
            void
        """
        half_width_max = list()
        for i in range(0, radius):
            half_width_max.append(int((radius**2 - i**2)**0.5))

        for i, p in enumerate(self.interest_points):
            self.interest_points[i] = *p, self.compute_theta(p[0], p[1], radius, half_width_max)

        return

    def brief_descriptor(self):
        # padding image
        image = np.pad(self.image, ((8, 8), (8, 8)), 'constant', constant_values=((0, 0), (0, 0)))

        def R_matrix(angle):
            ret = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
            return np.array(ret)

        for i, p in enumerate(self.interest_points):
            cur_p = np.array([p[0] + 8, p[1] + 8]).reshape((-1, 1))
            R = R_matrix(p[-1])
            cur_pt1 = np.dot(R, brief_pt1).astype(np.int) + cur_p
            cur_pt2 = np.dot(R, brief_pt2).astype(np.int) + cur_p
            descriptor = image[cur_pt1[1], cur_pt1[0]] > image[cur_pt2[1], cur_pt2[0]]
            self.interest_points[i] = (*p, descriptor)

        return

    def show_interest_points(self, image=None):
        """visualize the detected corner points

        Returns:
            void
        """

        if image is not None:
            show_image = image.copy()
        else:
            show_image = self.image.copy()

        for p in self.interest_points:
            dx = int(math.cos(p[-2]) * 12)
            dy = int(math.sin(p[-2]) * 12)
            pt2 = p[0] + dx, p[1] + dy
            if p[2] == self.BRIGHTER:
                cv2.circle(show_image, center=p[:2], radius=3, color=(255, 0, 0), thickness=1)
                cv2.arrowedLine(show_image, pt1=p[:2], pt2=pt2, color=(255, 0, 0), thickness=1)
            else:
                cv2.circle(show_image, center=p[:2], radius=3, color=(0, 0, 255), thickness=1)
                cv2.arrowedLine(show_image, pt1=p[:2], pt2=pt2, color=(0, 0, 255), thickness=1)
        cv2.imshow('fast', show_image)
        cv2.waitKey(0)
        cv2.imwrite('fast.png', show_image)


class ORBFeature(object):
    """Detect ORB feature points

        Attributes:
            image: image to detect feature points, needs to be grayscale.

            n_threshold: for OrientedFast to detect feature points.

            n_layer: pyramid's layer

            t_threshold: for OrientedFast to detect feature points.
    """
    DARKER = 1
    BRIGHTER = 0

    def __init__(self, image, n_threshold, n_layer, t_threshold):
        self.image = image
        self.n_layer = n_layer
        self.t_threshold = t_threshold
        self.n_threshold = n_threshold
        self.corner_points = list()
        self.fast_detector = list()

    def down_sample(self, image, n):
        """down sample the image to 2^n

        Args:
            image: image to be down sampled.
            n: down sample's level

        Returns:
            image be down sampled.
        """
        x = [i for i in range(0, image.shape[0], 2**n)]
        y = [i for i in range(0, image.shape[1], 2**n)]
        return image[x, :][:, y]

    def detector(self):
        for i in range(self.n_layer):
            cur_image = self.down_sample(self.image, i)
            fast_detector = OrientedFast(cur_image, self.n_threshold, self.t_threshold)
            fast_detector.detector()
            self.fast_detector.append(fast_detector)

        for i in range(self.n_layer):
            for p in self.fast_detector[i].interest_points:
                scale_x = self.image.shape[1] / self.fast_detector[i].image.shape[1]
                scale_y = self.image.shape[0] / self.fast_detector[i].image.shape[0]
                # x, y, pixel_type, theta, descriptor
                self.corner_points.append((p[0] * scale_x, p[1] * scale_y, p[2], p[3], p[4], i))
        return

    def show_corner_points(self, image, title='orb'):
        if image is not None:
            show_image = image.copy()
        else:
            show_image = self.image.copy()

        for p in self.corner_points:
            dx = int(math.cos(p[3]) * 12)
            dy = int(math.sin(p[3]) * 12)
            x, y = int(p[0]), int(p[1])
            pt2 = x + dx, y + dy
            if p[2] == self.BRIGHTER:
                cv2.circle(show_image, center=(x, y), radius=2**p[-1], color=(255, 0, 0), thickness=1)
                cv2.arrowedLine(show_image, pt1=(x, y), pt2=pt2, color=(255, 0, 0), thickness=1)
            else:
                cv2.circle(show_image, center=(x, y), radius=2**p[-1], color=(0, 0, 255), thickness=1)
                cv2.arrowedLine(show_image, pt1=(x, y), pt2=pt2, color=(0, 0, 255), thickness=1)
        cv2.imshow(title, show_image)
        # cv2.waitKey(0)
        cv2.imwrite(title + '.png', show_image)
        return show_image


def orb_match(orb1: ORBFeature, orb2: ORBFeature):
    """match the orb feature points from two different image.

    Args:
        orb1: first image's detected ORB feature points
        orb2: second image's detected ORB feature points

    Returns:
        match pairs in different image, for example:
        [(i, j), ...], i is the i'th orb point detected, j is j'th orb point detected.
    """
    matches = list()
    for i1, p1 in enumerate(orb1.corner_points):
        for i2, p2 in enumerate(orb2.corner_points):
            distance = np.count_nonzero(p1[4] ^ p2[4])
            if distance < 64 and p1[2] == p2[2]:
                matches.append((distance, i1, i2))

    matches.sort()
    orb1_used = list(False for p in orb1.corner_points)
    orb2_used = list(False for p in orb2.corner_points)

    ret = list()
    for score, i1, i2 in matches:
        if not orb1_used[i1] and not orb2_used[i2]:
            ret.append((i1, i2, score))
        orb1_used[i1] = True
        orb2_used[i2] = True

    return ret


def show_match(orb1: ORBFeature, orb2: ORBFeature, image1, image2):
    """match orb points, and visualization

    Args:
        orb1: first image's orb feature points
        orb2: second image's orb feature points
        image1: first image, np.array format
        image2: second image, np.array format

    Returns:
        void
    """
    # concate image1 and image2 to a big image
    image1 = orb1.show_corner_points(image1, 'orb1')
    image2 = orb2.show_corner_points(image2, 'orb2')
    h1, w1, _ = image1.shape
    h2, w2, _ = image2.shape
    if h1 < h2:
        zeros = np.zeros((h2 - h1, w1, 3)).astype(np.uint8)
        image1 = np.concatenate((image1, zeros), axis=0)
    if h1 > h2:
        zeros = np.zeros((h1 - h2, w2, 3)).astype(np.uint8)
        image2 = np.concatenate((image2, zeros), axis=0)
    matchs = orb_match(orb1, orb2)
    full_image = np.concatenate((image1, image2), axis=1)

    for i1, i2, score in matchs:
        r, g, b = np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)
        p1x, p1y = int(orb1.corner_points[i1][0]), int(orb1.corner_points[i1][1])
        p2x, p2y = int(orb2.corner_points[i2][0]), int(orb2.corner_points[i2][1])
        thick = 1
        if score < 50:
            thick += 1
        if score < 40:
            thick += 1
        if score < 30:
            thick += 2
        if score < 20:
            thick += 2
        cv2.line(full_image, (p1x, p1y), (p2x + w1, p2y), color=(r, g, b), thickness=thick)
    cv2.imshow('pair_image', full_image)
    cv2.imwrite('pair_image_match.png', full_image)
    cv2.waitKey(0)
    return 

