import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageRectifier:
    @staticmethod
    def rectify_images(img1: np.ndarray, img2: np.ndarray, F: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray):
        """
        Rectify a pair of images using their fundamental matrix and corresponding points.
        :param img1: Left image (grayscale).
        :param img2: Right image (grayscale).
        :param F: Fundamental matrix.
        :param src_pts: Source points in the left image.
        :param dst_pts: Destination points in the right image.
        :return: Rectified images, updated fundamental matrix, and transformed points.
        """
        h, w = img2.shape
        _, H1, H2 = cv2.stereoRectifyUncalibrated(src_pts, dst_pts, F, imgSize=(w, h), threshold=0)

        print("Homography Matrix for Left Image (H1):\n", H1)
        print("Homography Matrix for Right Image (H2):\n", H2)

        img1_rect = cv2.warpPerspective(img1, H1, (w, h))
        img2_rect = cv2.warpPerspective(img2, H2, (w, h))

        F_new = (np.linalg.inv(H2).T) @ F @ np.linalg.inv(H1)

        src_new_pts = ImageRectifier._transform_points(H1, src_pts)
        dst_new_pts = ImageRectifier._transform_points(H2, dst_pts)

        return img1_rect, img2_rect, F_new, src_new_pts, dst_new_pts

    @staticmethod
    def _transform_points(H: np.ndarray, pts: np.ndarray):
        """
        Transform points using a homography matrix.
        :param H: Homography matrix.
        :param pts: Points to transform.
        :return: Transformed points.
        """
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
        pts_transformed = H @ pts_h.T
        pts_transformed /= pts_transformed[-1]
        return pts_transformed[:2].T

    @staticmethod
    def draw_lines(img1: np.ndarray, img2: np.ndarray, lines: np.ndarray, pts1: np.ndarray, pts2: np.ndarray):
        """
        Draw epipolar lines and corresponding points on rectified images.
        :param img1: Left rectified image.
        :param img2: Right rectified image.
        :param lines: Epipolar lines.
        :param pts1: Points in the left image.
        :param pts2: Points in the right image.
        :return: Images with epipolar lines drawn.
        """
        r, c = img1.shape
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        for line, pt1, pt2 in zip(lines, pts1, pts2):
            color = (255, 0, 0)
            x0, y0 = map(int, [0, -line[2] / line[1]])
            x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])

            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            img1 = cv2.circle(img1, (int(pt1[0]), int(pt1[1])), 5, color, -1)
            img2 = cv2.circle(img2, (int(pt2[0]), int(pt2[1])), 5, color, -1)

        return img1, img2

    @staticmethod
    def draw_epipolar_lines(img1_rect: np.ndarray, img2_rect: np.ndarray, F: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray):
        """
        Draw epipolar lines on rectified images.
        :param img1_rect: Left rectified image.
        :param img2_rect: Right rectified image.
        :param F: Fundamental matrix.
        :param src_pts: Source points in the left image.
        :param dst_pts: Destination points in the right image.
        """
        lines1 = cv2.computeCorrespondEpilines(dst_pts.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
        img1_with_lines, _ = ImageRectifier.draw_lines(img1_rect, img2_rect, lines1, src_pts, dst_pts)

        lines2 = cv2.computeCorrespondEpilines(src_pts.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
        img2_with_lines, _ = ImageRectifier.draw_lines(img2_rect, img1_rect, lines2, dst_pts, src_pts)

        # Visualize side-by-side
        combined_img = np.concatenate((img1_with_lines, img2_with_lines), axis=1)
        plt.figure(figsize=(12, 6))
        plt.imshow(combined_img)
        plt.title("Epipolar Lines on Rectified Images")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    # Example usage (replace with real data)
    img1 = cv2.imread("path_to_left_image", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("path_to_right_image", cv2.IMREAD_GRAYSCALE)
    F = np.eye(3)  # Replace with a valid fundamental matrix
    src_pts = np.random.rand(10, 2) * 100  # Replace with actual source points
    dst_pts = np.random.rand(10, 2) * 100  # Replace with actual destination points

    rectifier = ImageRectifier()
    img1_rect, img2_rect, F_new, src_new_pts, dst_new_pts = rectifier.rectify_images(img1, img2, F, src_pts, dst_pts)
    rectifier.draw_epipolar_lines(img1_rect, img2_rect, F_new, src_new_pts, dst_new_pts)
