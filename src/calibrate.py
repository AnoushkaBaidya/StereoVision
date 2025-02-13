import cv2
import numpy as np
import matplotlib.pyplot as plt

class StereoCalibrator:
    def __init__(self):
        pass

    @staticmethod
    def features_match(img1: np.ndarray, img2: np.ndarray):
        """
        Detect and match features between two images using ORB and FLANN.
        """
        orb = cv2.ORB_create(nfeatures=500)

        # Detect and compute keypoints and descriptors
        kp1, desc1 = orb.detectAndCompute(img1, None)
        kp2, desc2 = orb.detectAndCompute(img2, None)

        # FLANN-based matcher parameters
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2)
        search_params = {}
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_matches.append(m)

        # Extract points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # Draw matches
        matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
        plt.imshow(matched_img)
        plt.title("Feature Matches")
        plt.axis("off")
        plt.show()

        return src_pts, dst_pts

    @staticmethod
    def normalize_points(pts: np.ndarray):
        """
        Normalize a set of 2D points to improve numerical stability.
        """
        mean_x, mean_y = np.mean(pts, axis=0)
        scale = np.sqrt(2) / np.sqrt(np.mean(np.sum(np.square(pts - [mean_x, mean_y]), axis=1)))

        # Transformation matrix
        T = np.array([
            [scale, 0, -scale * mean_x],
            [0, scale, -scale * mean_y],
            [0, 0, 1]
        ])

        # Apply transformation
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
        pts_normalized = (T @ pts_h.T).T
        pts_normalized /= pts_normalized[:, -1][:, None]

        return pts_normalized[:, :2], T

    @staticmethod
    def compute_fundamental_matrix(src_pts: np.ndarray, dst_pts: np.ndarray):
        """
        Compute the fundamental matrix using the normalized 8-point algorithm.
        """
        src_norm, T_src = StereoCalibrator.normalize_points(src_pts)
        dst_norm, T_dst = StereoCalibrator.normalize_points(dst_pts)

        # Construct the matrix A for the 8-point algorithm
        A = np.array([
            [
                src_norm[i, 0] * dst_norm[i, 0],
                src_norm[i, 0] * dst_norm[i, 1],
                src_norm[i, 0],
                src_norm[i, 1] * dst_norm[i, 0],
                src_norm[i, 1] * dst_norm[i, 1],
                src_norm[i, 1],
                dst_norm[i, 0],
                dst_norm[i, 1],
                1
            ]
            for i in range(len(src_norm))
        ])

        # Solve for F using SVD
        _, _, Vt = np.linalg.svd(A)
        F_norm = Vt[-1].reshape(3, 3)

        # Enforce rank-2 constraint
        U, S, Vt = np.linalg.svd(F_norm)
        S[-1] = 0
        F_norm = U @ np.diag(S) @ Vt

        # Denormalize the fundamental matrix
        F = T_dst.T @ F_norm @ T_src
        F /= F[-1, -1]

        return F

    @staticmethod
    def ransac_fundamental_matrix(src_pts: np.ndarray, dst_pts: np.ndarray, threshold: float = 0.001, max_iter: int = 2000):
        """
        Estimate the fundamental matrix using RANSAC to handle outliers.
        """
        max_inliers = 0
        best_F = None

        for _ in range(max_iter):
            # Randomly sample 20 points
            rand_indices = np.random.choice(len(src_pts), size=20, replace=False)
            src_sample = src_pts[rand_indices]
            dst_sample = dst_pts[rand_indices]

            # Compute the fundamental matrix for the sample
            F_candidate = StereoCalibrator.compute_fundamental_matrix(src_sample, dst_sample)

            # Compute inliers
            src_h = np.hstack([src_pts, np.ones((src_pts.shape[0], 1))])
            dst_h = np.hstack([dst_pts, np.ones((dst_pts.shape[0], 1))])
            errors = np.abs(np.diagonal(dst_h @ F_candidate @ src_h.T))
            inliers = errors < threshold

            # Update the best model if more inliers are found
            num_inliers = np.sum(inliers)
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_F = F_candidate

        return best_F

    @staticmethod
    def compute_essential_matrix(F: np.ndarray, cam0: np.ndarray, cam1: np.ndarray):
        """
        Compute the essential matrix from the fundamental matrix.
        """
        E = cam1.T @ F @ cam0

        # Enforce rank-2 constraint
        U, S, Vt = np.linalg.svd(E)
        S[-1] = 0
        E = U @ np.diag(S) @ Vt

        return E

    @staticmethod
    def decompose_essential_matrix(E: np.ndarray):
        """
        Decompose the essential matrix into rotation and translation components.
        """
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        U, _, Vt = np.linalg.svd(E)

        # Four possible solutions for rotation and translation
        R1 = U @ W @ Vt
        R2 = U @ W.T @ Vt
        t1 = U[:, 2]
        t2 = -U[:, 2]

        return (R1, t1), (R1, t2), (R2, t1), (R2, t2)


if __name__ == "__main__":
    calibrator = StereoCalibrator()

    # Example usage (replace with actual image paths)
    img1 = cv2.imread("path_to_img1", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("path_to_img2", cv2.IMREAD_GRAYSCALE)

    src_pts, dst_pts = calibrator.features_match(img1, img2)
    F = calibrator.compute_fundamental_matrix(src_pts, dst_pts)
    print("Fundamental Matrix:", F)
