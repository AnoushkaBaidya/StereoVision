import cv2
import numpy as np
import matplotlib.pyplot as plt


class DepthEstimator:
    @staticmethod
    def computeDepth(disp_map: np.ndarray, cam0: np.ndarray, baseline: float):
        """
        Compute the depth map from the disparity map, camera matrix, and baseline.
        :param disp_map: The disparity map.
        :param cam0: The intrinsic camera matrix of the left camera.
        :param baseline: The distance between the two cameras.
        :return: The depth map and its corresponding visualization.
        """
        # Focal length from the camera matrix
        f = cam0[0][0]
        
        # Avoid division by zero
        disp_map[disp_map == 0] = 0.1

        # Compute depth map
        depth_map = np.multiply(1/disp_map, (f*baseline))
        depth_map = np.uint8(depth_map)

        # Normalize depth map for visualization
        depth_img = DepthEstimator._normalize_depth_map(depth_map)

        return depth_map, depth_img

    @staticmethod
    def _normalize_depth_map(depth_map: np.ndarray) -> np.ndarray:
        """
        Normalize the depth map for visualization.
        :param depth_map: The raw depth map.
        :return: A normalized depth image for visualization.
        """
        max_depth = np.amax(depth_map)
        min_depth = np.amin(depth_map)
        scale = (255/(max_depth - min_depth))

        # Scale and convert to 8-bit
        depth_img = np.multiply(depth_map, scale)
        depth_img = np.subtract(depth_img, ((255*min_depth)/(max_depth-min_depth)))
        depth_img = np.uint8(depth_img)

        # Apply heatmap for better visualization
        depth_img_heat = cv2.applyColorMap(depth_img, cv2.COLORMAP_TURBO)
        return depth_img

    @staticmethod
    def plot_depth_map(depth_img: np.ndarray, title: str = "Depth Map"):
        """
        Plot the depth map with proper titles.
        :param depth_img: The depth map image to plot.
        :param title: Title for the plot.
        """
        plt.figure(figsize=(10, 5))
        plt.imshow(depth_img)
        plt.title(title)
        plt.axis("off")
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    # Example usage
    disp_map = np.random.rand(100, 100) * 255  # Example disparity map (replace with real data)
    cam0 = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])  # Example intrinsic matrix
    baseline = 0.5  # Example baseline in meters

    # Compute depth
    depth_estimator = DepthEstimator()
    depth_map, depth_img = depth_estimator.computeDepth(disp_map, cam0, baseline)

    # Plot the depth map
    depth_estimator.plot_depth_map(depth_img)
