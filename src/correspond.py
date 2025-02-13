import cv2
import numpy as np
import matplotlib.pyplot as plt


class StereoCorrespondence:
    def __init__(self, window_size: int = 30, disparity_range: int = 100, step_size: int = 5):
        """
        Initialize the StereoCorrespondence class with parameters for block matching.
        :param window_size: Size of the sliding window for SSD calculation.
        :param disparity_range: Range of disparity values to search.
        :param step_size: Step size for searching disparity values.
        """
        self.window_size = window_size
        self.disparity_range = disparity_range
        self.step_size = step_size

    @staticmethod
    def compute_ssd(kernel1: np.ndarray, kernel2: np.ndarray) -> float:
        """
        Compute the Sum of Squared Differences (SSD) between two kernels.
        :param kernel1: The first image patch.
        :param kernel2: The second image patch.
        :return: The SSD value.
        """
        return np.sum(np.square(kernel1 - kernel2))

    def compute_disparity_map(self, img1: np.ndarray, img2: np.ndarray):
        """
        Compute the disparity map using block matching with SSD.
        :param img1: The rectified left image.
        :param img2: The rectified right image.
        :return: The disparity map and its corresponding visualization.
        """
        h, w = img1.shape
        disp_map = np.zeros((h, w))

        for i in range(0, h, self.window_size):
            for j in range(0, w, self.window_size):
                # Define the sliding window in the first image
                kernel1 = img1[i:i + self.window_size, j:j + self.window_size]

                ssd_min = np.inf
                best_offset = 0

                for k in range(max(0, j - self.disparity_range), j, self.step_size):
                    # Define the sliding window in the second image
                    kernel2 = img2[i:i + self.window_size, k:k + self.window_size]

                    # Ensure both kernels have the same shape
                    if kernel1.shape != kernel2.shape:
                        continue

                    ssd = self.compute_ssd(kernel1, kernel2)

                    if ssd < ssd_min:
                        ssd_min = ssd
                        best_offset = k

                # Compute disparity
                disparity = np.abs(j - best_offset)
                disp_map[i:i + self.window_size, j:j + self.window_size] = disparity

        disp_img = self._scale_disparity_map(disp_map)
        return disp_map, disp_img

    @staticmethod
    def _scale_disparity_map(disp_map: np.ndarray) -> np.ndarray:
        """
        Scale the disparity map for visualization.
        :param disp_map: The disparity map to scale.
        :return: A scaled disparity map for visualization.
        """
        max_val = np.max(disp_map)
        min_val = np.min(disp_map)
        scale = 255 / (max_val - min_val)

        disp_img = (disp_map - min_val) * scale
        disp_img = np.uint8(disp_img)
        disp_img_color = cv2.applyColorMap(disp_img, cv2.COLORMAP_TURBO)
        return disp_img_color

    @staticmethod
    def plot_disparity_map(disp_img: np.ndarray, title: str = "Disparity Map"):
        """
        Plot the disparity map with proper titles.
        :param disp_img: The disparity map to plot.
        :param title: The title for the plot.
        """
        plt.figure(figsize=(10, 5))
        plt.imshow(disp_img)
        plt.title(title)
        plt.axis("off")
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    # Example usage
    img1 = cv2.imread("path_to_img1", cv2.IMREAD_GRAYSCALE)  # Replace with actual image paths
    img2 = cv2.imread("path_to_img2", cv2.IMREAD_GRAYSCALE)  # Replace with actual image paths

    stereo_correspondence = StereoCorrespondence(window_size=30, disparity_range=100, step_size=5)
    disp_map, disp_img = stereo_correspondence.compute_disparity_map(img1, img2)

    # Plot the results
    stereo_correspondence.plot_disparity_map(disp_img)
