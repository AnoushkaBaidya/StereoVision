import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class StereoVisionProcessor:
    def __init__(self, folder: str):
        self.folder = folder
        self.cam0, self.cam1, self.baseline, self.params = self._load_camera_parameters(folder)
        self.img1, self.img2 = self._load_images(folder)

        # StereoSGBM for disparity calculation
        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * 12,  # High disparity range
            blockSize=11,
            P1=8 * 3 * 11 ** 2,
            P2=32 * 3 * 11 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        # WLS filter for better disparity map
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.left_matcher)

    def _load_camera_parameters(self, folder: str):
        calib_file = os.path.join(folder, "calib.txt")
        with open(calib_file, "r") as file:
            data = file.readlines()

        parameters = {}
        for line in data:
            if "=" in line:
                key, value = line.strip().split("=")
                parameters[key.strip()] = value.strip()

        cam0 = np.array([[float(x) for x in row.split()] for row in parameters["cam0"][1:-1].split(";")])
        cam1 = np.array([[float(x) for x in row.split()] for row in parameters["cam1"][1:-1].split(";")])
        baseline = float(parameters["baseline"])
        params = {
            "width": int(parameters["width"]),
            "height": int(parameters["height"]),
            "ndisp": int(parameters["ndisp"]),
            "vmin": int(parameters["vmin"]),
            "vmax": int(parameters["vmax"]),
        }

        return cam0, cam1, baseline, params

    def _load_images(self, folder: str):
        img1 = cv2.imread(os.path.join(folder, "im0.png"), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(folder, "im1.png"), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            raise FileNotFoundError(f"Images for folder {folder} not found!")
        img1 = cv2.GaussianBlur(img1, (5, 5), 0)  # Noise reduction
        img2 = cv2.GaussianBlur(img2, (5, 5), 0)
        return img1, img2

    def compute_disparity(self):
        right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
        disparity_left = self.left_matcher.compute(self.img1, self.img2).astype(np.float32) / 16.0
        disparity_right = right_matcher.compute(self.img2, self.img1).astype(np.float32) / 16.0

        # WLS Filtering
        self.wls_filter.setLambda(8000)
        self.wls_filter.setSigmaColor(1.5)
        filtered_disparity = self.wls_filter.filter(disparity_left, self.img1, disparity_map_right=disparity_right)

        # Post-processing
        filtered_disparity = np.clip(filtered_disparity, 0, 255)
        filtered_disparity = cv2.bilateralFilter(filtered_disparity.astype(np.uint8), 9, 75, 75)  # Stronger filter
        filtered_disparity = cv2.equalizeHist(filtered_disparity)  # Enhance contrast
        filtered_disparity = cv2.medianBlur(filtered_disparity, 5)  # Smooth edges

        return filtered_disparity

    @staticmethod
    def apply_colormap(image):
        """Normalize and apply a reversed colormap to an image."""
        image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colormap = cv2.applyColorMap(image_normalized, cv2.COLORMAP_TURBO)
        return colormap

    def visualize(self, disparity, output_folder):
        disparity_colored = self.apply_colormap(disparity)

        # Create a larger figure
        plt.figure(figsize=(16, 12))  # Increased figure size for better visibility

        # Display the left image
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(self.img1, cmap='gray')
        ax1.set_title("Left Image", fontsize=16, fontweight='bold', style='italic')  # Larger title
        ax1.axis("off")

        # Display the enhanced disparity map
        ax2 = plt.subplot(1, 2, 2)
        ax2.imshow(cv2.cvtColor(disparity_colored, cv2.COLOR_BGR2RGB))
        ax2.set_title("Enhanced Disparity Map", fontsize=16, fontweight='bold', style='italic')

        ax2.axis("off")

        # Adjust subplot spacing for larger visuals
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1)

        # Save the plot
        output_file = os.path.join(output_folder, "disparity_map.png")
        plt.savefig(output_file, dpi=300)  # Save with higher resolution for clarity
        plt.close()


    def process(self, output_folder):
        disparity = self.compute_disparity()
        self.visualize(disparity, output_folder)


def process_all_folders(dataset_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing: {folder_name}")
            output_folder = os.path.join(output_path, folder_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            processor = StereoVisionProcessor(folder=folder_path)
            processor.process(output_folder)


if __name__ == "__main__":
    dataset_path = "./DataSet"
    output_path = "./output"
    process_all_folders(dataset_path, output_path)
