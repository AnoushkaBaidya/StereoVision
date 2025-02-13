import cv2
import numpy as np
import matplotlib.pyplot as plt


class EnhancedStereoVisionProcessor:
    def __init__(self, folder: str):
        self.folder = folder
        self.cam0, self.cam1, self.baseline, self.params = self._load_camera_parameters(folder)
        self.img1, self.img2 = self._load_images(folder)
        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * 8,  # Disparity range
            blockSize=9,  # Optimized block size
            P1=8 * 3 * 9 ** 2,
            P2=32 * 3 * 9 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.left_matcher)

    def _load_camera_parameters(self, folder: str):
        calib_file = f"./data/{folder}/calib.txt"
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
        img1 = cv2.imread(f"./data/{folder}/im0.png", cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(f"./data/{folder}/im1.png", cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            raise FileNotFoundError(f"Images for folder {folder} not found!")
        img1 = cv2.GaussianBlur(img1, (5, 5), 0)  # Apply Gaussian blur
        img2 = cv2.GaussianBlur(img2, (5, 5), 0)
        return img1, img2

    def compute_disparity(self):
        right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
        disparity_left = self.left_matcher.compute(self.img1, self.img2).astype(np.float32) / 16.0
        disparity_right = right_matcher.compute(self.img2, self.img1).astype(np.float32) / 16.0

        # WLS Filtering
        self.wls_filter.setLambda(8000)  # Adjust lambda for smoothness
        self.wls_filter.setSigmaColor(1.0)
        filtered_disparity = self.wls_filter.filter(disparity_left, self.img1, disparity_map_right=disparity_right)

        # Clip and normalize disparity
        filtered_disparity = np.clip(filtered_disparity, 0, 255)
        filtered_disparity = cv2.medianBlur(filtered_disparity.astype(np.uint8), 5)

        return filtered_disparity

    @staticmethod
    def apply_colormap(image):
        """Normalize and apply a cooler colormap to an image."""
        image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colormap = cv2.applyColorMap(image_normalized, cv2.COLORMAP_JET)  # Shift toward cooler tones
        return colormap
    
    def visualize(self, disparity):
        disparity_colored = self.apply_colormap(disparity)
        combined = np.vstack((cv2.cvtColor(self.img1, cv2.COLOR_GRAY2BGR), disparity_colored))
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        plt.title("Stereo Visualization Image and Enhanced Disparity Map)")
        plt.axis("off")
        plt.show()

    def process(self):
        disparity = self.compute_disparity()
        self.visualize(disparity)


if __name__ == "__main__":
    processor = EnhancedStereoVisionProcessor(folder="curule")
    #processor = EnhancedStereoVisionProcessor(folder="octagon")
    #processor = EnhancedStereoVisionProcessor(folder="pendulum")
    processor.process()
