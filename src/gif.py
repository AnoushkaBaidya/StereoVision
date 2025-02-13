import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio


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

    def save_frame(self, disparity, folder_name, frame_index):
        """Save individual frames for GIF creation with cropped focus on the center."""
        disparity_colored = self.apply_colormap(disparity)

        # Determine the crop dimensions (center focus)
        height, width = self.img1.shape
        crop_size = int(min(height, width) * 0.7)  # Crop to 60% of the smaller dimension
        center_y, center_x = height // 2, width // 2
        y1 = max(center_y - crop_size // 2, 0)
        y2 = min(center_y + crop_size // 2, height)
        x1 = max(center_x - crop_size // 2, 0)
        x2 = min(center_x + crop_size // 2, width)

        # Crop the left image and disparity map
        img1_cropped = self.img1[y1:y2, x1:x2]
        disparity_colored_cropped = disparity_colored[y1:y2, x1:x2]

        # Save the disparity map as an image
        output_file = os.path.join(folder_name, f"frame_{folder_name}_{frame_index}.png")

        # Plot and save the frame
        plt.figure(figsize=(8, 4))  # Slightly smaller figure for focused visuals
        plt.subplot(1, 2, 1)
        plt.imshow(img1_cropped, cmap='gray')
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(disparity_colored_cropped, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)  # Remove white space
        plt.close()



def create_gif(input_folder, output_gif):
    """Combine saved frames into a GIF."""
    images = []
    for file_name in sorted(os.listdir(input_folder)):
        if file_name.endswith(".png"):
            file_path = os.path.join(input_folder, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(output_gif, images, duration=0.5)  # Adjust duration as needed


if __name__ == "__main__":
    dataset_folder = "DataSet"
    output_frames_folder = "frames"
    output_gif_file = "disparity_map.gif"

    # Ensure the frames folder exists
    os.makedirs(output_frames_folder, exist_ok=True)

    frame_index = 0  # Initialize frame index for naming
    for folder_name in os.listdir(dataset_folder):
        full_folder_path = os.path.join(dataset_folder, folder_name)
        if os.path.isdir(full_folder_path):
            processor = StereoVisionProcessor(folder=full_folder_path)
            disparity = processor.compute_disparity()
            processor.save_frame(disparity, output_frames_folder, frame_index)
            frame_index += 1  # Increment frame index

    # Create a GIF from saved frames
    create_gif(output_frames_folder, output_gif_file)
    print(f"GIF saved to {output_gif_file}")
