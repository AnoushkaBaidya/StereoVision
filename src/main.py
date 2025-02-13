import cv2
import numpy as np
import matplotlib.pyplot as plt
from rectify import ImageRectifier
from depth import DepthEstimator
from calibrate import StereoCalibrator
from correspond import StereoCorrespondence

class StereoVisionProcessor:
    def __init__(self, folder: str):
        self.folder = folder
        self.cam0, self.cam1, self.baseline, self.params = self._load_camera_parameters(folder)
        self.img1, self.img2 = self._load_images(folder)

    def _load_camera_parameters(self, folder: str):
        """Load intrinsic and extrinsic camera parameters from calib.txt."""
        calib_file = f'./data/{folder}/calib.txt'
        try:
            with open(calib_file, 'r') as file:
                data = file.readlines()

            parameters = {}
            for line in data:
                key, value = line.strip().split('=')
                parameters[key.strip()] = value.strip()

            # Parse parameters
            cam0 = np.array([[float(x) for x in row.split()] for row in parameters['cam0'][1:-1].split(';')])
            cam1 = np.array([[float(x) for x in row.split()] for row in parameters['cam1'][1:-1].split(';')])
            baseline = float(parameters['baseline'])
            params = {
                'width': int(parameters['width']),
                'height': int(parameters['height']),
                'ndisp': int(parameters['ndisp']),
                'vmin': int(parameters['vmin']),
                'vmax': int(parameters['vmax']),
            }

            return cam0, cam1, baseline, params
        except FileNotFoundError:
            raise FileNotFoundError(f"Calibration file not found in folder: {folder}")
        except KeyError as e:
            raise ValueError(f"Missing key {e} in calibration file: {calib_file}")
        except Exception as e:
            raise ValueError(f"Error reading calibration file: {calib_file}, {e}")

    def _load_images(self, folder: str):
        """Load and convert stereo images to grayscale."""
        img1 = cv2.imread(f'./data/{folder}/im0.png', cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(f'./data/{folder}/im1.png', cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            raise FileNotFoundError(f"Images for folder {folder} not found!")
        return img1, img2

    def compute_matrices(self):
        """Compute fundamental and essential matrices."""
        src_pts, dst_pts = StereoCalibrator.features_match(self.img1, self.img2)
        F = StereoCalibrator.compute_fundamental_matrix(src_pts, dst_pts)
        E = StereoCalibrator.compute_essential_matrix(F, self.cam0, self.cam1)

        print("Fundamental Matrix (F):")
        print(F)
        print("\nEssential Matrix (E):")
        print(E)

        return F, E, src_pts, dst_pts

    def rectify_and_draw(self, F, src_pts, dst_pts):
        """Rectify images and draw epipolar lines."""
        
        img1_rect, img2_rect, F_new, src_pts_new, dst_pts_new = ImageRectifier.rectify_images(
            self.img1, self.img2, F, src_pts, dst_pts
        )
        ImageRectifier.draw_epipolar_lines(img1_rect, img2_rect, F_new, src_pts_new, dst_pts_new)
        return img1_rect, img2_rect

    def compute_disparity_and_depth(self, img1_rect, img2_rect):
        """Compute disparity and depth maps."""
        stereo_correspondence = StereoCorrespondence(window_size=30, disparity_range=100, step_size=5)
        disp_map, disp_img = stereo_correspondence.compute_disparity_map(img1_rect, img2_rect)

        self._plot_image(disp_img, "Disparity Map", cmap="inferno")
        depth_map, depth_img = DepthEstimator.computeDepth(disp_map, self.cam0, self.baseline)
        self._plot_image(depth_img, "Depth Map", cmap="inferno")

        return disp_map, depth_map

    @staticmethod
    def _plot_image(image, title, cmap="gray"):
        """Plot an image with proper title and color map."""
        plt.figure(figsize=(10, 5))
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.colorbar()
        plt.axis("off")
        plt.show()

    def process(self):
        """Run the stereo vision pipeline."""
        F, E, src_pts, dst_pts = self.compute_matrices()
        img1_rect, img2_rect = self.rectify_and_draw(F, src_pts, dst_pts)
        self.compute_disparity_and_depth(img1_rect, img2_rect)


if __name__ == "__main__":
    #processor = StereoVisionProcessor(folder="octagon")
    processor = StereoVisionProcessor(folder="curule")
    #processor = StereoVisionProcessor(folder="pendulum")
    processor.process()
