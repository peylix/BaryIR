import cv2
import numpy as np
import torch

# This script is adapted from the following repository: https://github.com/JingyunLiang/SwinIR


def calculate_psnr(img1, img2, test_y_channel=True):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f"Image shapes are differnet: {img1.shape}, {img2.shape}."
    )
    assert img1.shape[2] == 3
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def calculate_ssim(img1, img2, test_y_channel=False):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f"Image shapes are differnet: {img1.shape}, {img2.shape}."
    )
    assert img1.shape[2] == 3
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.0
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.0


def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    convertion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.0
    else:
        raise TypeError(
            f"The img type should be np.float32 or np.uint8, but got {img_type}"
        )
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace convertion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(
            f"The dst_type should be np.float32 or np.uint8, but got {dst_type}"
        )
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.0
    return img.astype(dst_type)


def calculate_mae(img1, img2, test_y_channel=True):
    """Calculate MAE (Mean Absolute Error) on the [0, 1] scale.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        test_y_channel (bool): Test on Y channel of YCbCr. Default: True.

    Returns:
        float: mae result on the [0, 1] scale.
    """

    assert img1.shape == img2.shape, (
        f"Image shapes are differnet: {img1.shape}, {img2.shape}."
    )
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    return float(np.mean(np.abs(img1 - img2)) / 255.0)


def _bgr_to_rgb_tensor(img_bgr, device):
    """Convert a BGR uint8 image (HWC) to an RGB float tensor in [0, 1] (1CHW)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    return tensor


class PerceptualMetricComputer:
    """Lazily-loaded LPIPS / DISTS computer that reuses the loaded networks.

    The models are heavy to construct, so they are instantiated on first use and
    cached for subsequent calls. Inputs are BGR uint8 images (as returned by
    ``cv2.imread``).
    """

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._lpips_model = None
        self._dists_model = None

    def _get_lpips_model(self):
        if self._lpips_model is None:
            try:
                import lpips
            except ImportError as exc:
                raise ImportError("LPIPS requires `pip install lpips`.") from exc
            self._lpips_model = lpips.LPIPS(net="alex").to(self.device).eval()
        return self._lpips_model

    def _get_dists_model(self):
        if self._dists_model is None:
            try:
                from piq import DISTS
            except ImportError as exc:
                raise ImportError("DISTS requires `pip install piq`.") from exc
            self._dists_model = DISTS().to(self.device).eval()
        return self._dists_model

    @torch.no_grad()
    def calculate_lpips(self, img1, img2):
        """LPIPS distance between two BGR uint8 images (lower is better)."""
        model = self._get_lpips_model()
        pred = _bgr_to_rgb_tensor(img1, self.device) * 2.0 - 1.0
        gt = _bgr_to_rgb_tensor(img2, self.device) * 2.0 - 1.0
        return float(model(pred, gt).item())

    @torch.no_grad()
    def calculate_dists(self, img1, img2):
        """DISTS distance between two BGR uint8 images (lower is better)."""
        model = self._get_dists_model()
        pred = _bgr_to_rgb_tensor(img1, self.device)
        gt = _bgr_to_rgb_tensor(img2, self.device)
        return float(model(pred, gt).item())


def summarize_metric(values):
    """Return the (mean, sample std) of a list of metric values.

    Non-finite values (e.g. ``inf`` PSNR from identical images) are ignored.
    The standard deviation uses ``ddof=1`` (sample std), matching the reference
    evaluation. Returns ``(nan, nan)`` if there are no finite values.

    Args:
        values (Sequence[float]): Per-image metric values.

    Returns:
        tuple[float, float]: (mean, std).
    """
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    mean = float(finite.mean())
    std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    return mean, std


def format_mean_std(mean, std, decimals=4):
    """Format a (mean, std) pair as ``"mean ± std"`` with fixed decimals."""
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img,
            [
                [24.966, 112.0, -18.214],
                [128.553, -74.203, -93.786],
                [65.481, -37.797, 112.0],
            ],
        ) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img
