import numpy as np
from convolution import convolve2d
import kernels

def get_bayer_masks(height, width, pattern='RGGB'):
#  Tworzy binarne maski dla kanałów R, G, B zgodnie ze wzorem Bayera (RGGB).

    mask_r = np.zeros((height, width))
    mask_g = np.zeros((height, width))
    mask_b = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            if i % 2 == 0:  # parzyste
                if j % 2 == 0:
                    mask_g[i, j] = 1  # G
                else:
                    mask_r[i, j] = 1  # R
            else:  # nieparzyste
                if j % 2 == 0:
                    mask_b[i, j] = 1  # B
                else:
                    mask_g[i, j] = 1  # G
    return mask_r, mask_g, mask_b


def demosaic_bayer(raw_image):
    h, w = raw_image.shape
    mask_r, mask_g, mask_b = get_bayer_masks(h, w)

    # Mnożymy obraz wyjściowy przez maski, aby uzyskać rzadkie próbki
    R_sparse = raw_image * mask_r
    G_sparse = raw_image * mask_g
    B_sparse = raw_image * mask_b

    # Rekonstrukcja kanałów przez konwolucję
    R_interp = convolve2d(R_sparse, kernels.DEMOSAIC_KERNEL_RB)
    G_interp = convolve2d(G_sparse, kernels.DEMOSAIC_KERNEL_G)
    B_interp = convolve2d(B_sparse, kernels.DEMOSAIC_KERNEL_RB)

    # Składamy wynikowy obraz RGB
    rgb_image = np.stack([R_interp, G_interp, B_interp], axis=2)
    return rgb_image


def get_fuji_masks(height, width):
# Tworzy binarne maski dla kanałów R, G, B według wzoru Fuji

    mask_r = np.zeros((height, width))
    mask_g = np.zeros((height, width))
    mask_b = np.zeros((height, width))

# Definiujemy wzór Fuji X-Trans 6x6
    pattern = [
        "GBRGBR",
        "RGGBGG",
        "BGGRGG",
        "GBRGBR",
        "RGGBGG",
        "BGGRGG"
    ]

    for i in range(height):
        for j in range(width):
            c = pattern[i % 6][j % 6]
            if c == 'R':
                mask_r[i, j] = 1
            elif c == 'G':
                mask_g[i, j] = 1
            elif c == 'B':
                mask_b[i, j] = 1

    return mask_r, mask_g, mask_b


def visualize_mosaic(raw_image, mask_r, mask_g, mask_b):
# Tworzy wizualizację mozaiki Bayera/ X-Trans w kolorze.
# Piksele zachowują swoją jasność z surowego obrazu.
    
    h, w = raw_image.shape
    visulaisation_image = np.zeros((h, w, 3))

    visulaisation_image[:, :, 0] = raw_image * mask_r
    visulaisation_image[:, :, 1] = raw_image * mask_g
    visulaisation_image[:, :, 2] = raw_image * mask_b

    return visulaisation_image


def demosaic_fuji(raw_image):
# Demozaikowanie Fuji X-Trans (6x6) z prostą interpolacją uśredniającą.

    h, w = raw_image.shape
    mask_r, mask_g, mask_b = get_fuji_masks(h, w)

    R_sparse = raw_image * mask_r
    G_sparse = raw_image * mask_g
    B_sparse = raw_image * mask_b

# Prosty kernel uśredniający dla interpolacji
    kernel_fuji = np.ones((3, 3)) / 4.5  

    R_interp = convolve2d(R_sparse, kernel_fuji)
    G_interp = convolve2d(G_sparse, kernel_fuji)
    B_interp = convolve2d(B_sparse, kernel_fuji)

# Normalizacja każdego kanału do zakresu 0-1
    def normalize_channel(ch):
        return (ch - ch.min()) / (ch.max() - ch.min() + 1e-5)

    R_interp = normalize_channel(R_interp)
    G_interp = normalize_channel(G_interp)
    B_interp = normalize_channel(B_interp)

    return np.stack([R_interp, G_interp, B_interp], axis=2)
