import os
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from convolution import convolve2d, normalize_image
import kernels
import demosaic as demosaicing


def save_image(path, image):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    img_out = normalize_image(image)

    # Konwersja RGB -> BGR
    if img_out.ndim == 3 and img_out.shape[2] == 3:
        img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

    cv2.imwrite(path, img_out)
    print("Saved: {path}")


def main():
# Wczytanie obrazów w szarości 
    img_gray = cv2.imread('pies.jpg', cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print("File 'pies.jpg' not found. Creating 100x100 image...")
        img_gray = np.zeros((100, 100), dtype=float)
        img_gray[25:75, 25:75] = 100
    img_gray = img_gray.astype(float) / 255.0 

# Wczytanie obrazów w kolorze
    img_color_in = cv2.imread('pies.jpg')
    if img_color_in is not None:
        img_color_in = cv2.cvtColor(img_color_in, cv2.COLOR_BGR2RGB)
        img_color_norm = img_color_in.astype(float) / 255.0
    else:
        img_color_norm = None

    print("Processing image...")
    

# ZADANIE 1: Wykrywanie krawędzi
    edges_sobel_x = convolve2d(img_gray, kernels.SOBEL_X)
    edges_sobel_y = convolve2d(img_gray, kernels.SOBEL_Y)
    edges_laplace = convolve2d(img_gray, kernels.LAPLACE)
    edges_sobel_mag = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)

    edges_prewitt_x = convolve2d(img_gray, kernels.PREWITT_X)
    edges_prewitt_y = convolve2d(img_gray, kernels.PREWITT_Y)
    edges_prewitt_mag = np.sqrt(edges_prewitt_x**2 + edges_prewitt_y**2)

    edges_scharr_x = convolve2d(img_gray, kernels.SCHARR_X)
    edges_scharr_y = convolve2d(img_gray, kernels.SCHARR_Y)
    edges_scharr_mag = np.sqrt(edges_scharr_x**2 + edges_scharr_y**2)

# Zapis krawędzi i zapisywanie wynikow do folderu
    save_image('output/edges/sobel_x.png', edges_sobel_x)
    save_image('output/edges/sobel_y.png', edges_sobel_y)
    save_image('output/edges/sobel_magnitude.png', edges_sobel_mag)

    save_image('output/edges/prewitt_x.png', edges_prewitt_x)
    save_image('output/edges/prewitt_y.png', edges_prewitt_y)
    save_image('output/edges/prewitt_magnitude.png', edges_prewitt_mag)

    save_image('output/edges/scharr_x.png', edges_scharr_x)
    save_image('output/edges/scharr_y.png', edges_scharr_y)
    save_image('output/edges/scharr_magnitude.png', edges_scharr_mag)

# ZADANIE 2: Rozmywanie (Blurring) 
    img_for_filter = img_color_norm if img_color_norm is not None else img_gray
    blurred = convolve2d(img_for_filter, kernels.GAUSSIAN_BLUR)
    save_image('output/filters/gaussian_blur.png', blurred)

# ZADANIE 3: Wyostrzanie (Sharpening)
    sharpened = convolve2d(img_for_filter, kernels.SHARPEN)
    sharpened = np.clip(sharpened, 0, 1)
    save_image('output/filters/sharpen.png', sharpened)
    save_image('output/filters/laplace.png', edges_laplace)

# ZADANIE 4: Demozaikowanie 
    demosaiced_bayer = None
    if img_color_in is not None:
        h, w, _ = img_color_in.shape

    # Sztuczna mozaika Bayera 
        mask_r, mask_g, mask_b = demosaicing.get_bayer_masks(h, w)
        raw_bayer = img_color_in[:, :, 0] * mask_r + img_color_in[:, :, 1] * mask_g + img_color_in[:, :, 2] * mask_b

    # Sztuczna mozaika Fuji X-Trans 
        mask_r_f, mask_g_f, mask_b_f = demosaicing.get_fuji_masks(h, w)
        raw_fuji = img_color_in[:, :, 0] * mask_r_f + img_color_in[:, :, 1] * mask_g_f + img_color_in[:, :, 2] * mask_b_f

    # Demozaikowanie
        demosaiced_bayer = demosaicing.demosaic_bayer(raw_bayer)
        demosaiced_fuji = demosaicing.demosaic_fuji(raw_fuji)

    # Wizualizacja mozaiki
        vis_bayer = demosaicing.visualize_mosaic(raw_bayer, mask_r, mask_g, mask_b)
        vis_fuji = demosaicing.visualize_mosaic(raw_fuji, mask_r_f, mask_g_f, mask_b_f)

    # Zapis wyników ostatecznych
        save_image('output/demosaicing/raw_bayer_color.png', vis_bayer)
        save_image('output/demosaicing/demosaiced_bayer.png', demosaiced_bayer)
        save_image('output/demosaicing/raw_fuji_color.png', vis_fuji)
        save_image('output/demosaicing/demosaiced_fuji.png', demosaiced_fuji)

# Prezentacja wyników 
    plt.figure(figsize=(15, 20))

    # Sobel
    plt.subplot(6, 3, 1), plt.imshow(normalize_image(edges_sobel_x), cmap='gray'), plt.title('Sobel X')
    plt.subplot(6, 3, 2), plt.imshow(normalize_image(edges_sobel_y), cmap='gray'), plt.title('Sobel Y')
    plt.subplot(6, 3, 3), plt.imshow(normalize_image(edges_sobel_mag), cmap='gray'), plt.title('Sobel Magnitude')

    # Prewitt
    plt.subplot(6, 3, 4), plt.imshow(normalize_image(edges_prewitt_x), cmap='gray'), plt.title('Prewitt X')
    plt.subplot(6, 3, 5), plt.imshow(normalize_image(edges_prewitt_y), cmap='gray'), plt.title('Prewitt Y')
    plt.subplot(6, 3, 6), plt.imshow(normalize_image(edges_prewitt_mag), cmap='gray'), plt.title('Prewitt Magnitude')

    # Scharr
    plt.subplot(6, 3, 7), plt.imshow(normalize_image(edges_scharr_x), cmap='gray'), plt.title('Scharr X')
    plt.subplot(6, 3, 8), plt.imshow(normalize_image(edges_scharr_y), cmap='gray'), plt.title('Scharr Y')
    plt.subplot(6, 3, 9), plt.imshow(normalize_image(edges_scharr_mag), cmap='gray'), plt.title('Scharr Magnitude')

    # Inne filtry
    plt.subplot(6, 3, 10), plt.imshow(normalize_image(edges_laplace), cmap='gray'), plt.title('Laplace')
    plt.subplot(6, 3, 11), plt.imshow(normalize_image(blurred), cmap='gray'), plt.title('Gaussian Blur')
    plt.subplot(6, 3, 12), plt.imshow(normalize_image(sharpened), cmap='gray'), plt.title('Sharpen')

# Demozaikowanie Bayer
    if demosaiced_bayer is not None:
        plt.subplot(6, 3, 13), plt.imshow(normalize_image(vis_bayer)), plt.title('Raw Bayer Color')
        plt.subplot(6, 3, 14), plt.imshow(normalize_image(demosaiced_bayer)), plt.title('Demosaiced Bayer')
        plt.subplot(6, 3, 16), plt.imshow(normalize_image(vis_fuji)), plt.title('Raw Fuji Color')
        plt.subplot(6, 3, 17), plt.imshow(normalize_image(demosaiced_fuji)), plt.title('Demosaiced Fuji')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
