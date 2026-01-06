import numpy as np

def convolve2d(image, kernel):
# Obsługa obrazów kolorowych
    if image.ndim == 3:
        h, w, channels = image.shape
        output = np.zeros_like(image)
        for ch in range(channels):
            output[:, :, ch] = convolve2d(image[:, :, ch], kernel)
        return output

    img_h, img_w = image.shape
    k_h, k_w = kernel.shape

# Obliczenie paddingu (dla nieparzystych wymiarów kernela)
    pad_h = k_h // 2
    pad_w = k_w // 2

# Tworzymy obraz z paddingiem zerowym
    padded = np.zeros((img_h + 2*pad_h, img_w + 2*pad_w), dtype=image.dtype)
    padded[pad_h:pad_h+img_h, pad_w:pad_w+img_w] = image

# Wynik
    output = np.zeros_like(image, dtype=float)

# Splot
    for i in range(img_h):
        for j in range(img_w):
            region = padded[i:i+k_h, j:j+k_w]
            output[i, j] = np.sum(region * kernel)

    return output

def normalize_image(img):
# Normalizuje obraz do zakresu 0-255 i konwertuje do uint8
    min_val, max_val = img.min(), img.max()
    if max_val - min_val == 0:
        return np.zeros_like(img, dtype=np.uint8)

    norm_img = 255 * (img - min_val) / (max_val - min_val)
    return norm_img.astype(np.uint8)
