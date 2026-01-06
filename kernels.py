import numpy as np

# Wykrywanie krawędzi 


# Operator Sobela 
SOBEL_X = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

SOBEL_Y = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

# Operator Laplace'a 
LAPLACE = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])

# Operator Scharra [cite: 63]
SCHARR_X = np.array([
    [3, 0, -3],
    [10, 0, -10],
    [3, 0, -3]
])

SCHARR_Y = np.array([
    [3, 10, 3],
    [0, 0, 0],
    [-3, -10, -3]
])

# Operator Prewitta [cite: 63]
PREWITT_X = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

PREWITT_Y = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
])



# Rozmywanie (Blurring) 


# Przybliżone rozmycie Gaussowskie 3x3
GAUSSIAN_BLUR = (1/16) * np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
])

#  Wyostrzanie (Sharpening) 

# Filtr wyostrzający W
SHARPEN = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])



# Jądra do demozaikowania 


# Interpolacja dwuliniowa dla kanału Zielonego (G) w masce Bayera
DEMOSAIC_KERNEL_G = (1/4) * np.array([
    [0, 1, 0],
    [1, 4, 1],
    [0, 1, 0]
])

# Interpolacja dla kanałów Czerwonego/Niebieskiego (R/B) w masce Bayera
DEMOSAIC_KERNEL_RB = (1/4) * np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
])
