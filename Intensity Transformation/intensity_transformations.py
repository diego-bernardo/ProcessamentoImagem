# Name: Diego da Costa Bernardo
# USP Number: 11362565
# Course Code: SCC5830
# Turma01_1Sem_2020
# Assignment 1 : intensity transformations

# Imports
import numpy as np
import imageio

# Initial Inputs
def inputs():
    filename = str(input()).rstrip()
    input_img = imageio.imread(filename)
    method = int(input())
    save = int(input())
    
    return input_img, method, save


# Inversion Transformation
def inversion():
    img_invert = 255-input_img
    return img_invert


# Contrast Modulation
def contrast_modulation():
    a = np.min(input_img)
    b = np.max(input_img)
    img_contrast = (input_img - a) * ((d-c)/(b-a)) + c    
    return img_contrast


# Logarithmic Transformation
def logarithmic():
    R = np.max(input_img)
    #img_log = (255 * (np.log2(1 + input_img) / np.log2(1 + R))).astype(np.uint8)
    
    rows = input_img.shape[0]
    cols = input_img.shape[1]
    img_log = np.zeros(input_img.shape, dtype=float)

    # Loop through all pixels
    for x in range(rows):
        for y in range(cols):
            img_log[x,y] = 255 * (np.log2(1 + input_img[x,y]) / np.log2(1 + R))
    
    return img_log


# Gamma Adjustmen Transformation
def gamma_adjustment():
    img_gamma = (W * np.power(input_img, lambd)).astype(np.uint8)
    return img_gamma


# Compare with reference
def compare():
    accumulate = 0
    rows = input_img.shape[0]
    cols = input_img.shape[1]

    # Loop through all pixels
    for x in range(rows):
        for y in range(cols):
            accumulate += np.power(float(output_img[x,y]) - float(input_img[x,y]), 2)

    rse = np.around(np.sqrt(accumulate), decimals=4)
    print(str(rse))



# Starting the process
input_img, method, save = inputs()

# Select the transformation method
if method == 1:
    output_img = inversion()
if method == 2:
    c = int(input())
    d = int(input())
    output_img = contrast_modulation()
if method == 3:
    output_img = logarithmic()
if method == 4:
    W = int(input())
    lambd = float(input())
    output_img = gamma_adjustment()
    
# Compare with reference
compare()

# Store output file
if save == 1:
    imageio.imwrite('output_img.png',output_img)