# Name: Diego da Costa Bernardo
# USP Number: 11362565
# Course Code: SCC5830
# Turma01_1Sem_2020
# Assignment 2 : Image Enhancement and Filtering

# Imports
import numpy as np
import imageio

np.seterr(over='ignore')

def euclidean_distance(n):
    distance = np.linspace(-(n - 1) / 2, (n - 1) / 2, n).astype(int)
    ed = np.zeros([n,n])
    central_point = int((n-1)/2)
    for x in distance:
        for y in distance:
            ed[x+central_point,y+central_point] = np.sqrt((np.power(x, 2) + np.power(y, 2)))
    return ed

def gaussian_kernel(x, sigma):
    a = 1.0/(2*np.pi*np.square(sigma))
    b = np.exp(-np.square(x)/(2*np.square(sigma)))
    return a * b

def spatial_component(n, sigma):
    ed = euclidean_distance(n)
    gs = np.zeros([n,n]) # Gaussian Spatial Componente
    for x in range(n):
        for y in range(n):
            gs[x, y] = gaussian_kernel(ed[x,y], sigma)
    return gs

def window(sub_i, gs, a, b):    
    N,M = sub_i.shape
    Wp = 0
    pixel_value = 0
    # Gaussian Range
    gr = np.zeros([n,n]) # , dtype=np.uint8
    # for every pixel
    for i in range(N):
        for j in range(M):
            # diff: I_i - I_(x,y)
            diff = sub_i[i,j] - sub_i[a,b]
            # computes g at (x,y)
            gr[i,j] = gaussian_kernel(diff, sigma_r)
            w = gr[i,j] * gs[i,j]
            Wp += w
            pixel_value += sub_i[i,j] * w
    pixel_value = pixel_value / Wp
    return pixel_value


def bilateral_filter(n):
    N,M = input_img.shape
    a = int((n-1)/2) # Center Row
    b = int((n-1)/2) # Center Col
    
    output_img = np.zeros([N+(2*a),M+(2*b)], dtype=np.uint8) # add zero-padding in the image
    #output_img = np.zeros(input_img.shape, dtype=np.uint8)
    
    gs = spatial_component(n, sigma_s)
    
    # for every pixel
    for x in range(a,N-a):
        for y in range(b,M-b):
            # gets subimage
            sub_i = input_img[ x-a : x+a+1 , y-b:y+b+1 ]
            output_img[x,y] = window(sub_i, gs, a, b).astype(np.uint8)
    return output_img


def window_laplacian(sub_i, k):
    kernel_1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    kernel_2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    
    kernel = kernel_1 if k ==1 else kernel_2
    return np.sum( np.multiply(sub_i, kernel))


def laplacian_filter():
    N,M = input_img.shape
    n = 3 # size of window
    a = int((n-1)/2)

    # new image to store filtered pixels
    output_img = np.zeros(input_img.shape)

    # for every pixel
    for x in range(a,N-a):
        for y in range(a,M-a):
            # gets subimage
            sub_i = input_img[ x-a : x+a+1 , y-a:y+a+1 ]
            output_img[x,y] = window_laplacian(sub_i, k)

    # Normalize the image
    imax = np.max(output_img)
    imin = np.min(output_img)
    output_img = ((output_img - imin) * 255) / imax
    
    # Add c weight
    output_img = (c * output_img) + input_img
    
    # Normalize
    imax = np.max(output_img)
    imin = np.min(output_img)
    output_img = ((output_img - imin) * 255) / imax

    return output_img.astype(np.uint8)


def vignette_filter(std_row, std_col):
    # Get Rows size and Column Size
    rows,cols = input_img.shape

    # Get the center of the rows and columns
    center_row = int(round((rows/2)-1))
    center_col = int(round((cols/2)-1))

    # Get positon values: -2, -1, 0, 1, 2
    distance_row = np.linspace(-(rows - 1) / 2, (rows - 1) / 2, rows).astype(int)
    distance_col = np.linspace(-(cols - 1) / 2, (cols - 1) / 2, cols).astype(int)

    # Gaussian Kernel Row
    k_row = []
    for i in range(rows):
        #std_row = np.std(input_img[i])
        k_row.append(gaussian_kernel(distance_row[i], std_row))

    # Gaussian Kernel Col
    k_col = []
    for i in range(cols):
        #std_col = np.std(input_img[:][i])
        k_col.append(gaussian_kernel(distance_col[i], std_col))

    # Apply filter in the input image    
    k_col = np.array(k_col).reshape(cols,1)
    k_row = np.array(k_row).reshape(rows,1)
    w = k_col.T * k_row
    output_img = w * input_img

    # Normalize the output_img
    imax = np.max(output_img)
    imin = np.min(output_img)
    output_img = ((output_img - imin) * 255) / imax

    return output_img


# Inputs
def inputs():
    filename = str(input()).rstrip()
    input_img = imageio.imread(filename)
    method = int(input())
    save = int(input())
    
    return input_img, method, save


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
    n = int(input())
    sigma_s = float(input())
    sigma_r = float(input())
    output_img = bilateral_filter(n)
if method == 2:
    c = float(input())
    k = int(input())
    output_img = laplacian_filter()
if method == 3:
    std_row = float(input())
    std_col = float(input())
    output_img = vignette_filter(std_row, std_col)
    
# Compare with reference
compare()

if save == 1:
    imageio.imwrite('output_img.png',output_img)