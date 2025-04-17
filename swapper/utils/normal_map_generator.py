import argparse
import math
import numpy as np
from scipy import ndimage
from matplotlib import pyplot
from PIL import Image, ImageOps
import os
import multiprocessing as mp
import cv2

def smooth_gaussian(im:np.ndarray, sigma) -> np.ndarray:

    if sigma == 0:
        return im

    im_smooth = im.astype(float)
    kernel_x = np.arange(-3*sigma,3*sigma+1).astype(float)
    kernel_x = np.exp((-(kernel_x**2))/(2*(sigma**2)))

    im_smooth = ndimage.convolve(im_smooth, kernel_x[np.newaxis])

    im_smooth = ndimage.convolve(im_smooth, kernel_x[np.newaxis].T)

    return im_smooth


def gradient(im_smooth:np.ndarray):

    gradient_x = im_smooth.astype(float)
    gradient_y = im_smooth.astype(float)

    kernel = np.arange(-1,2).astype(float)
    kernel = - kernel / 2

    gradient_x = ndimage.convolve(gradient_x, kernel[np.newaxis])
    gradient_y = ndimage.convolve(gradient_y, kernel[np.newaxis].T)

    return gradient_x,gradient_y


def sobel(im_smooth):
    gradient_x = im_smooth.astype(float)
    gradient_y = im_smooth.astype(float)

    kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    gradient_x = ndimage.convolve(gradient_x, kernel)
    gradient_y = ndimage.convolve(gradient_y, kernel.T)

    return gradient_x,gradient_y


def compute_normal_map(gradient_x:np.ndarray, gradient_y:np.ndarray, intensity=1):

    width = gradient_x.shape[1]
    height = gradient_x.shape[0]
    max_x = np.max(gradient_x)
    max_y = np.max(gradient_y)

    max_value = max_x

    if max_y > max_x:
        max_value = max_y

    normal_map = np.zeros((height, width, 3), dtype=np.float32)

    intensity = 1 / intensity

    strength = max_value / (max_value * intensity)

    normal_map[..., 0] = gradient_x / max_value
    normal_map[..., 1] = gradient_y / max_value
    normal_map[..., 2] = 1 / strength

    norm = np.sqrt(np.power(normal_map[..., 0], 2) + np.power(normal_map[..., 1], 2) + np.power(normal_map[..., 2], 2))

    normal_map[..., 0] /= norm
    normal_map[..., 1] /= norm
    normal_map[..., 2] /= norm

    normal_map *= 0.5
    normal_map += 0.5

    return normal_map

def normalized(a) -> float: 
    factor = 1.0/math.sqrt(np.sum(a*a)) # normalize
    return a*factor

def my_gauss(im:np.ndarray):
    return ndimage.uniform_filter(im.astype(float),size=20)

def shadow(im:np.ndarray):
    
    shadowStrength = .5
    
    im1 = im.astype(float)
    im0 = im1.copy()
    im00 = im1.copy()
    im000 = im1.copy()

    for _ in range(0,2):
        im00 = my_gauss(im00)

    for _ in range(0,16):
        im0 = my_gauss(im0)

    for _ in range(0,32):
        im1 = my_gauss(im1)

    im000=normalized(im000)
    im00=normalized(im00)
    im0=normalized(im0)
    im1=normalized(im1)
    im00=normalized(im00)

    shadow=im00*2.0+im000-im1*2.0-im0 
    shadow=normalized(shadow)
    mean = np.mean(shadow)
    rmse = np.sqrt(np.mean((shadow-mean)**2))*(1/shadowStrength)
    shadow = np.clip(shadow, mean-rmse*2.0,mean+rmse*0.5)

    return shadow

def flipgreen(path:str):
    try:
        with Image.open(path) as img:
            red, green, blue, alpha= img.split()
            image = Image.merge("RGB",(red,ImageOps.invert(green),blue))
            image.save(path)
    except ValueError:
        with Image.open(path) as img:
            red, green, blue = img.split()
            image = Image.merge("RGB",(red,ImageOps.invert(green),blue))
            image.save(path)

def CleanupAO(path:str):
    '''
    Remove unnsesary channels.
    '''
    try:
        with Image.open(path) as img:
            red, green, blue, alpha= img.split()
            NewG = ImageOps.colorize(green,black=(100, 100, 100),white=(255,255,255),blackpoint=0,whitepoint=180)
            NewG.save(path)
    except ValueError:
        with Image.open(path) as img:
            red, green, blue = img.split()
            NewG = ImageOps.colorize(green,black=(100, 100, 100),white=(255,255,255),blackpoint=0,whitepoint=180)
            NewG.save(path)

def adjustPath(Org_Path:str,addto:str):
    '''
    Adjust the given path to correctly save the new file.
    '''

    path = Org_Path.split("/")
    file = path[-1]
    print(file)
    filename = file.split(".")[0]
    fileext = file.split(".")[-1]

    newfilename = addto+"/"+filename + "_" + addto + "." + fileext
    path.pop(-1)
    path.append(newfilename)

    newpath = '/'.join(path)

    return newpath

def Convert(input_file, smoothness, intensity, mask=None):
    """
    Convert an image to normal map and ambient occlusion map.
    
    Args:
        input_file: Path to input image
        smoothness: Gaussian blur smoothness
        intensity: Normal map intensity
        mask: Optional binary mask to apply to the image (1 for face, 0 for background)
        
    Returns:
        tuple: (normal_map, ao_map)
    """
    im = pyplot.imread(input_file)

    if im.ndim == 3:
        im_grey = np.zeros((im.shape[0],im.shape[1])).astype(float)
        im_grey = (im[...,0] * 0.3 + im[...,1] * 0.6 + im[...,2] * 0.1)
        im = im_grey

    # Apply mask if provided
    if mask is not None:
        # Ensure mask is boolean
        mask = mask.astype(bool)
        # Create a background value (use the average of the image edges)
        background_value = np.mean(im[~mask])
        # Apply mask to image
        im = im * mask + background_value * (~mask)

    im_smooth = smooth_gaussian(im, smoothness)

    sobel_x, sobel_y = sobel(im_smooth)

    # Flip the X and Y gradients to match our coordinate system
    sobel_x = -sobel_x  # Flip X to match left (-1) to right (+1)
    sobel_y = -sobel_y  # Flip Y to match down (-1) to up (+1)

    normal_map = compute_normal_map(sobel_x, sobel_y, intensity)

    # Apply mask to normal map if provided
    if mask is not None:
        # Set background to neutral normal (0.5, 0.5, 1.0)
        background_normal = np.array([0.5, 0.5, 1.0])
        normal_map[~mask] = background_normal

    # Convert normal map to BGR for OpenCV compatibility
    normal_map = cv2.cvtColor((normal_map * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    im_shadow = shadow(im)

    # Apply mask to ambient occlusion if provided
    if mask is not None:
        # Set background to white (1.0)
        im_shadow[~mask] = 1.0

    # Convert AO map to BGR for OpenCV compatibility
    ao_map = cv2.cvtColor((im_shadow * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    return normal_map, ao_map

def startConvert(input_file=None, smooth=0., intensity=1., mask=None):
    """
    Start the conversion process.
    
    Args:
        input_file: Path to input file or directory
        smooth: Gaussian blur smoothness
        intensity: Normal map intensity
        mask: Optional binary mask to apply to the image (1 for face, 0 for background)
        
    Returns:
        tuple: (normal_map, ao_map) if input is a file, or None if input is a directory
    """
    if input_file is None:
        parser = argparse.ArgumentParser(description='Compute normal map of an image')
        parser.add_argument('input_file', type=str, help='input folder path')
        parser.add_argument('-s', '--smooth', default=0., type=float, help='smooth gaussian blur applied on the image')
        parser.add_argument('-it', '--intensity', default=1., type=float, help='intensity of the normal map')
        args = parser.parse_args()
        input_file = args.input_file
        smooth = args.smooth
        intensity = args.intensity
    
    input_path = input_file
    
    # Check if input is a file or directory
    if os.path.isfile(input_path):
        # Process a single file
        return Convert(input_path, smooth, intensity, mask)
    else:
        # Process all files in directory
        files_to_process = []
        for root, _, files in os.walk(input_path, topdown=False):
            for name in files:
                if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    file_path = os.path.join(root, name)
                    files_to_process.append(file_path)
        
        # Process files
        if len(files_to_process) == 0:
            print("No image files found in the directory.")
        else:
            for file_path in files_to_process:
                try:
                    ctx = mp.get_context('spawn')
                    p = ctx.Process(target=Convert, args=(file_path, smooth, intensity, mask))
                    p.start()
                    p.join()
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        return None

# Add main block to call startConvert() when script is run
if __name__ == "__main__":
    startConvert()