# Whatever the vision task (object detection, recognition, classification),
# the work begins with image processing...  

# This program shows how numpy arrays may be generated from image files
# using the Open CV library and various Python utilities.

# We generate grayscale and color representations in numpy arrays
# for use in subsequent solutions for the binary classification task.

# Program developed by Thomas W. Miller, August 7, 2018 

import os # operating system functions, shutil # high-level file operations
import os.path  # for manipulation of file path names
import numpy as np

from matplotlib import pyplot as plt  # for display of images

# Open CV for image processing
# Installed on Mac with pip install opencv-python 
import cv2  

# Original data from Kaggle (we use the first 1000 cats and 1000 dogs):
#   https://www.kaggle.com/c/dogs-vs-cats 

# The paths to the directories where the original raster files are located
cat_image_dir_name = \
    r'\Users\johnk\Desktop\Grad School\3. Summer 2018\1. MSDS 422 - ML\6. Homework\Week7\cats_dogs_images\cats'
dog_image_dir_name = \
    r'\Users\johnk\Desktop\Grad School\3. Summer 2018\1. MSDS 422 - ML\6. Homework\Week7\cats_dogs_images\dogs'
   
# "Human" sorting of file names facilitated by
# https://nedbatchelder.com/blog/200712/human_sorting.html
import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    
# Generate nicely sorted list of file names, excluding hidden files    
def directory_list (dir_name):
    start_list = os.listdir(dir_name)
    end_list = []
    for file in start_list:
        if (not file.startswith('.')):
            end_list.append(file) 
    end_list.sort(key = alphanum_key)        
    return(end_list)        

cat_file_names = directory_list(cat_image_dir_name)
dog_file_names = directory_list(dog_image_dir_name)    
    
# Convert image to numpy array... 
# Three channels for color converted to grayscale
def parse_grayscale(image_file_path):
    image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
    return(image)
    
# Convert image to numpy array... three channels for color
def parse_color(image_file_path):
    image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
    # Default cv2 is BGR... need RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return(image)
  
def parse_grayscale_and_resize(image_file_path, size = (64, 64)):
    image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, size)
    return(image)

def parse_color_and_resize(image_file_path, size = (64, 64)):
    image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
    # Default cv2 is BGR... need RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    return(image)  
    
def show_grayscale_image(image):
    plt.imshow(image, cmap = 'gray') 
    plt.axis('off')
    plt.show()

def show_color_image(image):
    plt.imshow(image) 
    plt.axis('off')
    plt.show()   
        
# Sample commands to convert color image to grayscale and display to screen
# Note the original images are (374, 500, 3)  
# image_file_path = os.path.join(cat_image_dir_name, cat_file_names[0])        
# image = parse_grayscale(image_file_path)
# image.shape  # shows size of the numpy array    
# show_grayscale_image(image)
# And to show the original color image    
# image = parse_color(image_file_path)
# show_color_image(image)
# image.shape  # shows size of the numpy array 
# Work with resized color image using default size 64x64    
# image = parse_color_and_resize(image_file_path)
# show_color_image(image)
# image.shape  # shows size of the numpy array  
# Work with resized color image using size 128x128    
# image = parse_color_and_resize(image_file_path, size = (128, 128))
# show_color_image(image)
# image.shape  # shows size of the numpy array      
# Work with resized color image using size 256x256    
# image = parse_color_and_resize(image_file_path, size = (256, 256))
# show_color_image(image)
# image.shape  # shows size of the numpy array 
  
# ----------------------------------------------------------------------
# Examine dimensions of original raster images 
# Results show considerable variability in image pixel dimensions
cats_shapes = []
for ifile in range(len(cat_file_names)):
    image_file_path = os.path.join(cat_image_dir_name, cat_file_names[ifile])
    image = parse_color(image_file_path)
    cats_shapes.append(image.shape)
print('\n\nCat image file shapes:\n')    
print(cats_shapes)    

dogs_shapes = []
for ifile in range(len(cat_file_names)):
    image_file_path = os.path.join(cat_image_dir_name, cat_file_names[ifile])
    image = parse_color(image_file_path)
    dogs_shapes.append(image.shape)    
print('\n\nDog image file shapes:\n') 
print(dogs_shapes)
  
# ----------------------------------------------------------------------
print('\nProcessing image files to 64x64 color or grayscale arrays')
# Create cats_1000_64_64_3 and numpy array for 1000 cat images in color
cats_1000_64_64_3 = np.zeros((1000, 64, 64, 3))  
for ifile in range(len(cat_file_names)):
    image_file_path = os.path.join(cat_image_dir_name, cat_file_names[ifile])
    image = parse_color_and_resize(image_file_path, size = (64, 64))
    cats_1000_64_64_3[ifile,:,:,:] = image
       
# Create dogs_1000_64_64_3 and numpy array for 1000 dog images in color   
dogs_1000_64_64_3 = np.zeros((1000, 64, 64, 3))  
for ifile in range(len(dog_file_names)):
    image_file_path = os.path.join(dog_image_dir_name, dog_file_names[ifile])
    image = parse_color_and_resize(image_file_path, size = (64, 64))
    dogs_1000_64_64_3[ifile,:,:,:] = image

# Create cats_1000_64_64_1 and numpy array for 1000 cat images in grayscale
cats_1000_64_64_1 = np.zeros((1000, 64, 64, 1))  
for ifile in range(len(cat_file_names)):
    image_file_path = os.path.join(cat_image_dir_name, cat_file_names[ifile])
    image = parse_grayscale_and_resize(image_file_path, size = (64, 64))
    cats_1000_64_64_1[ifile,:,:,0] = image
       
# Create dogs_1000_64_64_1 and numpy array for 1000 dog images in grayscale   
dogs_1000_64_64_1 = np.zeros((1000, 64, 64, 1))  
for ifile in range(len(dog_file_names)):
    image_file_path = os.path.join(dog_image_dir_name, dog_file_names[ifile])
    image = parse_grayscale_and_resize(image_file_path, size = (64, 64))
    dogs_1000_64_64_1[ifile,:,:,0] = image

# ------------------------------------------------------------------------
print('\nProcessing image files to 128x128 color or grayscale arrays')
# Create cats_1000_128_128_3 and numpy array for 1000 cat images in color
cats_1000_128_128_3 = np.zeros((1000, 128, 128, 3))  
for ifile in range(len(cat_file_names)):
    image_file_path = os.path.join(cat_image_dir_name, cat_file_names[ifile])
    image = parse_color_and_resize(image_file_path, size = (128, 128))
    cats_1000_128_128_3[ifile,:,:,:] = image
       
# Create dogs_1000_128_128_3 and numpy array for 1000 dog images in color   
dogs_1000_128_128_3 = np.zeros((1000, 128, 128, 3))  
for ifile in range(len(dog_file_names)):
    image_file_path = os.path.join(dog_image_dir_name, dog_file_names[ifile])
    image = parse_color_and_resize(image_file_path, size = (128, 128))
    dogs_1000_128_128_3[ifile,:,:,:] = image

# Create cats_1000_128_128_1 and numpy array for 1000 cat images in grayscale
cats_1000_128_128_1 = np.zeros((1000, 128, 128, 1))  
for ifile in range(len(cat_file_names)):
    image_file_path = os.path.join(cat_image_dir_name, cat_file_names[ifile])
    image = parse_grayscale_and_resize(image_file_path, size = (128, 128))
    cats_1000_128_128_1[ifile,:,:,0] = image
       
# Create dogs_1000_128_128_1 and numpy array for 1000 dog images in grayscale   
dogs_1000_128_128_1 = np.zeros((1000, 128, 128, 1))  
for ifile in range(len(dog_file_names)):
    image_file_path = os.path.join(dog_image_dir_name, dog_file_names[ifile])
    image = parse_grayscale_and_resize(image_file_path, size = (128, 128))
    dogs_1000_128_128_1[ifile,:,:,0] = image   

# ------------------------------------------------------------------------
print('\nProcessing image files to 256x256 color or grayscale arrays')
# Create cats_1000_256_256_3 and numpy array for 1000 cat images in color
cats_1000_256_256_3 = np.zeros((1000, 256, 256, 3))  
for ifile in range(len(cat_file_names)):
    image_file_path = os.path.join(cat_image_dir_name, cat_file_names[ifile])
    image = parse_color_and_resize(image_file_path, size = (256, 256))
    cats_1000_256_256_3[ifile,:,:,:] = image
       
# Create dogs_1000_256_256_3 and numpy array for 1000 dog images in color   
dogs_1000_256_256_3 = np.zeros((1000, 256, 256, 3))  
for ifile in range(len(dog_file_names)):
    image_file_path = os.path.join(dog_image_dir_name, dog_file_names[ifile])
    image = parse_color_and_resize(image_file_path, size = (256, 256))
    dogs_1000_256_256_3[ifile,:,:,:] = image

# Create cats_1000_256_256_1 and numpy array for 1000 cat images in grayscale
cats_1000_256_256_1 = np.zeros((1000, 256, 256, 1))  
for ifile in range(len(cat_file_names)):
    image_file_path = os.path.join(cat_image_dir_name, cat_file_names[ifile])
    image = parse_grayscale_and_resize(image_file_path, size = (256, 256))
    cats_1000_256_256_1[ifile,:,:,0] = image
       
# Create dogs_1000_256_256_1 and numpy array for 1000 dog images in grayscale   
dogs_1000_256_256_1 = np.zeros((1000, 256, 256, 1))  
for ifile in range(len(dog_file_names)):
    image_file_path = os.path.join(dog_image_dir_name, dog_file_names[ifile])
    image = parse_grayscale_and_resize(image_file_path, size = (256, 256))
    dogs_1000_256_256_1[ifile,:,:,0] = image  
    
# ------------------------------------------------------------------------
print('\nProcessing image files to 512x512 color or grayscale arrays')
# Create cats_1000_512_512_3 and numpy array for 1000 cat images in color
cats_1000_512_512_3 = np.zeros((1000, 512, 512, 3))  
for ifile in range(len(cat_file_names)):
    image_file_path = os.path.join(cat_image_dir_name, cat_file_names[ifile])
    image = parse_color_and_resize(image_file_path, size = (512, 512))
    cats_1000_512_512_3[ifile,:,:,:] = image
       
# Create dogs_1000_512_512_3 and numpy array for 1000 dog images in color   
dogs_1000_512_512_3 = np.zeros((1000, 512, 512, 3))  
for ifile in range(len(dog_file_names)):
    image_file_path = os.path.join(dog_image_dir_name, dog_file_names[ifile])
    image = parse_color_and_resize(image_file_path, size = (512, 512))
    dogs_1000_512_512_3[ifile,:,:,:] = image

# Create cats_1000_512_512_1 and numpy array for 1000 cat images in grayscale
cats_1000_512_512_1 = np.zeros((1000, 512, 512, 1))  
for ifile in range(len(cat_file_names)):
    image_file_path = os.path.join(cat_image_dir_name, cat_file_names[ifile])
    image = parse_grayscale_and_resize(image_file_path, size = (512, 512))
    cats_1000_512_512_1[ifile,:,:,0] = image
       
# Create dogs_1000_512_512_1 and numpy array for 1000 dog images in grayscale   
dogs_1000_512_512_1 = np.zeros((1000, 512, 512, 1))  
for ifile in range(len(dog_file_names)):
    image_file_path = os.path.join(dog_image_dir_name, dog_file_names[ifile])
    image = parse_grayscale_and_resize(image_file_path, size = (512, 512))
    dogs_1000_512_512_1[ifile,:,:,0] = image   
    
    
# Documentation on npy binary format for saving numpy arrays for later use
#     https://towardsdatascience.com/
#             why-you-should-start-using-npy-file-more-often-df2a13cc0161 
    
# The directory where we store the numpy array objects
# store our smaller dataset
outdir = r'Users\johnk\Desktop\Grad School\3. Summer 2018\1. MSDS 422 - ML\6. Homework\Week7'
os.mkdir(outdir)    
       
np.save(os.path.join(outdir, 'cats_1000_64_64_3.npy'), cats_1000_64_64_3)
np.save(os.path.join(outdir, 'dogs_1000_64_64_3.npy'), dogs_1000_64_64_3)
np.save(os.path.join(outdir, 'cats_1000_64_64_1.npy'), cats_1000_64_64_1)
np.save(os.path.join(outdir, 'dogs_1000_64_64_1.npy'), dogs_1000_64_64_1)

np.save(os.path.join(outdir, 'cats_1000_128_128_3.npy'), cats_1000_128_128_3)
np.save(os.path.join(outdir, 'dogs_1000_128_128_3.npy'), dogs_1000_128_128_3)
np.save(os.path.join(outdir, 'cats_1000_128_128_1.npy'), cats_1000_128_128_1)
np.save(os.path.join(outdir, 'dogs_1000_128_128_1.npy'), dogs_1000_128_128_1)

np.save(os.path.join(outdir, 'cats_1000_256_256_3.npy'), cats_1000_256_256_3)
np.save(os.path.join(outdir, 'dogs_1000_256_256_3.npy'), dogs_1000_256_256_3)
np.save(os.path.join(outdir, 'cats_1000_256_256_1.npy'), cats_1000_256_256_1)
np.save(os.path.join(outdir, 'dogs_1000_256_256_1.npy'), dogs_1000_256_256_1)

np.save(os.path.join(outdir, 'cats_1000_512_512_3.npy'), cats_1000_512_512_3)
np.save(os.path.join(outdir, 'dogs_1000_512_512_3.npy'), dogs_1000_512_512_3)
np.save(os.path.join(outdir, 'cats_1000_512_512_1.npy'), cats_1000_512_512_1)
np.save(os.path.join(outdir, 'dogs_1000_512_512_1.npy'), dogs_1000_512_512_1)
    
# Note. Due to file size issues, only the 64x64 and 128x128 files were 
# uploaded to the Canas cours site. These files are in the zip archive
#   cats_dogs_64_128.zip    
    
print('\nRun complete')    
    
    
    
    