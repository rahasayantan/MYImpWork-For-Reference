import os; os.environ['OMP_NUM_THREADS'] = '1'
from PIL import Image
from zipfile import ZipFile
from collections import defaultdict
from scipy.stats import itemfreq
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import feature
import numpy as np
import pandas as pd 
import operator
import cv2
import os 

## Dullness
def color_analysis(img):
    # obtain the color palatte of the image 
    palatte = defaultdict(int)
    for pixel in img.getdata():
        palatte[pixel] += 1
    
    # sort the colors present in the image 
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]
        
    light_percent = round((float(light_shade)/shade_count)*100, 2)
    dark_percent = round((float(dark_shade)/shade_count)*100, 2)
    return light_percent, dark_percent
    
def perform_color_analysis(img, flag):
    path = images_path + img 
    try:
        im = Image.open(path) #.convert("RGB")
        print('color'+path)
        # cut the images into two halves as complete average may give bias results
        size = im.size
        halves = (size[0]/2, size[1]/2)
        im1 = im.crop((0, 0, size[0], halves[1]))
        im2 = im.crop((0, halves[1], size[0], size[1]))

        light_percent1, dark_percent1 = color_analysis(im1)
        light_percent2, dark_percent2 = color_analysis(im2)
    except Exception as e:
        return None

    light_percent = (light_percent1 + light_percent2)/2 
    dark_percent = (dark_percent1 + dark_percent2)/2 
    
    if flag == 'black':
        return dark_percent
    elif flag == 'white':
        return light_percent
    else:
        return None
        
def average_pixel_width(img):
    path = images_path + img 
    try:
        print('pixel'+path)
        im = Image.open(path)    
        im_array = np.asarray(im.convert(mode='L'))
        edges_sigma1 = feature.canny(im_array, sigma=3)
        apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
        return apw*100
    except Exception as e:
        return None    

## Dominant color analysis
def get_dominant_color(img):
    try:
        path = images_path + img 
        print('domColor'+path)
        img = cv2.imread(path)
        arr = np.float32(img)
        pixels = arr.reshape((-1, 3))

        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

        palette = np.uint8(centroids)
        quantized = palette[labels.flatten()]
        quantized = quantized.reshape(img.shape)

        dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]
        return dominant_color
    except Exception as e:
        return None

##Avergae Color    
'''
def get_average_color(img):
    try:
        path = images_path + img 
        print(path)
        img = cv2.imread(path)
        average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
        return average_color
    except Exception as e:
        return None
'''
## Image Dimension

def getSize(filename):
    try:
        filename = images_path + filename
        print('size')
        st = os.stat(filename)
        return st.st_size
    except Exception as e:
        return None

def getDimensions(filename):
    try:
        filename = images_path + filename
        print('dim')
        img_size = Image.open(filename).size
        return img_size 
    except Exception as e:
        return None
        
## Is the image blurry

def get_blurrness_score(image):
    try:
        path =  images_path + image 
        print('blur'+path)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(image, cv2.CV_64F).var()
        return fm
    except Exception as e:
        return None

images_paths = ['../input/train_jpg_0/', '../input/train_jpg_1/', '../input/train_jpg_2/', '../input/train_jpg_3/', '../input/train_jpg_4/']

train_df = pd.read_csv('../input/train.csv',  parse_dates=["activation_date"])
test_df = pd.read_csv('../input/test.csv',  parse_dates=["activation_date"])
train_df = train_df[['item_id','image']]
test_df = train_df[['item_id','image']]

train_df['image'] = train_df['image']+'.jpg'

dfLst = []
for images_path in images_paths:
    imgs = os.listdir(images_path)
#    imgs = imgs[:100]
    features = pd.DataFrame()
    features['image'] = imgs
    features['folder'] = images_path
    dfLst.append(features)
    
features = pd.concat(dfLst, axis = 'rows')
features.reset_index(drop = True, inplace = True)

train_df = train_df.merge(features, how='left', on = 'image')
    
dfLst = []
for images_path in images_paths:
    print(images_path)
    imgs = list(train_df.loc[train_df.folder == images_path, 'image'].values) #os.listdir(images_path)
    
    features = pd.DataFrame()
    features['image'] = imgs
    ## Average edge width

#    im1 = Image.open(images_path+imgs[0])
#    im2 = im1.convert(mode='L')
#    im = np.asarray(im2)

#    edges1 = feature.canny(im, sigma=1)
#    edges2 = feature.canny(im, sigma=3)
    features['average_pixel_width'] = features['image'].apply(average_pixel_width)

    features['dominant_color'] = features['image'].apply(get_dominant_color)
    features['dominant_red'] = features['dominant_color'].apply(lambda x: x[0]) / 255
    features['dominant_green'] = features['dominant_color'].apply(lambda x: x[1]) / 255
    features['dominant_blue'] = features['dominant_color'].apply(lambda x: x[2]) / 255
#    features[['dominant_red', 'dominant_green', 'dominant_blue']].head(5)

    features['average_color'] = features['image'].apply(get_dominant_color)
    features['average_red'] = features['average_color'].apply(lambda x: x[0]) / 255
    features['average_green'] = features['average_color'].apply(lambda x: x[1]) / 255
    features['average_blue'] = features['average_color'].apply(lambda x: x[2]) / 255

#    features[['average_red', 'average_green', 'average_blue']].head(5)
    features['image_size'] = features['image'].apply(getSize)
    features['temp_size'] = features['image'].apply(getDimensions)
    features['width'] = features['temp_size'].apply(lambda x : x[0])
    features['height'] = features['temp_size'].apply(lambda x : x[1])
    print("avg")
    
    features['blurrness'] = features['image'].apply(get_blurrness_score)
    print("blurrness")    

    features['dullness'] = features['image'].apply(lambda x : perform_color_analysis(x, 'black'))
    print("dullness")
    ## Image whiteness
    features['whiteness'] = features['image'].apply(lambda x : perform_color_analysis(x, 'white'))
    print("whiteness")

    dfLst.append(features)

features = pd.concat(dfLst, axis = 'rows')
features.reset_index(drop = True, inplace = True)
features.drop(['dominant_color','temp_size','average_color'], inplace = True)


#dfLst = []
#for i in np.arange(train_df.shape[0]):
#    imgpath = train_df.iloc[i]['folder'] + train_df.iloc[i]['image']
    
    
### Exif = not giving anythin for the first 2 images - will park
'''from PIL import Image, ExifTags

img = Image.open('../input/images_sample/6811960/6811960_3685d3542328b820980642535d8ccb72.jpg')
ex = img._getexif()
if ex != None:
    for (k,v) in img._getexif().items():
            print (ExifTags.TAGS.get(k), v)    
'''
