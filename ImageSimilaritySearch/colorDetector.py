from sklearn.cluster import KMeans
import argparse
import numpy as np
import cv2
import os
import webcolors
from PIL import Image

def pre_process(input_image, output_path):
    image = Image.open(input_image)
    image = image.resize((250, 250), Image.ANTIALIAS)
    
    file_name = os.path.basename(input_image)
    resized_file = os.path.join(output_path, file_name)
    image.save(resized_file)

    image = cv2.imread(resized_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)[1]

    cv2.imwrite(os.path.join(output_path,"gray.png"), gray)
    cv2.imwrite(os.path.join(output_path,"blurred.png"), blurred)
    cv2.imwrite(os.path.join(output_path,"thresh.png"), thresh)

    count_white = 0
    count_black = 0

    for i in range(0, len(thresh)):
        for j in range(0, len(thresh[0])):
            if (thresh[i][j] == 0): # black pixels
                count_black += 1
            elif (thresh[i][j] == 255): # white pixels
                count_white += 1

    color = 0 # black pixels
    if count_black < 10000:
        color = 255 # white pixels

    for i in range(0, len(thresh)):
        for j in range(0, len(thresh[0])):
            if (thresh[i][j] != color):
                # change background color to white/black
                image[i][j] = [color,color,color]

    cv2.imwrite(os.path.join(output_path, 'output.jpg'), image)

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - 2*requested_colour[0]) ** 2
        gd = (g_c - 2*requested_colour[1]) ** 2
        bd = (b_c - 2*requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        name =  webcolors.rgb_to_name(requested_colour)
    except ValueError:
        name = closest_colour(requested_colour)
    return name

def get_shade(name):
    color_map = {}
    color_map["pink"] = ["pink","lightpink","hotpink","deeppink","palevioletred","mediumvioletred"]
    color_map["red"] = ["lightsalmon","salmon","darksalmon","lightcoral","indianred","crimson","firebrick","darkred","red"]
    color_map["orange"] = ["orangered","tomato","coral","darkorange","orange"]
    color_map["yellow"] = ["yellow","lightyellow","lemonchiffon","lightgoldenrodyellow","papayawhip","moccasin","peachpuff","palegoldenrod","khaki","darkkhaki","gold"]
    color_map["brown"] = ["cornsilk","blanchedalmond","bisque","navajowhite","wheat","burlywood","tan","rosybrown","sandybrown","goldenrod","darkgoldenrod","peru","chocolate","saddlebrown","sienna","brown","maroon"]
    color_map["green"] = ["darkolivegreen","olive","olivedrab","yellowgreen","limegreen","lime","lawngreen","chartreuse","greenyellow","springgreen","mediumspringgreen","lightgreen","palegreen","darkseagreen","mediumaquamarine","mediumseagreen","seagreen","forestgreen","green","darkgreen"]
    color_map["cyan"] = ["aqua","cyan","lightcyan","paleturquoise","aquamarine","turquoise","mediumturquoise","darkturquoise","lightseagreen","cadetblue","darkcyan","teal"]
    color_map["blue"] = ["lightsteelblue","powderblue","lightblue","skyblue","lightskyblue","deepskyblue","dodgerblue","cornflowerblue","steelblue","royalblue","blue","mediumblue","darkblue","navy","midnightblue","darkslateblue","slateblue","mediumslateblue"]
    color_map["purple"] = ["lavender","thistle","plum","violet","orchid","fuchsia","magenta","mediumorchid","mediumpurple","blueviolet","darkviolet","darkorchid","darkmagenta","purple","indigo"]
    color_map["white"] = ["white","snow","honeydew","mintcream","azure","aliceblue","ghostwhite","whitesmoke","seashell","beige","oldlace","floralwhite","ivory","antiquewhite","linen","lavenderblush","mistyrose"]
    color_map["black"] = ["gainsboro","lightgray","silver","darkgray","gray","dimgray","lightslategray","slategray","darkslategray","black"]

    for key in color_map.keys():
        if name in color_map[key]:
            return key

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    dominant_color = None
    dominant_percent = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX
        if percent > dominant_percent:
            dominant_color = color
            dominant_percent = percent
    return dominant_color


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='detect object color')
    parser.add_argument("--image", help="path of image")
    parser.add_argument("--prefix", help="path prefix of intermediate image")
    args = parser.parse_args()
    image_path = args.image
    path_prefix = args.prefix
    output_path = os.path.join(path_prefix, "output.jpg")
    # path_prefix = "/Users/xinyanl/Desktop/hack14/"
    # image_path = "/Users/xinyanl/Desktop/hack14/4_cropped.jpg"
    # output_path = "/Users/xinyanl/Desktop/hack14/output.jpg"

    #pre-process
    pre_process(image_path, path_prefix)

    # get dominant color
    image = cv2.imread(output_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=1)
    clt.fit(image)
    hist = centroid_histogram(clt)
    color = plot_colors(hist, clt.cluster_centers_)

     #get color name
    r = color[0]
    g = color[1]
    b = color[2]
    requested_colour = (r, g, b)
    name = get_colour_name(requested_colour)
    print get_shade(name)



