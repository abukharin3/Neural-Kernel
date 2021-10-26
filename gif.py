import imageio
import glob
from PIL import Image

files = glob.glob("ci_png/*")
files.sort()
print(files)
i = 0
for file in files:
    im = Image.open(file)
    width, height = im.size
    left = 300
    top = height / 9
    right = 1920
    bottom =  18 * height / 20
     
    # Cropped image of above dimension
    # (It will not change original image)
    im1 = im.crop((left, top, right, bottom))
    im1.save("model_%d.png"%i)
    # Shows the image in image viewer
    #im1.show()
    i += 1

images = []
for i in range(49):
    images.append(imageio.imread("model_%d.png"%i))
imageio.mimsave('ci.gif', images, duration=1)