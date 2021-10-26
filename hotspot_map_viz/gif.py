import imageio
import glob
from PIL import Image

files = glob.glob("raw_prob/*")
files.sort()
print(files)
i = 0
for file in files:
    im = Image.open(file)
    width, height = im.size
    left = 600
    top = height / 3
    right = 1350
    bottom = 3 * height / 4
     
    # Cropped image of above dimension
    # (It will not change original image)
    im1 = im.crop((left, top, right, bottom))
    im1.save("model_%d.png"%i)
    # Shows the image in image viewer
    #im1.show()
    i += 1

images = []
for i in range(48):
    images.append(imageio.imread("model_%d.png"%i))
imageio.mimsave('model_hotspots.gif', images, duration=1)