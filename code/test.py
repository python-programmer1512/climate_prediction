import math
from PIL import Image, ImageDraw 
from PIL import ImagePath  
side = 8
xy = [ 
    ((math.cos(th) + 1) * 90, 
     (math.sin(th) + 1) * 60) 
    for th in [i * (2 * math.pi) / side for i in range(side)] 
    ]   
  
  
print(xy)


image = ImagePath.Path(xy).getbbox()   
size = list(map(int, map(math.ceil, image[2:]))) 
  
img = Image.new("RGB", size, "# f9f9f9")  
img1 = ImageDraw.Draw(img)   
img1.polygon(xy, fill ="# eeeeff", outline ="blue")  
  
img.show() 