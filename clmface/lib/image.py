import numpy as np

# extract image centered at (x, y) which is pw x pl
# if image out of bound, pad with 0s
def getImageData(img, x, y, pw, pl):
    assert pw%2==1 and pl%2==1
    result = np.zeros((pl, pw))
    height, width = img.shape
    if y-(pl/2)<0:
        starty = 0
        basey = pl/2-y
    else:
        starty=y-(pl/2)
        basey=0
    if x-(pw/2)<0:
        startx = 0
        basex = pw/2-x
    else:
        startx=x-(pw/2)
        basex=0
    if y+pl/2+1>height:
        endy=height
    else:
        endy=y+pl/2+1
    if x+pw/2+1>width:
        endx=width
    else:
        endx=x+pw/2+1
    result[basey:(endy-starty+basey), basex:(endx-startx+basex)] = img[starty:endy, startx:endx]
    return result
