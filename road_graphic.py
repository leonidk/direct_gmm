from pylab import *

from skimage import io, filters
img =  io.imread('aerial.jpeg')
diff = (abs(img - img.mean(2,keepdims=True))).mean(2) * (img.mean(2))
diff = diff**2
diff = diff/diff.max()*255*12*10
diff = np.minimum(diff,255)

plt.imshow(diff.astype(np.uint8))
clean_diff = 255-filters.rank.median(diff.astype(np.uint8),np.ones((51,51)))
crop = clean_diff[400:1100:4,2700:3300:4].astype(np.float)
crop = crop/crop.sum()
plt.imshow(crop)
plt.colorbar()
plt.tight_layout()

plt.show()
#diff = np.exp(-diff**2/10)
#diff = diff/diff.max()*255
#plt.show()