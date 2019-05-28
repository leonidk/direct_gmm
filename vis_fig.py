from pylab import *
import scipy.stats
from matplotlib import rc
rc('font',**{'family':'mono','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.style.use('seaborn-pastel')
means = [-2,0,1]
stds = [1,0.8,0.6]
weights = [1/4,1/4,1/2]

xs = np.linspace(-5,5,200)

plt.figure(figsize=(9,3))
all_ys = np.zeros_like(xs)
for i in range(3):
    ys = weights[i]*scipy.stats.norm(means[i],stds[i]).pdf(xs)
    all_ys+=ys
    plt.plot(xs,ys,label=r'$\mu=${} $\sigma=${:.1f} $\lambda=${:.2f}'.format(means[i],stds[i],weights[i]))

plt.plot(xs,all_ys,label='GMM',c='k',lw=4)
plt.legend()
plt.tight_layout()
plt.savefig('thing.png',dpi=300)
plt.show()
