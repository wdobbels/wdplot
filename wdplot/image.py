'''
Functions for plotting images.
'''
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm

def show_galaxy(img, **kwargs):
    '''
    Plot an image on a log-scale, perfect for galaxies!
    '''
    
    kwargs.setdefault('norm', simple_norm(img, stretch='log', percent=99))
    kwargs.setdefault('cmap', 'inferno')
    kwargs.setdefault('vmin', 0)
    if not 'ax' in kwargs:
        plt.figure(figsize=kwargs.pop('figsize', (10, 10)))
        ax = plt.gca()
    else:
        ax = kwargs.pop('ax')
    ax.imshow(img, origin='lower', **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])