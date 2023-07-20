#
# NOTE: based on skimage.morphology.remove_small_objects()
#
import numpy as np
from scipy import ndimage as ndi

def remove_large_objects(bool_array, cull=0):
    out = bool_array.copy()
    if cull==0:  # shortcut for efficiency
        return out

    selem = ndi.generate_binary_structure(bool_array.ndim, 1)
    ccs = np.zeros_like(bool_array, dtype=np.int32)
    ndi.label(bool_array, selem, output=ccs)

    component_sizes = np.bincount(ccs.ravel())
    s = np.sort(component_sizes[1:])
    max_size = s[int(len(s)*(100-cull)/100)]
    too_big = component_sizes >= max_size
    too_big_mask = too_big[ccs]
    out[too_big_mask] = 0

    return out

