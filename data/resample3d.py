# Resample.py
# Andrew Brock, 2017
# This code resamples a 3d grid using catmull-rom spline interpolation, and is GPU accelerated.

# Resample along the trailing dimension
# Assumes a more-than-1D array? Or just directly assumes a 3D array? we'll find out
# 

# TODO: Some things could be shared (such as the mgrid call, which can presumably be done once? hmm)
#       between resample1d calls.
# Since coords[-1] is just range(0,out_shape[-1]) repeated a bunch of times, can we do this all more efficiently?
import numpy as np
import torch

# Resample 3d by calling resample1d thrice
# Decide the order based on the minimum amount of computation required vs accuracy
# If we want the most accuracy, we'll interpolate on the resampling axes
# over which we're inferring values between input points, whereas if we 
# want the most speed, we'll start by interpolating on the ones where
# we're downsampling
def resample3d(inp,inp_space,out_space=(1,1,1)):
    # Infer new shape
    out = resample1d(inp,inp_space[2],out_space[2]).permute(0,2,1)
    out = resample1d(out,inp_space[1],out_space[1]).permute(2,1,0)
    out = resample1d(out,inp_space[0],out_space[0]).permute(2,0,1)
    return out

def resample1d(inp,inp_space,out_space=1):
    #Output shape
    out_shape = list(np.int64(inp.size()[:-1]))+[int(np.floor(inp.size()[-1]*inp_space/out_space))] #Optional for if we expect a float_tensor
    out_shape = [int(item) for item in out_shape]
    # Get output coordinates, deltas, and t (chord distances)
    torch.cuda.set_device(inp.get_device())
    
    # Output coordinates in real space
    coords = torch.cuda.HalfTensor(range(out_shape[-1]))*out_space
    delta = coords.fmod(inp_space).div(inp_space).repeat(out_shape[0],out_shape[1],1)
    t = torch.cuda.HalfTensor(4,out_shape[0],out_shape[1],out_shape[2]).zero_()
    t[0] = 1
    t[1] = delta
    t[2] = delta**2
    t[3] = delta**3

    
    # Nearest neighbours indices
    nn = coords.div(inp_space).floor().long()    

    # Stack the nearest neighbors into P, the Points Array
    P = torch.cuda.HalfTensor(4,out_shape[0],out_shape[1],out_shape[2]).zero_()
    for i in range(-1,3):
        P[i+1] = inp.index_select(2,torch.clamp(nn+i,0,inp.size()[-1]-1))    
    
    #Take catmull-rom  spline interpolation:
    return 0.5*t.mul(torch.cuda.HalfTensor([[ 0,  2,  0,  0],
                            [-1,  0,  1,  0],
                            [ 2, -5,  4, -1],
                            [ -1, 3, -3,  1]]).mm(P.view(4,-1))\
                                                              .view(4,
                                                                    out_shape[0],
                                                                    out_shape[1],
                                                                    out_shape[2]))\
                                                              .sum(0)\
                                                              .squeeze()