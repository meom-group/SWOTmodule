# SWOTdenoise.py
"""
The SWOTdenoise module is a toolbox developed specifically in preparation of the SWOT mission. It provides a toolbox to remove small-scale noise from SWOT data. The main function is SWOTdenoise (same name as the module itself), and for standard applications, the user should not need to call other module functions. Optionally, other functions that can be directly useful are read_data (to read data from a netcdf file) and fill_nadir_gap: this function fills the lon and lat arrays in the SWOT nadir gap, and introduces fill values in the SSH array. For more details look at the dedicated helps.

# AUTHORS:
Laura Gomez Navarro (1,2), Emmanuel Cosme (1), Nicolas Papadakis (3), Le Sommer, J. (1), Pascual, A. (2), Poel, N. (1), Monsimer, A. (1)

(1) CNRS/UGA/IRD/G-INP, IGE, Grenoble, France
(2) IMEDEA (CSIC-UIB), Esporles, Spain
(3) CNRS/Univ. Bordeaux/B-INP, IMB, Bordeaux, France

# HISTORY:
- April 2018: version 1 (_orig)
- May 2018: version 2
- Update: 14/06/2018
- Last update: 19/12/2019
""" 

import numpy as np
from netCDF4 import Dataset
from scipy import ndimage as nd
from scipy.interpolate import RectBivariateSpline
from types import *
import sys
from configparser import ConfigParser

import os

def read_var_name(filename):
    """
    Read in the config file the names of the variables in the input netcdf file.
    
    Parameters:
    ----------
    filename: input config file
    
    Returns:
    -------
    list of variables names in order: SSH, lon, lat, xac, xal
    """
    config = ConfigParser()
    config.read(filename)
    ssh_name = config.get('VarName','ssh_name')
    lon_name = config.get('VarName','lon_name')
    lat_name = config.get('VarName','lat_name')
    xac_name = config.get('VarName','xac_name')
    xal_name = config.get('VarName','xal_name')
    listvar = ssh_name, lon_name, lat_name, xac_name, xal_name
    return listvar

def read_data(filename, *args):
    """
    Read arrays from netcdf file.
    
    Parameters:
    ----------
    filename: input file name
    *args: strings, variables to be read as named in the netcdf file.
    
    Returns:
    -------
    arrays. The number of output arrays must be identical to the number of variables.
    """
    
    fid = Dataset(filename)
    output = []
    for entry in args:
        output.append( fid.variables[entry][:] )
    fid.close()
    
    return tuple(output)

def write_data(filename, output_filename, ssh_d, lon_d, lat_d, x_ac_d, time_d, norm_d, method, param, iter_max, epsilon, iters_d):
    """
    Write SSH in output file.
    
    Parameters:
    ----------
    filename: input filename (directory + filename)
    output_filename: 
    - None by default: creates an ouput file in the same directory with the same filename + the extension _denoised.  
    - otherwise can specify outfiledirectory + outputfilename .nc ('/out/x.nc')
    ssh_d, lon_d, lat_d, x_ac_d, time_d, norm_d, method, param, iter_max, epsilon, iters_d: standard SWOT data arrays. See SWOTdenoise function.
    
    Returns:
    -------
    fileout: Output file directory + filename.
    """
    
    # Output filename
    if output_filename == 'None': #default
        rootname = filename.split('.nc')[0]
        fileout  = rootname + '_denoised.nc'
        
    else:
        fileout = output_filename
    
    # Read variables (not used before) in input file:
    x_al_r = read_data(filename, 'x_al')
    
    # Create output file:
    fid = Dataset(fileout, 'w', format='NETCDF4') 
    fid.description = "Filtered SWOT data"
    fid.creator_name = "SWOTdenoise module"  

    # Dimensions
    time = fid.createDimension('time', len(time_d))
    x_ac = fid.createDimension('x_ac', len(x_ac_d))
    iters = fid.createDimension('iters', iters_d) #%

    # Create variables
    lat = fid.createVariable('lat', 'f8', ('time','x_ac'))
    lat.long_name = "Latitude" 
    lat.units = "degrees_north"
    lat[:] = lat_d
  
    lon = fid.createVariable('lon', 'f8', ('time','x_ac'))
    lon.long_name = "longitude" 
    lon.units = "degrees_east"
    lon[:] = lon_d
       
    vtime = fid.createVariable('time', 'f8', ('time'))
    vtime.long_name = "time from beginning of simulation" 
    vtime.units = "days"
    vtime[:] = time_d

    x_al = fid.createVariable('x_al', 'f8', ('time'))
    x_al.long_name = "Along track distance from the beginning of the pass" 
    x_al.units = "km"
    x_al[:] = x_al_r

    vx_ac = fid.createVariable('x_ac', 'f8', ('x_ac'))
    vx_ac.long_name = "Across track distance from nadir" 
    vx_ac.units = "km"
    vx_ac[:] = x_ac_d

    ssh = fid.createVariable('SSH', 'f8', ('time','x_ac'), fill_value=ssh_d.fill_value)
    ssh.long_name = "SSH denoised" 
    ssh.units = "m"
    ssh[:] = ssh_d

    ssh.method   = method
    ssh.param    = str(param)
    ssh.iter_max = str(iter_max)
    ssh.epsilon  = str(epsilon)

    viters = fid.createVariable('iters', 'f8', ('iters'))
    viters.long_name = "Number of iterations done in filtering"
    viters[:] = np.arange(1, iters_d+1)

    norm = fid.createVariable('norm', 'f8', ('iters'))
    norm.long_name = "norm xxx"  
    norm.units = "m" 
    norm[:] = norm_d 

    fid.close()  # close the new file
    
    return fileout 
    
    
def copy_arrays(*args):
    """numpy-copy arrays.
    
    Parameters:
    ----------
    *args: arrays to copy.
    
    Returns:
    -------
    arrays. The number of output arrays must be identical to the number of inputs.
    """
    
    output = []
    for entry in args:
        #print entry
        output.append( entry.copy() )
    return tuple(output)


def fill_nadir_gap(ssh, lon, lat, x_ac, time, method = 'fill_value'): 

    """
    Fill the nadir gap in the middle of SWOT swath.
    Longitude and latitude are interpolated linearly. For SSH, there are two options:
    
    If the gap is already filled in the input arrays, it returns the input arrays.
    
    Parameters:
    ----------
    ssh, lon, lat, x_ac, time: input masked arrays from SWOT data. See SWOTdenoise function.
    method: method used to fill SSH array in the gap. Two options:
        - 'fill_value': the gap is filled with the fill value of SSH masked array;
        - 'interp': the gap is filled with a 2D, linear interpolation.
    
    Returns:
    -------
    ssh_f, lon_f, lat_f, x_ac_f: Filled SSH (masked), lon, lat 2D arrays, and across-track coordinates.
    """
    # Extend x_ac, positions of SWOT pixels across-track
    nhsw = len(x_ac)//2                                    # number of pixels in half a swath
    step = int(abs(x_ac[nhsw+1]-x_ac[nhsw]))                   # x_ac step, constant

    
    ins  = np.arange(x_ac[nhsw-1], x_ac[nhsw], step)[1:]  # sequence to be inserted
    nins = len(ins)                                       # length of inserted sequence
    
    if nins == 0: # if nadir gap already filled, return input arrays
        lon_f  = lon
        lat_f  = lat
        x_ac_f = x_ac           
        ssh_f  = ssh

    else:
        x_ac_f = np.insert(x_ac, nhsw, ins)                   # insertion

        # 2D arrays: lon, lat. Interpolation of regular grids.
        lon_f = RectBivariateSpline(time, x_ac, lon)(time, x_ac_f)
        lat_f = RectBivariateSpline(time, x_ac, lat)(time, x_ac_f)

        ###### Explanation of RectBivariateSpline function use:
        ## fx = RectBivariateSpline(time, x_ac, ssh)
        ## fx(time, x_ac_f)

        ## Chack if SSH is masked:
        if np.ma.isMaskedArray(ssh) == False:
            ssh = np.ma.asarray(ssh)
            print('ssh had to be masked1')

        # SSH: interpolate or insert array of fill values, and preserve masked array characteristics
        if method == 'interp':
            ssh_f = np.ma.masked_values( RectBivariateSpline(time, x_ac, ssh)(time, x_ac_f), ssh.fill_value )
        else:
            ins_ssh = np.full( ( nins, len(time) ), ssh.fill_value, dtype='float32' )
            ssh_f = np.ma.masked_values( np.insert( ssh, nhsw, ins_ssh, axis=1 ), ssh.fill_value )

    return ssh_f, lon_f, lat_f, x_ac_f

def empty_nadir_gap(ssh_f, x_ac_f, ssh, x_ac):
    """
    Remove entries of the nadir gap from ssh array.
    
    Parameters:
    ----------
    ssh_f: input 2D masked array of SSH data with filled gap
    x_ac_f: across-track coordinates of ssh_f
    ssh: 2D masked array of original SWOT SSH, with the gap
    x_ac: across-track coordinates of ssh
    
    Returns:
    -------
    2D masked array is of the same shape as the initial SWOT array.
    """
    
    ninter = len(x_ac_f) - len(x_ac)
    
    if ninter != 0: 
        nx = ( np.shape(ssh_f)[1] - ninter ) // 2
        #ssh_out = np.concatenate([ ssh_f.data[:,0:nx], ssh_f.data[:,-nx:] ], axis=1)
        ssh_out = np.concatenate([ ssh_f[:,0:nx], ssh_f[:,-nx:] ], axis=1)
        ssh_out = np.ma.array(ssh_out, mask = ssh.mask, fill_value = ssh.fill_value)
    else:
        ssh_out = ssh_f
    
    return ssh_out


def convolution_filter(ssh, param, method):
    """
    Filter an image with a convolution of a generic function (Gaussian or boxcar).
    The input image can contain gaps (masked values).
    Gaps are filled with 0. An array of 1 is created with gaps set to 0. Both are filtered and divided. Inspired from
    http://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    This function calls scipy.ndimage.
    
    Parameters:
    ----------
    ssh: 2D masked array to filter
    param: parameter for the method:
        - standard deviation for the Gaussian
        - box size for boxcar
    method: Gaussian or boxcar.
    
    Returns:
    -------
    2D ndarray (not a masked array).
    """
    assert np.ma.any(ssh.mask), 'u must be a masked array'
    mask = np.flatnonzero(ssh.mask)            # where u is masked
    v = ssh.data.copy()
    v.flat[mask] = 0                           # set masked values of data array to 0
    w = np.ones_like(ssh.data)
    w.flat[mask] = 0                           # same with the '1' array
    
    if method == 'boxcar':
        param = int(param)
        v[:] = nd.generic_filter(v ,function=np.nanmean, size = param)
        w[:] = nd.generic_filter(w, function=np.nanmean, size = param)
    elif method == 'gaussian':
        v[:] = nd.gaussian_filter(v ,sigma = param)
        w[:] = nd.gaussian_filter(w, sigma = param)
    elif method == 'do_nothing':
        pass
    else:
        write_error_and_exit(2)
    
    w = np.clip( w, 1.e-8, 1.)                 # to avoid division by 0. resulting values will be masked anyway.

    return v/w


def gradx(I): 
    """
    Calculates the gradient in the x-direction of an image I and gives as output M.
    In order to keep the size of the initial image the last row is left as 0s.
    """
    
    m, n = I.shape
    M = np.ma.zeros([m,n])

    M[0:-1,:] = np.ma.subtract(I[1::,:], I[0:-1,:])
    return M


def grady(I): 
    """
    Calculates the gradient in the y-direction of an image I and gives as output M.
    In order to keep the size of the initial image the last column is left as 0s.
    """
    
    m, n = I.shape
    M = np.ma.zeros([m,n])
    M[:,0:-1] =  np.ma.subtract(I[:,1::], I[:,0:-1])
    return M


def div(px, py): 
    """
    Calculates the divergence of a 2D field. 
    For the specific application of image denoising, the calculation follows Chambolle (REF)
    ## BELOW, TO BE CLARIFIED
    The x component of M (Mx) first row is = to the first row of px.
    The x component of M (Mx) last row is = to - the before last row of px. (last one = 0)
    The y component of M (My) first column is = to the first column of py.
    The y component of M (My) last column is = to - the before last column of py. (last one = 0)
    ??#(de sorte que div=-(grad)^*)
    Parameters: two 2D ndarray
    Returns: 2D ndarray
    """
    m, n = px.shape
    M = np.ma.zeros([m,n])
    Mx = np.ma.zeros([m,n])
    My = np.ma.zeros([m,n])
 
    Mx[1:m-1, :] = px[1:m-1, :] - px[0:m-2, :]
    Mx[0, :] = px[0, :]
    Mx[m-1, :] = -px[m-2, :]

    My[:, 1:n-1] = py[:, 1:n-1] - py[:, 0:n-2]
    My[:, 0] = py[:,0]
    My[:, n-1] = -py[:, n-2]
     
    M = Mx + My;
    return M


def laplacian(u):
    """
    Calculates laplacian using the divergence and gradient functions defined in the module.
    Parameter: 2D ndarray
    Returns: 2D ndarray
    """
    Ml = div(gradx(u), grady(u));
    return Ml

def iterations_var_reg_fista(ssh, ssh_d, param, epsilon=1.e-6, itermax=2000):
    """
        Perform iterations for solving the variational regularization using accelerated Gradient descent
        
        Parameters:
        ----------
        ssh: original image (masked array)
        ssh_d: working image (2D ndarray)
        param: parameters, weights of the cost function
        itermax: maximum number of iterations in the gradient descent method.
        epsilon: for convergence criterium.
        
        Returns:
        -------
        ssh_d: 2D ndarray containing denoised ssh data (ssh_d is not a masked array!)
        ##norm_array: Array of the norms calculated at each iteration to confirm convergence.
        """
    
    # Gradient descent
    param_orig = param
    param_max=max(param)
    scal_data=1.
    if param_max> 1.:
        scal_data=scal_data/param_max
        param = [x/param_max for x in param]
    
    tau =   1./(scal_data+8*param[0]+64*param[1]+512*param[2])  # Fix the tau factor for iterations
    
    #print tau
    mask       = 1 - ssh.mask                    # set 0 on masked values, 1 otherwise. For the background term of cost function.
    iteration  = 0
    norm_array = [] #%
    cost       = np.ndarray((itermax,),float)
    ssh_y      = np.copy(ssh_d)
    t          = 1.
    
    while (iteration < itermax):

        lap_y    = laplacian(ssh_y)
        bilap_y  = laplacian(lap_y)
        trilap_y = laplacian(bilap_y)

        #FISTA acceleration
        incr = mask*(ssh.data-ssh_y)*scal_data + param[0]*lap_y - param[1]*bilap_y+param[2]*trilap_y
        ssh_tmp = ssh_y + tau *incr
     
        t0 = t;
        t = (1 + np.sqrt(1 + 4*np.power(t0,2))) / 2
        ssh_y = ssh_tmp + (t0 - 1.) / t*(ssh_tmp-ssh_d)
        
        norm = np.ma.max(np.abs(ssh_tmp-ssh_d))
        ssh_d = np.copy(ssh_tmp)
        # Can be removed:
        cost[iteration] = cost_function(mask, ssh.data, ssh_d, param_orig)
        iteration += 1
        norm_array.append(norm)
        
        if norm < epsilon:
            break
    print('Iteration reached: ' + str(iteration))
    print('norm/epsilon = ' + str(np.round(norm/epsilon,2)))

    norm_array = np.array(norm_array) #%
    
    return ssh_d, norm_array, iteration, cost
# iteration -1, as the iteration at which it stops it does not filter

def variational_regularization_filter_fista(ssh, param, itermax, epsilon, pc_method='gaussian', pc_param=10.):
    """
        Apply variational regularization filter. \n
        
        Parameters:
        ----------
        ssh: masked array with nadir gap filled.
        param: 2-entry tuple for first and second, terms of the cost function, respectively.
        itermax: maximum number of iterations in the gradient descent method.
        epsilon: for convergence criterium.
        pc_method: convolution method for preconditioning.
        pc_param: parameter for pre-conditioning method.
        
        Returns:
        -------
        ssh_d: 2D ndarray containing denoised ssh data (ssh_d is not a masked array!)
        """
    
    # Apply the Gaussian filter for preconditioning
    if any(param) is not 0.:
        ssh_d = convolution_filter(ssh, pc_param, method = pc_method)  # output here is a simple ndarray


    ssh_d, norm, iters, cost  = iterations_var_reg_fista(ssh, ssh_d, param, epsilon, itermax=itermax)
            # always gives back norm and iters but the one finally saved in the netcdf is the final one (the one we want)
            
            
    return ssh_d, norm, iters, cost

def write_error_and_exit(nb):
    """Function called in case of error, to guide the user towards appropriate adjustment."""
    
    if nb == 1:
        print("You must provide a SWOT filename and output filename, or SSH, lon, lat, x_ac and time arrays. SSH must be a masked array.")
    if nb == 2:
        print("The filtering method is not correctly set.")
    if nb == 3:
        print("For the variational regularization filter, lambd must be a 3-entry tuple.")
    if nb == 4:
        print("For convolutional filters, lambd must be a number.")
    if nb == 5:
        print("For the fista variational regularization filter, lambd must be a 2-entry tuple.")
    sys.exit()

def cost_function(mask,hobs, h, param):
    """
    Function to obtain the cost-function calculated. (not used within the module, but useful to have it as related with the output of the module.
    hobs = ssh_obs
    h = filtered image
    params = lambdas
    """
    if np.ma.isMaskedArray(hobs) == False:
        hobs = np.ma.asarray(hobs)
       
    #if np.ma.isMaskedArray(h) == False:
    #h = np.ma.array(h, mask = hobs.mask, fill_value = 1e9 )
    # above to check to improve like with an assert or type
    
    # h_derivs, _, _, _ = fill_nadir_gap(h, lon, lat, x_ac, time, method='fill_value')
    # -->returns masked array, with the gap included, but masked!
    
    gradx_h = gradx(h)
    grady_h = grady(h)
    # gradx_h = np.ma.array(gradx_h, mask = h_derivs.mask, fill_value = 1e9 )
    # grady_h = np.ma.array(grady_h, mask = h_derivs.mask, fill_value = 1e9 )
    grad_h  = gradx_h**2 + grady_h**2
    
    lap_h = laplacian(h)
    #lap_h = np.ma.array(lap_h, mask = h_derivs.mask, fill_value = 1e9 )

    gradxlap_h = gradx(lap_h)
    gradylap_h = grady(lap_h)
    # gradxlap_h = np.ma.array(gradxlap_h, mask = h_derivs.mask, fill_value = 1e9 )
    # gradylap_h = np.ma.array(gradylap_h, mask = h_derivs.mask, fill_value = 1e9 )
    gradlap_h =  gradxlap_h**2 + gradylap_h**2
    
    c_func = 0.5 * ( np.ma.sum(mask*(h - hobs)**2) + (param[0]*np.ma.sum(grad_h)) + (param[1]*np.ma.sum(lap_h**2)) + (param[2]*np.ma.sum(gradlap_h)) )
    
    #print('1st term: ', str(np.nansum((h - hobs)**2)))
    #print('2nd term: ', str(param[0]*np.nansum(grad_h)))
    #print('3rd term: ', str(param[1]))
    #print('4th term: ', str(param[2]))
          
    return c_func


################################################################
# Main function:

def SWOTdenoise(*args, **kwargs):
    # , method, parameter, inpainting='no',
    """
    Perform denoising of SWOT data.
    
    Parameters:
    ----------
    *args: name of file containing the SWOT SSH field to denoise (optional). Example of use:
        SWOTdenoise(filename, output_filename)  ##
        denoise data in file 'filename' and write an output file in the same directory. \n
        The output file is named 'foo_denoised.nc' if the input file name is 'foo.nc'.
        
    **kwargs include:
    - ssh : input ssh array (2D) in x_al(time), x_ac format (i.e., (lat, lon))
    - lon : input longitude array (2D)
    - lat : input latitude array (2D)
    - x_ac : input across-track coordinates (1D)
    - time : input along-track coordinates (1D)
    The above-mentioned arguments are mandatory if no file name is given. They are exactly in the format provided by the SWOT simulator for ocean science version 3.0.
    
    Other keywords arguments are:
    - config: name of the config file (default: SWOTdenoise.cfg)
    - method: gaussian, boxcar, or var_reg_fista (default);
    - param: number for gaussian and boxcar; 3-entry tuple for var_reg_fista (default: (1.5, 0, 0); under investigation) ;
    - inpainting: if True, the nadir gap is inpainted. If False, it is not and the returned SSH array is of the same shape as the original one. If the SWOTdenoise function is called using arrays (see above description) with inpainting=True, then it returns SSH, lon, and lat arrays. If it is called using arrays with inpainting=False, it returns only SSH, since lon and lat arrays are the same as for the input field. Default is False.
    - itermax: only for var_reg_fista: maximum number of iterations in the gradient descent algortihm (default: 2000);
    - epsilon: only for var_reg_fista: convergence criterion for the gradient descent algortihm (default: 1e-6);
    - pc_method: only for var_reg_fista: convolution method for pre-conditioning (default: gaussian);
    - pc_param: only for var_reg_fista: parameter for pre-conditioning method (default: 10);
   
    The algorithms are detailed in the scientific documentation.

    """
    
    ################################################################
    # 1. Read function arguments
    
    # 1.1. Input data
    
    file_input = len(args) == 2 ##
    
    if file_input:
        if type(args[0]) is not str: write_error_and_exit(1) 
        
        filename = args[0]
        output_filename = args[1] ##
        swotfile = filename.split('/')[-1]
        swotdir = filename.split(swotfile)[0]
        #listvar = 'SSH_obs', 'lon', 'lat', 'x_ac', 'time'
        configfilename = kwargs.get('config', 'SWOTdenoise.cfg')
        listvar = read_var_name(filename = configfilename)
        ssh, lon, lat, x_ac, time = read_data(filename, *listvar)
    
    elif len(args) == 1: 
        write_error_and_exit(1) 
        
    else:
        ssh         = kwargs.get('ssh', None)
        lon         = kwargs.get('lon', None)
        lat         = kwargs.get('lat', None)
        x_ac        = kwargs.get('x_ac', None)
        time        = kwargs.get('time', None)
        #if any( ( isinstance(ssh,'NoneType'), isinstance(lon, NoneType), isinstance(lat, NoneType), \
        #          isinstance(x_ac, NoneType), isinstance(time, NoneType) ) ): write_error_and_exit(1)
           
    # 1.2. Denoising method and options
           
    method = kwargs.get('method', 'var_reg_fista')
    param = kwargs.get('param', (0., 10., 0.) )          # default value to be defined, previously was: (1.5, 10., 10.) --> very long and not the optimal
    inpainting = kwargs.get('inpainting', False)
    
    ## For variational regularization only
    itermax   = kwargs.get('itermax', 2000)              
    epsilon   = kwargs.get('epsilon', 1.e-6)
    pc_method = kwargs.get('pc_method', 'gaussian')
    pc_param  = kwargs.get('pc_param', 10.)
    cost      = np.ndarray((itermax,), float)

    # 2. Perform denoising
    
    # 2.1. Fill nadir gap with masked fill values
    
    ## Check if SSH is masked:
    if np.ma.isMaskedArray(ssh) == False:
        ssh = np.ma.asarray(ssh)
        print('ssh had to be masked2')
        
    ssh_f, lon_f, lat_f, x_ac_f = fill_nadir_gap(ssh, lon, lat, x_ac, time)  # fill the nadir gap with masked fill values
        
    # 2.2. Call method
    print('Method: ' + method)
    
    if method == 'do_nothing':
        ssh_d = convolution_filter(ssh_f, param, method='do_nothing')
        norm  = np.nan
        iters = 0

    if method == 'boxcar':
        if isinstance(param, int) or isinstance(param, float):
            ssh_d = convolution_filter(ssh_f, param, method='boxcar')
            norm  = np.nan
            iters = 0
            itermax = 'none'
            epsilon = 'none'
        else:
            write_error_and_exit(4)
       
    if method == 'gaussian':
        if isinstance(param, int) or isinstance(param, float):
            ssh_d = convolution_filter(ssh_f, param, method='gaussian')
            norm  = np.nan
            iters = 0
            itermax = 'none'
            epsilon = 'none'
        else:
            write_error_and_exit(4)
            
    if method == 'var_reg_fista':
        #Fista acceleration
        if isinstance(param, tuple) and len(param) == 3:
            ssh_d, norm, iters, cost = variational_regularization_filter_fista(ssh_f, param, \
                                                               itermax=itermax, epsilon=epsilon, pc_method=pc_method, \
                                                               pc_param=pc_param)
                                                               
        else:
            write_error_and_exit(3)
        
    # 2.3. Handle inpainting option, and recover masked array

    if inpainting is True:
        ssh_tmp, _, _, _ = fill_nadir_gap(ssh, lon, lat, x_ac, time, method='interp')     # to get appropriate mask
        ssh_d = np.ma.array(ssh_d, mask = ssh_tmp.mask, fill_value = ssh.fill_value )     # generate masked array
        lon_d, lat_d, x_ac_d = copy_arrays(lon_f, lat_f, x_ac_f)
    else:
        ssh_d = np.ma.array(ssh_d, mask = ssh_f.mask, fill_value = ssh.fill_value )       # generate masked array
        ssh_d = empty_nadir_gap(ssh_d, x_ac_f, ssh, x_ac)                                 # Remove value in the gap
        lon_d, lat_d, x_ac_d = copy_arrays(lon, lat, x_ac)
        
    # Set masked values to fill value
    
    if np.ma.is_masked(ssh_d):# check if mask is not = False, because if so by default it selects the first row of the array and applies to it the fill_value
        #print('ssh_d is masked')
        mask = ssh_d.mask
        ssh_d.data[mask] = ssh_d.fill_value
        
    # 3. Manage results
    cost = cost[1:iters] # first value?
    
    if file_input:
        fileout = write_data(filename, output_filename, ssh_d, lon_d, lat_d, x_ac_d, time, norm, method, param, itermax, epsilon, iters) ## , cost
        print('Filtered field in ', fileout )
    else:
        if inpainting is True:
            return ssh_d, lon_d, lat_d, cost
        else:
            return ssh_d, cost
   
