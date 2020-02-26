
Module SWOTmodule_meom.SWOTdenoise
==================================

The SWOTdenoise module is a toolbox developed specifically in preparation of the SWOT mission. It provides a toolbox to remove small-scale noise from SWOT data. The main function is SWOTdenoise (same name as the module itself), and for standard applications, the user should not need to call other module functions. Optionally, other functions that can be directly useful are read_data (to read data from a netcdf file) and fill_nadir_gap: this function fills the lon and lat arrays in the SWOT nadir gap, and introduces fill values in the SSH array. For more details look at the dedicated helps.

AUTHORS:
========

Laura Gomez Navarro (1,2), Emmanuel Cosme (1), Nicolas Papadakis (3), Le Sommer, J. (1), Pascual, A. (2), Poel, N. (1), Monsimer, A. (1)

(1) CNRS/UGA/IRD/G-INP, IGE, Grenoble, France
(2) IMEDEA (CSIC-UIB), Esporles, Spain
(3) CNRS/Univ. Bordeaux/B-INP, IMB, Bordeaux, France

HISTORY:
========


* April 2018: version 1 (_orig)
* May 2018: version 2
* Update: 14/06/2018
* Last update: 19/12/2019

Functions
---------

``SWOTdenoise(*args, **kwargs)``
:   Perform denoising of SWOT data.

.. code-block::

   Parameters:
   ----------
   *args: name of file containing the SWOT SSH field to denoise (optional). Example of use:
       SWOTdenoise(filename, output_filename)  ##
       denoise data in file 'filename' and write an output file in the same directory. 

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



``convolution_filter(ssh, param, method)``
:   Filter an image with a convolution of a generic function (Gaussian or boxcar).
    The input image can contain gaps (masked values).
    Gaps are filled with 0. An array of 1 is created with gaps set to 0. Both are filtered and divided. Inspired from
    http://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    This function calls scipy.ndimage.

.. code-block::

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



``copy_arrays(*args)``
:   numpy-copy arrays.

.. code-block::

   Parameters:
   ----------
   *args: arrays to copy.

   Returns:
   -------
   arrays. The number of output arrays must be identical to the number of inputs.



``cost_function(mask, hobs, h, param)``
:   Function to obtain the cost-function calculated. (not used within the module, but useful to have it as related with the output of the module.
    hobs = ssh_obs
    h = filtered image
    params = lambdas

``div(px, py)``
:   Calculates the divergence of a 2D field. 
    For the specific application of image denoising, the calculation follows Chambolle (REF)

.. code-block::

   ## BELOW, TO BE CLARIFIED
   The x component of M (Mx) first row is = to the first row of px.
   The x component of M (Mx) last row is = to - the before last row of px. (last one = 0)
   The y component of M (My) first column is = to the first column of py.
   The y component of M (My) last column is = to - the before last column of py. (last one = 0)
   ??#(de sorte que div=-(grad)^*)
   Parameters: two 2D ndarray
   Returns: 2D ndarray



``empty_nadir_gap(ssh_f, x_ac_f, ssh, x_ac)``
:   Remove entries of the nadir gap from ssh array.

.. code-block::

   Parameters:
   ----------
   ssh_f: input 2D masked array of SSH data with filled gap
   x_ac_f: across-track coordinates of ssh_f
   ssh: 2D masked array of original SWOT SSH, with the gap
   x_ac: across-track coordinates of ssh

   Returns:
   -------
   2D masked array is of the same shape as the initial SWOT array.



``fill_nadir_gap(ssh, lon, lat, x_ac, time, method='fill_value')``
:   Fill the nadir gap in the middle of SWOT swath.
    Longitude and latitude are interpolated linearly. For SSH, there are two options:

.. code-block::

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



``gradx(I)``
:   Calculates the gradient in the x-direction of an image I and gives as output M.
    In order to keep the size of the initial image the last row is left as 0s.

``grady(I)``
:   Calculates the gradient in the y-direction of an image I and gives as output M.
    In order to keep the size of the initial image the last column is left as 0s.

``iterations_var_reg_fista(ssh, ssh_d, param, epsilon=1e-06, itermax=2000)``
:   Perform iterations for solving the variational regularization using accelerated Gradient descent

.. code-block::

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



``laplacian(u)``
:   Calculates laplacian using the divergence and gradient functions defined in the module.
    Parameter: 2D ndarray
    Returns: 2D ndarray

``read_data(filename, *args)``
:   Read arrays from netcdf file.

.. code-block::

   Parameters:
   ----------
   filename: input file name
   *args: strings, variables to be read as named in the netcdf file.

   Returns:
   -------
   arrays. The number of output arrays must be identical to the number of variables.



``read_var_name(filename)``
:   Read in the config file the names of the variables in the input netcdf file.

.. code-block::

   Parameters:
   ----------
   filename: input config file

   Returns:
   -------
   list of variables names in order: SSH, lon, lat, xac, xal



``variational_regularization_filter_fista(ssh, param, itermax, epsilon, pc_method='gaussian', pc_param=10.0)``
:   Apply variational regularization filter. 

.. code-block::

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



``write_data(filename, output_filename, ssh_d, lon_d, lat_d, x_ac_d, time_d, norm_d, method, param, iter_max, epsilon, iters_d)``
:   Write SSH in output file.

.. code-block::

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



``write_error_and_exit(nb)``
:   Function called in case of error, to guide the user towards appropriate adjustment.
