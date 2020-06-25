[![Documentation Status](https://readthedocs.org/projects/swot-module/badge/?version=latest)](https://swot-module.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/meom-group/SWOTmodule.svg?branch=master)](https://travis-ci.org/meom-group/SWOTmodule)
# SWOTmodule

## Documentation

The complete documentation of this module can be found : [here](https://swot-module.readthedocs.io)

## Description

This module is a collection of tools used to read data from the SWOT simulator, filter it and save it in an output file.

* Codes:

   * SWOTdenoise.cfg: Configuration file called by SWOTdenoise.py.  Used to specify the name of the variables of the input file to be filtered.

   * SWOTdenoise.py: Module to read SWOT data, filter it and save it in a new output file or obtain the SSH de-noised variable.

* Example data: The two examples below are SWOT simulated passes from the NATL60 model, generated for the fast-sampling phase in the western Mediterranean Sea.
  
   * MED_fastPhase_1km_swotFAST_c01_p009.nc: example SWOT dataset directly out of SWOT simulator (version 2.21)
   
   * MED_1km_nogap_JAS12_swotFastPhase_BOX_c01_p009_v2.nc: example SWOT dataset subregion (box_dataset) used in paper Gomez-Navarro et al. (in review).

* Example notebooks:

   * discover-SWOTmodule.ipynb : Example of using SWOTdenoise module with the SWOT simulator output netcdfs

   * discover-SWOTmodule_box_dataset.ipynb : Example of using SWOTdenoise module with modified SWOT simulator output netcdfs.  In this case the dataset used in the study Gomez-Navarro et al. (in review). 


## How to install the module :

- Clone this repository :

```git clone https://github.com/meom-group/SWOTmodule.git```

- Create and activate the swotmod conda environment :

```
conda env create -f env_swotmod.yml
conda activate swotmod
```
   
- Add the swotmod kernel to jupyter :
 
```
python -m ipykernel install --user --name swotmod --display swotmod
```
## How to use the module :

- Launch one the demonstration notebook
- Modify the SWOTdenoise.cfg for your own use

## Bibliography
[Gómez-Navarro, L., Cosme, E., Le Sommer, J., Papadakis, N., and Pascual, A. (2020). Development of
an image de-noising method in preparation for the surface water and ocean topography satellite mission.
Remote Sensing , 12(4).](https://www.researchgate.net/publication/339456605_Development_of_an_Image_De-Noising_Method_in_Preparation_for_the_Surface_Water_and_Ocean_Topography_Satellite_Mission)





