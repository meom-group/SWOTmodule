# SWOTmodule

New versions of this module can be now found at: xx

Last updated on: xx

* Example notebooks:

  * 2018-03-03-ec-discover-SWOTmodule.ipynb

  * 2018-04-18-lgn-discover-SWOTmodule.ipynb : Example of using SWOYdenoise module with the SWOT simualtor output netcdfs

  * 2018-04-18-lgn-discover-SWOTmodule_box_dataset.ipynb : Example of using SWOYdenoise module with modified SWOT simualtor output netcdfs.  In this case the dataset used in the study Gomez-Navarro et al. (in review). 

* Codes:

  * SWOTdenoise.cfg: Configuration file called by SWOTdenoise.py.  Used to specify the name of the variables of the input file to be filtered.

  * SWOTdenoise.py: Module to read SWOT data, filter it and save it in a new output file or obtain the SSH de-noised variable.

* Example data:
The two examples below are SWOT simulated passes from the NAtl60 model, generated for the fast-sampling phase in the western Mediterranean Sea.
  
   * MED_fastPhase_1km_swotFAST_c01_p009.nc: example SWOT dataset directly out of SWOT simulator (version 2.21)
   
   * MED_1km_nogap_JAS12_swotFastPhase_BOX_c01_p009_v2.nc: example SWOT dataset subregion (box_dataset) used in paper Gomez-Navarro et al. (in review).
