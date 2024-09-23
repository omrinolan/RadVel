# Radial Velocity fitting on Polluted White Dwarf stars

## Table of Contents
- [Description](#This is a )
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)

## Description
This is a project that I undetook as my Masters Research Project at the University of Cambridge in 2024. It's purpose was to investigate methods to precisely determine the radial velocity of polluted white dwarf stars.
The dataset included data from 6 different host stars, GD133, GD29-38, GD56-04, HE0106-3253, WD1457-086, and WD1929+011.
The data is taken from two different telescopes with high resolution spectroscopes, the SALT HRS (referred to as SALT), and the Magellan MIKE (referred to as MIKE).
The project involves importing and processing the data, and then running four different approaches in order to precisely determine the radial velocity shift of each individual timestamp measurement. 
The code then creates a timeseries of the individual measurements of the radial velocity and analyses this data with a periodogram to see if there is any periodic variability.
If there is no periodic variability, the spread of the data is taken to be the precision of the measurement.
This project is intended to make use of the narrow pollutant lines present in the spectra of polluted white dwarfs, and to test if they can be used for precise radial velocity measurements. White dwarfs have historically been ruled out for radial velocity measurements due to smooth featureless spectra, but this project shows that this method can detect hot jupiter planets orbiting on close-in orbits around host stars.

The four different approaches are as follows:
1. Direct Voigt fitting
2. Self Cross-correlation
3. Model Cross-correlation
4. Direct fitting with Bayesian Statistics

1,2, and 3 are contained in the file Salt_Data_readin_Radial_velocity.py
4 is contained in the file bayesian_statistics_trial.py
I have done some introductory analysis with simulated radial velocity curves for different planetary systems in the files RadialVelocity_intialanalysys.py and RadialVelocity_initialanalysis_testcases.py
I have used the precision of the RV measurements determined in these experiments to plot a parameter space of detection for each of the different instruments in the file Param_space.py
I have also run this exact same analysis for the data from G29-38, only using method 1, which is shown in the file G29-38_rv.py

There are also numerous plotting files that I have created, in rv_plotting.py and stacked_plotting.py



## Installation
Steps required to install the project and its dependencies.

data is found on the Cambridge IoA cluster at /data/wdplanetary/laura/
requirements.txt file has listed modules and dependencies

Virtual environment used is omrienv:
PATH: /data/wdplanetary/omri/envs/omrienv


# Example command for installing the project
pip install -r requirements.txt
source /data/wdplanetary/omri/envs/omrienv/bin/activate