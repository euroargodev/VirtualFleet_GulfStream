# Virtual Fleet simulations for the Gulf Stream Extension

This repository host work and results from analysis of Virtual Fleet simulations dedicated to improve observation of the Gulf Stream Extension with Argo floats. Our goal is to improve Argo sampling of the GSE. 


# Recap and conclusion 

For the Gulf Stream region it was decided to explore the impact on the long term Argo array sampling if one chooses to temporarily modify float configuration parameters (with Iridium command) when they enter the study area. This was motivated by the fact that it would be quite difficult to significantly alter the North Atlantic deployment plan to accommodate for a better GSE sampling, and that using 2-ways communications to update the fleet configuration temporarily may be more acceptable to the network operators.

We simulated 10 years (2008-2018) of a realistic Argo fleet using (i) the historical deployment plan and (ii) an eddy resolving state of the art ocean re-analysis (assimilating sea level, SST, in-situ T/S profiles and Sea Ice concentration and/or thickness).

A control simulation was performed without modifying Argo float parameters and using typical values for cycling frequency (10 days) and drifting depth (1000db). All floats profiling depth was set to 2000db. The control simulation was evaluated correct (see figures below) given the simulation limitations (eg: we used a similar life expectancy for all simulated floats, i.e. 159 cycles that was determined as an optimum to reproduce the same 175.000 amount of total profiles).

A series of experiments was then performed, where Argo float parameters are modified when they enter the study area (GSE box) and restored when they exit the area.
We finally compared the 10 years simulation difference in profile density computed on a 1x1 degree grid.

Results are shown here for experiments where the cycling frequency was increased to 5 days and drifting depths changed to: 500, 1000, and 1500 db:

![](https://raw.githubusercontent.com/euroargodev/VirtualFleet_GulfStream/main/img/synthesis.png)

We see that if floats drift at 500db, they are taken by the GS out of the box, downstream/eastward, too fast. The result is that the upstream region is now less sampled, to the benefit of the eastern part of the box and outside of it. This is not the expected outcome.
On the other hand, if floats drift at 1500db, they are taken by the southward flowing under current. The result is a better sampling of the GS along the U.S. east coast but a rather in-homogeneous increase over the GSE box.
Keeping the drifting depth to 1000db seems the best solution in the case where the cycling frequency is increased to 5 days. This set-up leads to an homogeneous increase of the profiles density in the GSE box and a smaller impact on the downstream/eastward sampling decrease.

**Recommendations**

We found that 2-ways communication “online” changes of the cycling frequency to 5 days leads to a 40/50% increase in profile density in the high EKE region of the Gulf Stream, using a drifting depth of 1000db.

If followed, this local change of Argo float mission parameters would have a “reasonable cost” of a smaller than 25% decrease in profile density up and downstream of the Gulf Stream Extension region, where sampling would remain above Argo nominal target (1 profile every 10 days on a 3x3 grid).

Therefore we recommend the following:
- Experiment in real world on a few floats the automatic change of mission parameters over the Gulf Stream area in order to assess the technical feasibility of the procedure, 
- Conduct an OSSE to assess the impact of local sampling changes to Ocean Climate Indicators such as heat content.


# Experiments

## Results

All figures available on the [RESULTS page](RESULTS.md) !

We first need to reproduce the observed sampling to validate the methodology. Below is our control run 2008-2018 Argo sampling of the North Atlantic, and it looks quite good ! [Check the full control run validation thread here](https://github.com/euroargodev/boundary_currents/discussions/11).

![](https://raw.githubusercontent.com/euroargodev/VirtualFleet_GulfStream/main/img/Simulation-Profile-Density-N159-Control-(%23613257).png)


## Simulations
This repo is using the Virtual Fleet software from the ["gulf-stream" branch](https://github.com/euroargodev/VirtualFleet/tree/gulf-stream).

The Argo deployment plan used is the one implemented in reality over 2008-2018. The data/latitude/longitude positions/wmo, and more, of [all Argo floats deployed is available in the json file](https://raw.githubusercontent.com/euroargodev/VirtualFleet_GulfStream/main/data/2008-2018-Deployment-Plan.json).  

The list of all Argo floats configuration json files are found [in this folder](https://github.com/euroargodev/VirtualFleet_GulfStream/tree/main/data), with the 'vf-floatcfg' preffix. These can be loaded with:
```python
from virtualargofleet import FloatConfiguration
cfg = FloatConfiguration('vf-floatcfg-gse-experiment-N300-5days.json')
```

All simulations we performed are documented in [this json file](https://raw.githubusercontent.com/euroargodev/VirtualFleet_GulfStream/main/data/simulations_db.json).
We created a command line program to explore this simulation's database, it's named ``simdb`` and is available under the ``cli`` folder. If you cloned this repo, simply add this folder to our path:
```bash
export PATH="~/git/github/euroargodev/VirtualFleet_GulfStream/cli:$PATH"
```
and then in a command line:
```bash
simdb --help
```


***
This work is developed by:
<div>
<img src="https://www.umr-lops.fr/var/storage/images/_aliases/logo_main/medias-ifremer/medias-lops/logos/logo-lops-2/1459683-4-fre-FR/Logo-LOPS-2.png" height="75">
<a href="https://wwz.ifremer.fr"><img src="https://user-images.githubusercontent.com/59824937/146353099-bcd2bd4e-d310-4807-aee2-9cf24075f0c3.jpg" height="75"></a>
<img src="https://github.com/euroargodev/euroargodev.github.io/raw/master/img/logo/ArgoFrance-logo_banner-color.png" height="75">
</div>
and part of the Euro-ArgoRISE project that has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement no 824131. Call INFRADEV-03-2018-2019: Individual support to ESFRI and other world-class research infrastructures.
<div>
<a href="https://www.euro-argo.eu/EU-Projects/Euro-Argo-RISE-2019-2022">
<img src="https://user-images.githubusercontent.com/59824937/146353317-56b3e70e-aed9-40e0-9212-3393d2e0ddd9.png" height="100">
</a>
</div>
