# Virtual Fleet simulations for the Gulf Stream Extension

This repository host work and results from analysis of Virtual Fleet simulations dedicated to improve observation of the Gulf Stream Extension with Argo floats. Our goal is to improve Argo sampling of the GSE. 


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
