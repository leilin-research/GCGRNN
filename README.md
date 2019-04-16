## Graph Convolutional Neural Networks with Data-driven Graph Filter (GCNN-DDGF)

This repository applies the GCNN-DDGF for two traffic-related data prediction studies:

* Network-wide Station-level Bike-Sharing Demand Prediction

* Network-wide Traffic Speed Prediction

### Bike-Sharing Demand Prediction 

The Bike-sharing demand dataset includes over 28 million bike-sharing transactions between 07/01/2013 and 06/30/2016, which are downloaded from [Citi BSS in New York City](https://www.citibikenyc.com/system-data). The data is processed as follows: 

* For each station, transactions are aggregrated to generate hourly demand;

* New stations were being set up from 2013 to 2016. Only stations existing in all three years are included;

* Stations with total three-year demand of less than 26304 (less than one bike per hour) are excluded. 

272 stations are considered in this study. 

### Network-wide Traffic Speed Prediction

We use the traffic data for Los Angeles (METR-LA) in the paper "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting", which can be found in [their repository](https://github.com/liyaguang/DCRNN). 
