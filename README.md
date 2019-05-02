## Graph Convolutional Neural Networks with Data-driven Graph Filter (GCNN-DDGF)

This repository includes the GCNN-DDGF work for the following problem:

* Network-wide Station-level Bike-Sharing Demand Prediction

### Bike-Sharing Demand Prediction 

The Bike-sharing demand dataset includes over 28 million bike-sharing transactions between 07/01/2013 and 06/30/2016, which are downloaded from [Citi BSS in New York City](https://www.citibikenyc.com/system-data). The data is processed as follows: 

* For each station, 26304 hourly bike demands are aggregrated based on the bike check-out time and start station in trasaction records;

* New stations were being set up from 2013 to 2016. Only stations existing in all three years are included;

* Stations with total three-year demand of less than 26304 (less than one bike per hour) are excluded. 

After preprocessing, 272 stations are considered in this study. The 272 by 26304 matrix is saved as ["data\NYCBikeHourly272.pickle"](https://github.com/transpaper/GCNN/tree/master/data). The Lat/Lon coordinates of 272 stations are saved in ["citi_bike_station_locations.csv"](https://github.com/transpaper/GCNN/tree/master/data).

### On-going Experiments for Network-wide Traffic Speed Prediction

We use the traffic data for Los Angeles (METR-LA) in the paper "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting", which can be found [here](https://github.com/liyaguang/DCRNN). 

### Citation
You are more than welcome to cite our paper:
```
@article{lin2018predicting,
  title={Predicting station-level hourly demand in a large-scale bike-sharing network: A graph convolutional neural network approach},
  author={Lin, Lei and He, Zhengbing and Peeta, Srinivas},
  journal={Transportation Research Part C: Emerging Technologies},
  volume={97},
  pages={258--276},
  year={2018},
  publisher={Elsevier}
}

```
