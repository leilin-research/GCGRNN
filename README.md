## Graph Convolutional Neural Networks with Data-driven Graph Filter (GCNN-DDGF)

This repository includes the GCNN-DDGF work for the following challenge:

* Network-wide Station-level Bike-Sharing Demand Prediction

### Bike-Sharing Demand Prediction 

The Bike-sharing demand dataset includes over 28 million bike-sharing transactions between 07/01/2013 and 06/30/2016, which are downloaded from [Citi BSS in New York City](https://www.citibikenyc.com/system-data). The data is processed as follows: 

* For each station, 26304 hourly bike demands are aggregrated based on the bike check-out time and start station in trasaction records;

* New stations were being set up from 2013 to 2016. Only stations existing in all three years are included;

* Stations with total three-year demand of less than 26304 (less than one bike per hour) are excluded. 

After preprocessing, 272 stations are considered in this study. The 272 by 26304 matrix is saved as [NYCBikeHourly272.pickle](https://github.com/transpaper/GCNN/tree/master/data). The Lat/Lon coordinates of 272 stations are saved in [citi_bike_station_locations.csv](https://github.com/transpaper/GCNN/tree/master/data).

### On-going Experiments for Network-wide Traffic Speed Prediction

We are using the traffic speed data from Los Angeles ([metr-la.h5](https://github.com/transpaper/GCNN/tree/master/data)) provided in the following paper:

* Yaguang Li, Rose Yu, Cyrus Shahabi, Yan Liu, ["Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting"](https://github.com/liyaguang/DCRNN), ICLR 2018. 

The current best performance is **3.53** (Mean Absolute Error) for a 12-step prediction, which is comparable to **3.60** in the DCRNN paper. Right now we are improving the model stability. 

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
