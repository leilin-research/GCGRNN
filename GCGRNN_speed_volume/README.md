# GCGRNN: GCNN-DDGF with recurrent architecture
The model is implemented on the basis of [DCRNN](https://github.com/liyaguang/DCRNN), we replace the diffusion convolution with DDGF convolution. This is the main reason that our model is much faster than DCRNN. 

## GCGRNN Model Training
```bash
# METR-LA-Speed
python gcgrnn_train.py --config_filename=model_config/GCGRNN_la_speed.yaml

# PEMS-Volume (1 hour)
python gcgrnn_train.py --config_filename=model_config/GCGRNN_DDGF_volume.yaml

# PEMS-Volume (15 mins)
python gcgrnn_train.py --config_filename=model_config/GCGRNN_volume_15mins.yaml

```
## DCRNN Model Training
For METR-LA-Speed, we just use the same hyperparameter file ([model_config/dcrnn_la.yaml](https://github.com/transpaper/GCNN/tree/master/GCNN-DDGF_speed_volume/model_config)) provided in DCRNN. 
For PEMS-Volume, we tried different configurations, and the best one ([model_config/DCRNN_volume.yaml](https://github.com/transpaper/GCNN/tree/master/GCNN-DDGF_speed_volume/data/model_config)) is also provided here for comparison. 

Note that because we change the DCRNN codes, the interested users need to download the original [DCRNN models](https://github.com/liyaguang/DCRNN) and run these training files there. 


