# GCNN-DDGF_{rec}: GCNN-DDGF with recurrent architecture
The model is implemented on the basis of DCRNN, we replace the diffusion convolution with DDGF convolution. This is the main reason that our model is much faster than DCRNN. 

## GCNN-DDGF Model Training
```bash
# METR-LA-Speed
python dcrnn_train.py --config_filename=data/model/dcrnn_la.yaml

# PEMS-Volume
python dcrnn_train.py --config_filename=data/model/dcrnn_bay.yaml # change file name
```
## DCRNN Model Training
For METR-LA-Speed, we just use the same hyperparameter file provided in DCRNN. 
For PEMS-Volume, we tried different configurations, and the best one is also provided here for comparison. 
Note that because we change the DCRNN codes, the interested users need to download the original DCRNN models and run these training files there. 


