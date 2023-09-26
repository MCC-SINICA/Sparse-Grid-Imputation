# PythonTensorflow-Contextual CycleGAN

## Description
Left is 73 stations PM2.5 provid by EPA(Environmental Protection Administration, Taiwan, https://airtw.epa.gov.tw/)

Right is our propose method to interpolation PM2.5 whole Taiwan, 

we feed 73 stations PM2.5 into our model, and it maybe change value of 73 stations,

so we calculate 73 stations MSE (EPA 73 stations and outcome of our model)

and also we compare with KNN, Kriging.

![image](https://github.com/MCC-SINICA/Sparse-Grid-Imputation/blob/main/example/image_2022_07_01T08_33_32_968Z.jpg)

![image](https://github.com/MCC-SINICA/Sparse-Grid-Imputation/blob/main/example/image_2022_07_01T09_35_45_149Z.png)
