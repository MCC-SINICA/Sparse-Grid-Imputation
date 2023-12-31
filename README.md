# Sparse Grid Imputation Using Unpaired Imprecise Auxiliary Data: Theory and Application to PM2.5 Estimation

Google Drive:

https://drive.google.com/drive/u/6/folders/13oEujsA-XKSNH4sytjWZd7RR8JvVln2P

## Author: MING-CHUAN YANG, GUO-WEI WONG, & MENG CHANG CHEN

## Abstract
Sparse grid imputation (SGI) is a challenging problem, as its goal is to infer the values of the entire grid from a limited number of cells with values. 
Traditionally, the problem is solved using regression methods such as KNN and kriging, whereas in the real world, there is often extra information---usually imprecise---that can aid inference and yield better performance.
In the SGI problem, in addition to the limited number of fixed grid cells with precise target domain values, there are contextual data and imprecise observations over the whole grid. 
To solve this problem, we propose a distribution estimation theory for the whole grid and realize the theory via the composition architecture of the Target-Embedding and the 
Contextual CycleGAN trained with contextual information and imprecise observations. 

Contextual CycleGAN is structured as two generator-discriminator pairs and uses different types of contextual loss to guide the training.
We consider the real-world problem of fine-grained PM2.5 inference with realistic settings: a few (less than 1%) grid cells with precise PM2.5 data and all grid cells with contextual
information concerning weather and imprecise observations from satellites and microsensors. The task is to infer reasonable
values for all grid cells.  

As there is no ground truth for empty cells, out-of-sample MSE (mean squared error) and JSD (Jensen--Shannon divergence) measurements are used in the empirical study. 
The results show that Contextual CycleGAN supports the proposed theory and outperforms the methods used for comparison.

![image](https://github.com/MCC-SINICA/Sparse-Grid-Imputation/blob/main/example/image_2022_07_01T08_33_32_968Z.jpg)

![image](https://github.com/MCC-SINICA/Sparse-Grid-Imputation/blob/main/example/image_2022_07_01T09_35_45_149Z.png)

![19_legend](https://github.com/MCC-SINICA/Sparse-Grid-Imputation/assets/116475151/be710e75-b789-48ad-9f2d-ba0cdd736dc1)

![4124_legend](https://github.com/MCC-SINICA/Sparse-Grid-Imputation/assets/116475151/654c3a31-d0dd-4c19-a339-7a48fe1e1808)

