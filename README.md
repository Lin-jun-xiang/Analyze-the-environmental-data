# Analyze the environmental data

---
### Introduction

* Using Principal-components-analysis (PCA) to analysis "air quality" data, find out the features related to "pm2.5"
* Using data of 2018 to predict pm2.5 of 2019
* Analyze the source of pm2.5

---
### Method

1. Using python-selenium to get the data of "air quality" first (data from Yunlin of Taiwan).
2. Using pca to analysis the relation of each features, then find the features which have most relation with "pm2.5".
3. Using the data of principle components to predict "pm2.5" with linear regression.
4. Explain the source of "pm2.5" in Taiwan.

---
### PCA Result

* pc1 variance: 34%
* pc2 variance: 20.5%

The features in pc1-pc2:

From the pc1 : the **"mean velocity of pm2.5", "mean velocity of pm10", "pm10", "NOx", "CO"** have most relation with "pm2.5".

From the pc2 : the **"CO", "mean velocity of pm2.5"** have most relation with "pm2.5".

<p align="center">
<img src="https://user-images.githubusercontent.com/63782903/169016632-86d0de02-626c-4ea6-83e2-2b7417f67561.png" width=50%/>
</p>

---
### Regression model

Using data of 2018 as our training data.

Then choose **"Linear regression"** model.

Cross validation method : **K-fold**

Predict pm2.5 of 2019.

---
### Predict Result

Prediction score : 80%

As following image :

We can found the pm2.5 was high from November to March (11~3) which are winter season of Taiwan.

Moreover, the pca result show the **mean velocity** are important correlation with pm2.5.

Therefore, we can discuss the pm2.5 from **"season"**.

The northeast monsoon prevails in Taiwan in winter.

However, Yunlin is located in the central and western parts of Taiwan (in the leeward belt of the Central Mountains) which have lower wind speed.

Therefore, it is speculated that the pollutants may be **emitted from the local area**, and the lack of wind will lead to unfavorable diffusion which will lead to high pm2.5. 

In summer, there are southwest monsoon and subtropical high pressure weather patterns with good diffusion conditions, so the concentration of pm2.5 is not high.

<p align="center">
<img src = "https://user-images.githubusercontent.com/63782903/178513614-9799464f-c477-4c82-ba75-dae4a23dd7a4.png" width=50%/>
</p>

---
### Analysis of high pm2.5

Now, we know the difference season will results in concentration of pm2.5.

Let's intensive the feature of **"Time"**.

Firstly, We find out the **outlier** of pm2.5 in each month according to **"percentile"** and **"boxplot"** :

<p align="center">
<img src="https://user-images.githubusercontent.com/63782903/178486743-d492cc96-8a49-4456-b5de-db0449281c67.png" width=50%/>
</p>

Then visualize the counts of outlier at each times as following image :

We can see the lower pm2.5 happen in the range of 12:00 pm to 20:00 pm which have good diffusion conditions.

In constrant, the higher pm2.5 happen in early morning which means the pm2.5 **emitted from the local area** cannot diffusion well.

<p align="center">
<image src="https://user-images.githubusercontent.com/63782903/178486808-996d3afc-16e3-4bbb-a1a9-6aa9189b9f13.png" width=50%/>
</p>
