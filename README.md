# Principal-components-analysis
Using PCA to analysis "air quality" data, then predict the "pm2.5"

---
### Method

1. Using python-selenium to get the data of "air quality" first (data from Yunlin of Taiwan).
2. Using pca to analysis the relation of each features, then find the features which have most relation with "pm2.5".
3. Using the data of principle components to predict "pm2.5".
4. Explain the source of "pm2.5" in Taiwan.

---
### PCA Result

* pc1 variance: 34%
* pc2 variance: 20.5%

The features in pc1-pc2:

From the pc1 : the **"mean velocity of pm2.5", "mean velocity of pm10", "pm10", "NOx", "CO"** have most relation with "pm2.5".

From the pc2 : the **"CO", "mean velocity of pm2.5"** have most relation with "pm2.5".

<img src="https://user-images.githubusercontent.com/63782903/169016632-86d0de02-626c-4ea6-83e2-2b7417f67561.png" width=50%/>

---
### Regression model

Using **"pm10", "mean velocity of pm2.5 and pm10"** as our training data.

Then choose **"Linear regression"** model.

---
### Predict Result and Summary

As following image :

We can found the pm2.5 was high in November~March (11~3) which are winter of Taiwan.

The northeast monsoon prevails in Taiwan in winter. 

However, Yunlin is located in the central and western parts of Taiwan (in the leeward belt of the Central Mountains) and the wind speed is low.

Therefore, it is speculated that the pollutants may be **emitted from the local area**, and the lack of wind will lead to unfavorable diffusion which will lead to high pm2.5. 

In summer, there are southwest monsoon and subtropical high pressure weather patterns with good diffusion conditions, so the concentration of pm2.5 is not high.

<img src = "https://user-images.githubusercontent.com/63782903/169017737-2e911dee-c17b-49a7-8f24-6afe49cda7fd.png" width=50%/>

Finally, looking at these deviation values, it can be found that almost all the time points are in the early morning.

<img src= "https://user-images.githubusercontent.com/63782903/169018989-2852d73e-076d-48c3-9263-e58cab415d1e.png" widht=50%/>

