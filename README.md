# Image_Scenery_Recognition
This repository contains R script to the in-class Kaggle activity on classifying images into scenery categories. 
This activity was part of the Behavioural Data Science course at the University of Amsterdam, carried out by me and Vincent 
Schreuder. 


In this activity we were given images of scenery. These could be classified into 4 main categories, "Trees and forests", "Rivers", "Sunsets" and "Cloudy Skies". The goal was to build a model that could classify an image into 1 of the 4 categories. 

We first split the image into 9 segments and then got the value for the red, green and blue colour channels in each segment. 
Then, for these color values in each segment we computed the mean, quantiles, auto-correlations, spectracl features etc. 

To build the final model we tried a range of tree based methods, including Random Forests and Gradient Boosting. In the end we got to 84% accuracy using a gradient boosting model. 
