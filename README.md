# Cereal Buddy

## Introduction
Like having cereal for breakfast? Tired of pouring your own milk? Worry not, Cereal Buddy is here to help!

Cereal Buddy runs on a Raspberry Pi and utilizes the [Raspberry Pi Camera Module](https://www.raspberrypi.org/products/camera-module-v2/) with [OpenCV](https://opencv.org/) to detect when a bowl of cereal is placed in front of it, then dispenses milk!

## How it works
Cereal Buddy uses a stochastic gradient descent (SGD) model from [Scikit-Learn](https://scikit-learn.org/) to classify images by whether or not they contain a bowl of cereal. First, it uses OpenCV to extract information on ellipse shapes from the image, and then it feeds that information into the SGD to figure out whether or not the ellipse is a bowl of cereal. If it is, then it's time for milk!

An SGD was chosen to make the model because it supports *online learning*, which is a type of machine learning that can continually take in new data and update itself, so instead of having a large dataset to train on right from the start, you can just start with a basic model and show it new examples as you get them, which is incredibly useful becuase there don't seem to be any large datasets of cereal pictures available online short of downloading the Google Image search results. Cereal Buddy utilizes its online learning capability through its `train.py` script. When you run it, it will show you images from the Pi camera and whether or not it thinks they contain cereal. Your job is then to press either the `y` or `n` key to tell it whether or not it was correct; it will learn this information and improve itself, creating a better model with every step!

## FAQ
**Wow this is exactly what I've always needed!**  
*I know, that's why I built it!*

**The model only uses geometric features about the bowl's shape? How can it tell the difference between cereal and miso soup, or a cheese wheel?**  
*Shut up*
