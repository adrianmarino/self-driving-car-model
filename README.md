#  Self driving car model

* The model learn to control steering wheel angle.
* Throttle & Break is controlled by a PI Controller.
* The model was trained using all tracks of * Using [Udacity self-driving-car-simator](https://github.com/udacity/self-driving-car-sim) _Version 2_ and tested with track 2 of _Version 1_. The idea was try model with a track that it never saw.
* The model was based to [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) network arquitecture.


[![](http://img.youtube.com/vi/B5Q4MbLvtwI/0.jpg)](http://www.youtube.com/watch?v=B5Q4MbLvtwI "Track 2 of self-driving-car-simator V2")

## Requeriments

* conda
* A respectable video card (i.e. GeForce GTX 1060 or higher)

## Setup

Create project environment:

```bash
$ conda env create --file environment.yml
```

**Step 2**: Activate environment.
```bash
$ conda activate self-driving-car-model
```
This last step is required adfer run train_mode.py or drive.py.

## Train model

You can train and adjust model from [
Self driving car model analysis
](https://github.com/adrianmarino/self-driving-car-model/blob/master/model-analysis.ipynb) notebook or use 
train.model.py

To train model using train.model.py:
```bash
$ python train_model.pyd
```
This script load model last weights from checkpoints path is exist. 


## Play model

To test model with a track that it never saw use [self-driving-car-sim](https://github.com/udacity/self-driving-car-sim) _Version 1_. You can also test model with any track from _Version 2_, but the model already know this tracks, given these were used to generate the training and validation samples.

### To play model

**Step 1**: First execute simulator, select _Track 2_ and press _Autonomous Mode_.

**Step 2**: Activate environment.

```bash
$ conda activate self-driving-car-model
```
**Step 2**: execute model client.

```bash
$ python drive.py checkpoints/weights__loss_0.0525__rmse_0.2055.h5
```
