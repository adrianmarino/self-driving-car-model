#  Self driving car model

* The model learn to control steering wheel angle.
* Throttle & Break is controlled by a PI Controller.
* The model was trained using all tracks of [Udacity self-driving-car-simator](https://github.com/udacity/self-driving-car-sim) _Version 2_ and tested with track 2 of _Version 1_. The idea was try model with a track that it never saw.
* The model was based to [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) network arquitecture.

## Tests

**Test 1**: Test model on a track that was used to generate training and test samples.

<p align="center">
    <a href="http://www.youtube.com/watch?v=B5Q4MbLvtwI" target="_tab"/>
    <img src="http://img.youtube.com/vi/B5Q4MbLvtwI/0.jpg" 
        title="Track 2 of self-driving-car-simator Version 2" 
        alt="Track 2 of self-driving-car-simator Version 2"/>
    </a>
</p>

**Test 2**: Test model over a track that it never saw.

<p align="center">
    <a href="http://www.youtube.com/watch?v=FAYoct9GfQc" target="_tab"/>
    <img src="http://img.youtube.com/vi/FAYoct9GfQc/0.jpg" 
        title="Track 2 of self-driving-car-simator Version 1" 
        alt="Track 2 of self-driving-car-simator Version 1"/>
    </a>
</p>

## Requeriments

* [anaconda](https://www.anaconda.com/download/#linux)
* 7z
* A respectable video card (i.e. [GeForce GTX 1060](https://www.nvidia.com/en-us/geforce/products/10series/geforce-gtx-1060/) or higher)

## Setup

**Step 1**: Create project environment.

```bash
$ conda env create --file environment.yml
```

**Step 2**: Activate environment.
```bash
$ conda activate self-driving-car-model
```
**Note**: This step is required after run `train_mode.py` or `drive.py`.

## Train model

You can train and adjust model from [
Self driving car model analysis
](https://github.com/adrianmarino/self-driving-car-model/blob/master/model-analysis.ipynb) notebook or use 
`train_model.py` script. 
First of all you need a dataset, but already exist a dataset that was created to train the model, so you can download this. To train model follow next steps:

**Step 1**: Then to train model first of all download dataset from [here](https://drive.google.com/file/d/1O84dTrE2j1J9xhPmlJdVwRJ55WcJlMQN/view?usp=sharing) to project path. 
 
**Step 2**: Next extract dataset.
```bash
$ 7z x self-driving-car-dataset.7z
```

**Step 3**: To train model using `train_model.py` script:
```bash
$ python train_model.py
```
This script load model last weights from /checkpoints path if it exists. 


## Play model

To test model with a track that it never saw use [self-driving-car-sim](https://github.com/udacity/self-driving-car-sim) _Version 1_. You can also test model with any track from _Version 2_, but the model already know this tracks, given these were used to generate the training and validation samples.

### To play model

**Step 1**: Download [self-driving-car-sim](https://github.com/udacity/self-driving-car-sim) _Version 1_.

**Step 2**: Execute simulator, select _Track 2_ and press _Autonomous Mode_.

**Step 3**: Activate environment.

```bash
$ conda activate self-driving-car-model
```

**Step 4**: execute model client.

```bash
$ python drive.py checkpoints/weights__loss_0.0525__rmse_0.2055.h5
```
