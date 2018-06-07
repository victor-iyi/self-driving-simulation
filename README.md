# Self Driving Simulation

A self-driving car in a simulated environment. Explore various state-of-the-art methods of autonomous self-driving car in a fun visual format.

- Built in [Unity3D](https://unity3d.com/) *(free game making engine)*.
- Add new tracks, change prebuilt scripts like gravity acceleration easily.

<!-- Simulator Screenshot -->
![Jungle Track](images/jungle_track.png)

**Download Links:** [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983558_beta-simulator-linux/beta-simulator-linux.zip), [Mac](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983385_beta-simulator-mac/beta-simulator-mac.zip), [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983318_beta-simulator-windows/beta-simulator-windows.zip)

## Table Of Content

- [Setup](#setup)

- [Usage](#usage)
  - [How the Simulator works](#how-the-simulator-works)
  - [Run the pre-trained model](#run-the-pre-trained-model)
  - [Train the model](#train-the-model)

- [About the Model](#about-the-model)
  - [Training Mode - Behavioral cloning](#training-mode---behavioral-cloning)
  - [Hardware design](#hardware-design)
  - [Software Design (supervised learning)](#software-design-(supervised-learning))

- [Credits](#credits)

- [Contribution](#contribution)

## Setup

All required dependencies are neatly packed in the ``requirements.txt`` file.

> **NOTE:** This project was developed with Python 3.6.5 therefore use the appropriate Python interpreter *(i.e. ``python3`` instead of ``python`` which could most likely be Python 2.7 & ``pip3`` instead of ``pip``)*.

```sh
> pip install --upgrade pip
```

Then you probably want to work from your local PC:

Start by cloning the project from github:

```sh
> git clone https://github.com/victor-iyiola/self-driving-simulation.git
> cd self-driving-simulation
```

or:

You can download the `.zip` project files [here](https://github.com/victor-iyiola/self-driving-simulation) and extract the project files.

```sh
> cd /path/to/self-driving-simulation
```

Install these requirements:

```sh
> pip install --upgrade -r requirements.txt
```

## Usage

### How the Simulator works

- Records images from center, left, and right cameras w/ associated *steering angle*, *speed*, *throttle* and *brake*.
- Saves to `driving_log.csv`
- Ideally you have a joystick, but keyboard works too.

### Run the pre-trained model

To drive, run simulator *Autonomous Mode* (click Autonomous Mode in main Menu), then run `drive.py` as follows:

```sh
> python drive.py model-005.h5
```

### Train the model

To train, generate training data (press `R` while in *Training Mode*) with the Simulator and save
recordings to `path/to/self_driving_car/data/`.

```sh
> python3 model.py
```

This will generate a file `model-{epoch}.h5` whenever the performance in the epoch is better
than the previous best.  For example, the first epoch will generate a file called `model-000.h5`.

## About the Model

### Training Mode - Behavioral cloning

A 9 layer convolutional network, based off of Nvidia's [End-to-end learning for self driving car](https://arxiv.org/pdf/1604.07316) paper.
72 hours of driving data was collected in all sorts of conditions from human drivers <https://www.youtube.com/watch?v=NJU9ULQUwng>

### Hardware design

- 3 cameras
- The steering command is obtained by tapping into the vehicle’s Controller Area Network (CAN) bus.
- Nvidia's Drive PX onboard computer with GPUs

In order to make the system independent of the car geometry, the steering command is 1/r, where r is the turning radius in meters.  1/r was used instead of r to prevent a singularity when driving straight (the turning radius for driving straight is infinity). 1/r smoothly transitions through zero from left turns (negative values) to right turns (positive values).

### Software Design (supervised learning)

Images are fed into a CNN that then computes a proposed steering command. The proposed command is compared to the desired command for that image, and the weights of the CNN are adjusted to bring the CNN output closer to the desired output. The weight adjustment is accomplished using back propagation

Eventually, it generated steering commands using just a single camera.

## Credits

- Udacity [Self-Driving Nanodegree](https://github.com/udacity/self-driving-car-sim)
- Nvidia
- naokishibuya

## Contribution

This project is opened under [MIT 2.0 license](https://github.com/victor-iyiola/self-driving-simulation/blob/master/LICENSE).
