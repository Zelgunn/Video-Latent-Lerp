### Code and Datasets ###

We provide all the code necessary to both train and test our method. However, we don't provide the datasets.
Datasets can be obtained the following ways:
- UCSD : [Download (from website)](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm)
- Subway : Email to Amit Adam `email.amitadam@gmail.com`
- Avenue : [Download (from website)](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)
- ShanghaiTech : [Download (from website)](https://svip-lab.github.io/dataset/campus_dataset.html)

### Run the code ###

If you want run the code, you will need to :
 
1. Download datasets
2. Put the datasets into their corresponding folder in the `./datasets` folder.
3. Run the corresponding scripts in `./code/datasets/tfrecord_builders`, they will build the datasets into TFRecord shards.
Note that you can edit the `main()` function in this files to build different subsets (eg. `entrance`or `exit`) or change the setup.
4. Select the desired dataset to train/test on in `./code/main.py` and run this script.
You can choose to train or test in the same file.

#### Datasets folder ####
After obtaining the datasets (and before building), your `./datasets` folder should look like this:

```
datasets
├───avenue
│   ├───ground_truth_demo
│   ├───testing_videos
│   ├───testing_vol
│   ├───training_videos
│   └───training_vol
├───shanghaitech
│   ├───testing
│   └───training
├───subway
│   ├───entrance
│   │   └───subway_entrance_turnstiles.AVI
│   ├───exit
│   │   └───subway_exit_turnstiles.AVI
└───ucsd
    ├───ped1
    │   ├───Test
    │   └───Train
    └───ped2
        ├───Test
        └───Train
```

#### Logs folder ####

You can find the pre-trained weights of our model in `./logs`, they will be used when testing by default. 
You don't need to move them. 

This folder is also the default directory for logs. 
Training runs can be found in the `train` folder, and testing runs can be found in the `anomaly_detection` folder.
For each run (test or train), it will create a folder. In this folder you will find:

- A copy of the config that was used for this run
- The Keras config generated for this run
- The Keras summary generated for this run

- Under `train`, you will also find:
    - Tensorboard log folders `train` and `validation`
    - Saved weights for each epoch
    
- Under `anomaly_detection`
    - A summary of the results of the test
    - Numpy arrays saved as `predictions.npy` and `labels.npy`, containing the model output and labels.
    - Plots generated from predictions (as regularity scores) and labels.

If you want to use your own saved weights, move them from the log folder of your selected run 
into the root of the logs for this dataset (along side ours).

#### Requirements ####

Python requirements can be found in `requirements.txt`. We used Python 3.6.7 and Tensorflow 2.0.0.