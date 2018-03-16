# HealthAnalytics
CS205 Health Analytics project


Introduction
------------

The project requires us to build a model that predicts the positioning of a user. A user can be either walking, standing, sitting and laying down. The data is collected from a Sony SmartWatch using an app called “Decent Logger”. The app needs to be installed on the watch. 

Installation and data collection
---------------------------------

The app “Decent Logger” needs to be installed on the watch first. The new watch needs to be initialized first. The app “Android Wear”  needs to be installed on Android phone. The smartwatch needs to be paired with the phone. The phone is no longer needed after the pairing. 
Trigger the “Developer” and enable ADB Debug mode on the watch . Install ADB on the laptop which allows the laptop to connect to watch through ADB. “adb install DecentLogger_0.apk” command on the terminal installs the app on the watch.”adb devices” shows the devices connected while indicating whether the laptop is authorized to install the app. 
Start the Decent_logger app and collect the data. We set the label for either “Sitting”,”Standing”, “Lying Down” and “Walking” and collect the data and then use “unknown” to stop collecting the data.
After collecting data , we need to merge the files, perform extraction and ultimately build the classification model. A part of the dataset is used as Test dataset and wouldn’t be used in building the model.

Our Solution
--------

Data Processing
----------------
In the data processing step, we clean the data and make it into the format that is readable by our model . The data collected from the watch is compressed and stored in tgz format. First we uncompress all the files and then build the input data as a table with the first column representing the label and the rest of columns represents the different inputs such as accelerometer, magnetic_field, rotation_vector, step_counter, orientation and light. Each row represents on reading recorded at that particular timestamp. We try to fill out missing timestamps if possible if it between two successive timestamps. At the end we remove all rows that don’t have timestamps. 

Model Training and Validation
------------------------------
The preprocessed data is split into Training and Testing datasets. We have decided to split in 75:25 ratio. The dataset is labelled and correspond to Walking (0), Standing(1),Sitting(2) and Lying down(3) output labels. The first column represents the output label , and the rest of the column data correspond to accelerometer and gyroscope sensor readings. We have used 3-layer Fully connected network. In a fully connected layer, each neuron is connected to every neuron in the previous layer and each connection has its own weight.The three layers use 4 neurons each. The first 2 layers has RELU activation. The output layer has softmax for getting probabilities for each of label. Adam is used for gradient descent optimization and categorical_crossentropy is used for loss calculation. 
The above network is trained for 500 epochs with batch_size = 50, and validation data is set to 10% of the training data. 

Model testing
--------------
The testing data is provided in csv format. It will preprocessed and sent to the our neural network. In our case , we had reserved 25% of the dataset for training. We got 80% accuracy on the training data.
