# Dogs vs. Cats Classification

This is the solution for part2 of Convolutional Networks programming assignment. To run the script, first make sure the data set has been downloaded. To do that, you can simply run the following command:

    python download_cat_dog.py

For more information please visit [this](https://github.com/CW-Huang/IFT6135H18_assignment) repository.

After the data set is downloaded (which is about 1GB), you can run the model by the following command:

    python model.py -f ../../config.yaml -r Q2

The above command will use the config file in [../../](https://github.com/hossein1387/IFT6135/blob/master/Assignments/Assignment2/config.yaml) which is in [YAML](https://en.wikipedia.org/wiki/YAML) format. We also pass Q2 as the configuration needed for Question2. You will get something like this:

    ========================================================
    Configuration:
    ========================================================
    initial learning rate is set to: 0.1
    number of epochs: 10 
    momentum is set to: 0.9 
    batch size for training is set to: 25 
    batch size for validation is set to: 50 
    batch size for test is set to: 1 
    
    ('Loaded images(train and valid), total', 20000)
    ('Loaded test images, total', 4990)
    ('Image size: ', (3L, 64L, 64L))

Which shows the configuration of the model and it will show if the dataset has been loaded properly.