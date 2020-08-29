# cnn_classifiers
Learning a multi-label architecture based on CNN to predict high-level skills from text resumes

This module has 3 main files :

- cnn_w2vec.py is the python program to build a binary cnn classifier centered on a particular competence.

- multi_cnn.py is the program to build and evaluate the multi-label architecture. 

- cnn_filter_analyser is used to analyse filters at a global level by determining the words they are specialized in
  detecting or to analyse a single prediction. Change the target datas to analyse by updating the variable curr_class. 
  Run the file with the command pyhton cnn_filter_analyser. At the line "if test_x[k,0] == 1889 :", 1889 represents the id of the
  resume to analyse. For a global analysis simply comment the if statement.

# Dependecies
- keras
- matplotlib
- sklearn
- numpy

# To train the cnn base models
- First, use the preprocessing project (https://github.com/florex/preprocessing) to generate the preprocessed training dataset from the raw text resumes. Edit de property output_dir to point to the directory which will contain the generated dataset.
- Then, open the file cnn_w2vec.py and edit the variable dataset_dir to point to the directory <path to the dataset>/500 containing the generated training set.
- After, edit the variable models_dir to point to the directory where you want the models (base classifiers) to be generated.
- Finally run the command : python cnn_w2vec.py 


# To evaluate the multilabel architecture :
- Edit the file multi_cnn.py and set the variables models_dir and dataset_dir with the correct paths
- Then run the command : python multi_cnn.py

# Cite this :
Jiechieu, K.F.F., Tsopze, N. Skills prediction based on multi-label resume classification using CNN with model predictions explanation. Neural Comput & Applic (2020). https://doi.org/10.1007/s00521-020-05302-x
