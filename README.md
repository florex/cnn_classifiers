# cnn_classifiers
Learning a multi-label architecture based on CNN to predict high-level skills from text resumes

This module have 3 main files :

- cnn_w2vec.py is the python program to build a binary cnn classifier centered on a particular competence.

- The multi_cnn.py is the program to build an evaluate the multi-label architecture. 

- The program cnn_filter_analyser is used to analyse filters at a global level by determining the words they are specialized in
  detecting or to analyse a single prediction. Change the target datas to analyse by updating the variable curr_class. 
  Run the file with the command pyhton cnn_filter_analyser. At the line "if test_x[k,0] == 1889 :", 1889 represents the id of the
  resume to analyse. For a global analysis simply comment the if statement.

# Dependecies
- keras
- matplotlib
- sklearn
- numpy

# To train the cnn base models
- Using the preprocessing project, generate the preprocessed training dataset from the raw text resume.
- inside the file cnn_w2vec.py, edit the variable dataset_dir to point to the directory <path to the dataset>/500
- Models_dir will contain the trained models of each base classifier. Edit this variable to point to the directory where you want the models to be generated

- FInally run the command : python cnn_w2vec.py 


# To evaluate the multilabel architecture :
- Edit the file multi_cnn.py and set the variables models_dir and dataset_dir with the correct paths
- Then run the command : python multi_cnn.py
