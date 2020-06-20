This module have 3 main programs :

- cnn_w2vec is the python program to build a binary cnn classifier centered on a particular competence
  To use this program, go to the line 38 and change curr_class = "Web_Developer" with the target classes.
  The list of classes is contained in the dictionnary classes. The generate models are contained in the folder models.
  After you can run the file either with and editor or with the command python cnn_w2vec.py

- The multi_cnn.py is the program to build an evaluate the multi-label architecture. To use this program, a model must
  be generated for each target class. To run it simply run the file : python multi_cnn.py or run in an editor.

- The program cnn_filter_analyser is design to analyse filters at a global level by determining the words they are specialized in
  detecting or analysed a single prediction. Change the target datas to analyse by updating the variable curr_class. 
  Run the file with the command pyhton cnn_filter_analyser. In line if test_x[k,0] == 1889 : 1889 represent the id of the
  resume to analyse. For a global analysis simply comment the if statement.

- Unzip the file datasets/clean/500/500.zip this file contains the pretraited resume datas
- to use the pretrained base classifiers models, unzip the file models/models.zip
