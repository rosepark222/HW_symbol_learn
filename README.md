# Bidirectional Recurrent Neural Network Training 

Overview of the project: Imagine a learner works on math problems in her iPad software and she writes down her answer in a math equation form using Apple Pencil. When education software recognizes an incorrect ansewr, it will provide relevant feedback to her so that she can learn necessary concepts to solve the original problem. Meanwhile, her misconceptions are analyzed from her answer and appropriate insights will be generated for relevant stake holders - her teachers or parents. This project is an initial step for this goal using the latest machine learning technique. An essential technique for this learning ecosystem is Handwritten Recognition System for Math Equations. In this project, a bidirectional RNN has been trained to recognize 53 math strokes (not symbols -- details comes below). Future developments are planned for complete math equation recognition.

The best way to have the feel for this project is to dabble with the deployment in the Google cloud: http://pradoxum001.appspot.com/

---

For some background information about data preparation, please refer to https://github.com/rosepark222/HW_Rcode

---

A bidirectional RNN was trained to recognize strokes. The dimensions of input, the first and second hidden layers are 15, 64, and 32, respectively. [LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) cells were used to overcome vanishing gradient problem. In addition, LSTM can be trained to forget and release accumulated information for better performance. For example, when the subjects in sentences have changed from singular to plural in Natural Language Processing, the previously registered information from the subject being singular should be updated by the current information. 

Prior to the training, data were balanced to have 1,000 sample per strokes. A total of 53 strokes are trained. At the end of training, the loss, MSE and accuracy were about 0.002, 3E-6, and .999, respectively. The batch size was 500, the number of epochs for the optimization was slightly above 1,000, and the number of steps within each epoch was 30. The learning rate was not adjusted at this moment. For this reason, there were spikes of loss values once they became close to zero (i.e., smaller than 0.01). Adjusting the learning rate might prevent these surge of loss values toward the end of optimization. Keras was programmed to save the model parameters whenever the loss decreases from the previous epoch during the training. In this way, the simulation always store the best result.

<!--- In the current project, it is not clear how the [forget gates](https://datascience.stackexchange.com/questions/19196/forget-layer-in-a-recurrent-neural-network-rnn) in LSTM would help the learning. --> 

It is observed that the network did not learn the differences between similar symbols such as '9', 'g', and 'q'. Furthermore, if one stroke completely overlaps with another (e.g., 's’ and ‘8’), the learned model often failed to classify them. Empirical tests showed that '8' was confused with 's' but 's' was not confused with '8'. An interesting note is that '8' written in clockwise was not confused with 's' written in counterclockwise. At the beginning of the project, there was an issue that RNN was confused the x_1_1 (i.e., the first stroke of lower case x, looks like a flipped c -- see below) and '2'. 

<img width="115" alt="x_1_1 shot 2018-11-02 at 5 22 08 pm" src="https://user-images.githubusercontent.com/38844805/47946055-e82c3380-dec3-11e8-9317-6b8ece3edfc2.png">

When offline features (see https://github.com/rosepark222/HW_Rcode) were added as inputs to the RNN, this confusion *magically* disappeared. This may be due to the offline features providing unique information distinguishing the two strokes. 

To overcome this shortcomings, additional models such as Convolutional Neural Network (CNN) can be added for hard-to-distinguish strokes. The sample size of unique 's' and '8' were 170 and 760, respectively, and these sample sizes seem too small for any neural network to learn. RNN should be able to learn to tell the difference between 's' and '8', if more samples are added.  

Important factors that have improved the learning are -- 
     1) adopting bidirectional RNN, 
     2) adding offline features, balancing the data, 
     3) removing mislabeled strokes. 

These design choices are ordered in the sequence of project modifications. As long as the confusions are consistent, the correct recognition can be accomplished using the probablistic language model [Muñoz, 2015]. Further, the problems in an e-learning platform may provide extra information regarding the likelihood of symbols. For example, students are likely to use 'k' in their answers if the description of the problem uses 'k' as a variable. The problem can even require students to use certain symbols to describe their answers. Based on this contextual information, the next step for this project would be the development of probabilistic models for symbol recognition.   

Ideas for performance improvement:

 1. Interpolate original trace points -- The number of trace points are different based on the device a user is on. How can those traces can be regulated? 
 
## Reference 

Muñoz, F. Á. (2015). Mathematical Expression Recognition based on Probabilistic Grammars (Doctoral dissertation).


