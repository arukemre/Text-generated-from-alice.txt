# Text generative model scratch with python
![image](https://user-images.githubusercontent.com/64266044/215060141-b23a8a37-aa72-4dda-bc5f-a9fb3c3e5501.png)

Hi. I want to introduce this repository towards enhance our nlp skill with python.
This repository includes Some preprocessing steps and model scratch using torch library.
well alraigth what did ı do.
I found a text from web `alice.txt` and with this text that have generated a new text as a continuaton.

    |─ alice.txt
    |─ TextGenerativeModel.py
    
        |─ Data Preview
        |─ Data preprocessing
        |─ Data preprocessing and Feature Engineering steps
        |─ textGenerator class
        |─ Model optimization with `Ray Tune` from `Pytorch` library
    
    
   #### Requaried libraries
          > pandas 
          > numpy 
          > nltk
          > re
          > torch
          > sklearn
          > collections 
        
### LSTM arcitecture
> What does LSTM ?

> Please look at   [LSTM link](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

![image](https://user-images.githubusercontent.com/64266044/215192932-99f6e9a2-ba8b-49f2-99f5-e0491bde0d0d.png)



        
### Alrigth,Which steps were placed?

``` Firstly, I have imported related libraries for nlp preprocessing and model. 
Then I created some functions for preprocessing steps these functions facilitated pre-processing steps.
After this section. we are in preprocessing step.
The Preprocessing step purpose that how can train computers with these sentences. In order to train the model, we must. Firstly we must label words in the text.
We can do this. We should find all unique words from passed preprocessing steps then we should create a dictionary for encoding unique words.
After all of these steps, we have a text that consists of labeled words.After the process of labeling. 
we created representative tensors that include all labeled words in tex. This process is a little complex for the beginner stage.
For more detail please look at the codes... ```

