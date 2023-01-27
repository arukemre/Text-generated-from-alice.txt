# Text-generated-from-alice.txt
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
         ```
          > pandas 
          > numpy 
          > nltk
          > re
          > torch
          > sklearn
          > collections 
        ```
### LSTM arcitecture
> What does LSTM ?
> Please look at [https://colah.github.io/posts/2015-08-Understanding-LSTMs/]

![image](https://user-images.githubusercontent.com/64266044/215192932-99f6e9a2-ba8b-49f2-99f5-e0491bde0d0d.png)



        
### Alrigth,Which steps were placed?

*   Firstly, I have imported related ibraries for nlp preprocessing and model.Than I have created some functions towards preproceesing steps these functions facilitated pre-procesing steps.
after this sections.we are in prerrocesing step.
`Preporcessing step` purpose that how  can training camputer this sentence. We should that for train model.Firstly we must label word which in text.we can do this.We should find all uniuque word from passed preprocessing steps than we should create a dictionary for encoding unique words. After all of these steps we have a text that consist of labeled words.
After process of labeling. we creating a repesentetive tensors  thar nludes all labeled words in tex. This process a little complex for beginner stage. Firsty,I have defined a batch size and timestep.

for more detail please look at codes...

