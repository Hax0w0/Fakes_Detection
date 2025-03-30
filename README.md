# Fakes Detection README
 **Project**: Fakes Detection<br>
 **Class**: Northwestern CS 461 Winter 2025<br>
 **Contributers**: Raymond Gu, Maanvi Sarwadi

## Fake Biographies Dataset
The `Fake Biographies` dataset was provided to us by Professor Demeter.<br>

Each entry in the dataset is in the following format:
```
<start_bio>
  = Name =
  ...
  == Section #1 ==
  ...
  == Section #2 ==
  ...
<end_bio>
```
One important thing to note is that each entry can have different sections. An example of 2 biographies with different sections is shown below:
```
<start_bio>  
  = Amelia Smith Calvert =   
  Amelia Smith Calvert ( 1876 , Philadelphia-1965 ) was an ...

  == Works ==   
  Calvert , Amelia Smith ; Calvert , Philip Powell ( 1917 ) ...
<end_bio>
```
```
<start_bio>
  = Richard C. Halverson =   
  The Reverend Richard Christian Halverson ...
    
  == Biography ==   
  He was born in Pingree , North Dakota ...

  == Awards ==   
  Halverson received the Distinguished Alumnus Award ...
    
  == Books ==   
  Halverson authored several books in the 1950s–1990s ...
<end_bio>
```

However, after the name of the person, there is always a small description of them. This is the only section that all entries in the dataset share. For this project, we've decided to only use this section to determine if an entry is real or fake in order to prevent our model from learning patterns that rely on optional sections.

## Cleaning Data
The `Data_cleaner.py` file contains all the code needed to clean and split the data.<br>

The distribution of real and fake biographies for the original dataset is shown below:
- **Real Biographies**: 9,947 <br>
- **Fake Biographies**: 9,970 <br>

However, some of the entries in the original dataset are not biographies. An example of an entry that isn't a biography is shown below:
```
<start_bio>  
 =   
 − sin(i − r)
 sin (i + r)
 { \displaystyle r_ { s } =- { \frac { \sin ( i-r ) } { \sin ( i+r ) } } }   
 and
 r
 p 
<end_bio>
```

After removing all entries that are not biographies, the distribution is:
- **Real Biographies**: 9,928 <br>
- **Fake Biographies**: 9,796 <br>

As stated earlier, only the description of the person at the very beginning of the biography is used since it is the only section that all entries have in common. Details about each of the datasets after cleaning and preprocessing is shown below:
- **Training**: This dataset contains 15,648 biographies (7,824 biographies of each type).<br>
- **Validation**: This dataset contains 1,956 biographies (978 biographies of each type).<br>
- **Test**: This dataset contains 1,956 biographies (978 biographies of each type).<br>

Each of these datasets are saved as excel files (`Train.xlsx`, `Valid.xlsx`, `Test.xlsx`).<br><br>

## Fakes Detection Model
To do this task, we decided to use a transformer-based model and take a classification approach. We used the bert-base-uncased checkpoint for the `BertModel` and `AutoTokenizer` from HuggingFace Transformers.<br>

After a question is passed through BERT, the model outputs a sequence of embeddings, in which the [CLS] embedding is extracted. A linear layer is then applied to the [CLS] embedding to produce logits, which represent the probability of the biography being fake or real. The class with the highest probability is used for the model's prediction. <br><br>

## Results Analysis
Our model was able to achieve a f1-score of ~0.8 for both classes with an accuracy of 82%. For our baseline result, we used logistic regression and achieved a f1-score of ~0.72 for both classes with an accuracy of 72%. These results are interesting since our model's results are only slightly above the baseline.<br>

One factor that may have limited the effectiveness of our model could've been the way we preprocessed the data. By only using the initial description at the very start of each biography (which ranged from 1 sentence to an entire paragraph), we could've left out a lot of valuable data from the other sections that could've been helpful for this task.