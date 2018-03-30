# Adult-Census-Income

#### Purpose:

  This project is to predict, as accurately as possible which category, either 50K+ or 50K-, that a person’s salary lies in. 

#### 1.1 Data Extraction

  The Adult-Census-Income is from kaggle:
  
    https://www.kaggle.com/uciml/adult-census-income

  The dataset include Features:

    Salary, age, workclass, fnlwgt, education, education_num, marital-status, occupation, relationship, race, sex, capital-gain,capital-Loss, hours-per-week, native-country
  
  The `Salary` Feature is the label we want to predict.
  
#### 1.2 Data Preprocess

  
  * 1.2.1 Transforming Feature
  
      ```
      Transforming Attributes containing string values into numerical values by assigning a unique numerical value to a string        value for each feature. 
      ```
  
  
  * 1.2.2 Imputing missing values
  
      ```
      Imputate the missing values by the feature mode.
      ```
  
  
  * 1.2.3 Dealing with imbalanced data
       
       ```
       Implement Bagging method with Undersampling to balance the data.
       ```

#### 1.3 Model Selection
  
  
  * 1.3.1 Random Sampling of Training Data
  
        
        From training dataset, use Undersampling method by selecting a subset of the majority examples to match the number of minority examples to create a balanced dataset.
        
  
  * 1.3.2 Building Classifier
  
        Classification algorithms:
  
        K-nearest-neighborhood
  
        Support Vector Machine
  
        Logistic Regression
  
        Random Forest
  
        Navie Bayes
  
        Decision Tree
  
        Adaboost Decision Tree

#### 1.4 Model Comparsion


#### 1.5 Ensemble


#### 1.6 Result

