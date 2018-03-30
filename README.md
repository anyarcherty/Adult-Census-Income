# Adult-Census-Income

#### Purpose:

  This project is to predict, as accurately as possible which category, either 50K+ or 50K-, that a person’s salary lies in. 

#### 1.1 Data Extraction

   * The Adult-Census-Income is from kaggle:
   
   
        https://www.kaggle.com/uciml/adult-census-income
   
    
   * The dataset include Features:

        Salary, age, workclass, fnlwgt, education, education_num, marital-status, occupation, relationship, race, sex, capital-gain,capital-Loss, hours-per-week, native-country
  
   * The `Salary` Feature is the label we want to predict.
  
#### 1.2 Data Preprocess

  
  * Transforming Feature
  
      ```
      Transforming Attributes containing string values into numerical values by assigning a unique numerical value to a string        value for each feature. 
      ```
  
  
  * Imputing Missing Values
  
      ```
      Imputate the missing values by the feature mode.
      ```
  
  
  * Dealing with Imbalanced Data
       
       ```
       Implement Bagging method with Undersampling to balance the data.
       ```

#### 1.3 Model Selection
  
  
  * Random Sampling of Training Data
      
      From training dataset, use Undersampling method by selecting a subset of the majority examples to match the number of minority examples to create a balanced dataset.
        
  
  * Building Classifier
  
      Classification algorithms:
  
      `K-nearest-neighborhood`, `Support Vector Machine`, `Logistic Regression`, `Random Forest`, `Navie Bayes`, `Decision Tree`, `Adaboost Decision Tree`
     
   * Ensemble Learning
    
        Each instance of training and test data is classified 0 (corresponding to less or equal than 50K dollars annually) or 1 (corresponding to greater than 50K annually) using the learned classifiers.
        
        Final prediction is made by taking a majority vote model among the predictions of these classifiers

#### 1.4 Result
   
   The result is shown by the confusion matrix
   
   * Bagging Decision tree:
   
        Confusion matrix:
    
        | Prediction         |  Truth        |   Truth       |
        | ----------------   |-------------  |-------------  | 
        |                    | <=50K         |    >50K       |
        | <=50K              | 0.63          |    0.13       |
        | >50K               | 0.05          |    0.19       |
        
        Prediction accuracy for instances label <= 50K is `0.83`.
        
        Prediction accuracy for instances label > 50K is `0.80`.
        
        Overall Test Accuracy is `0.82`.
        
   * Random Forest
   
        Confusion matrix:
    
        | Prediction         |  Truth        |   Truth       |
        | ----------------   |-------------  |-------------  | 
        |                    | <=50K         |    >50K       |
        | <=50K              | 0.62          |    0.15       |
        | >50K               | 0.04          |    0.20       |
        
        Prediction accuracy for instances label <= 50K is `0.81`.
        
        Prediction accuracy for instances label > 50K is `0.85`.
        
        Overall Test Accuracy is `0.82`.
        
   * Logistic regression
   
        Confusion matrix:
    
        | Prediction         |  Truth        |   Truth       |
        | ----------------   |-------------  |-------------  | 
        |                    | <=50K         |    >50K       |
        | <=50K              | 0.64          |    0.12       |
        | >50K               | 0.04          |    0.19       |
        
        Prediction accuracy for instances label <= 50K is `0.84`.
        
        Prediction accuracy for instances label > 50K is `0.82`.
        
        Overall Test Accuracy is `0.83`.
        
   * K-Neighbor
   
        Confusion matrix:
    
        | Prediction         |  Truth        |   Truth       |
        | ----------------   |-------------  |-------------  | 
        |                    | <=50K         |    >50K       |
        | <=50K              | 0.61          |    0.15       |
        | >50K               | 0.05          |    0.19       |
        
        Prediction accuracy for instances label <= 50K is `0.80`.
        
        Prediction accuracy for instances label > 50K is `0.80`.
        
        Overall Test Accuracy is `0.80`.
        
   * Support Vector Machine(SVM)
   
        Confusion matrix:
    
        | Prediction         |  Truth        |   Truth       |
        | ----------------   |-------------  |-------------  | 
        |                    | <=50K         |    >50K       |
        | <=50K              | 0.63          |    0.13       |
        | >50K               | 0.05          |    0.19       |
        
        Prediction accuracy for instances label <= 50K is `0.83`.
        
        Prediction accuracy for instances label > 50K is `0.80`.
        
        Overall Test Accuracy is `0.82`. 
   
   * Naïve Bayes
   
        Confusion matrix:
    
        | Prediction         |  Truth        |   Truth       |
        | ----------------   |-------------  |-------------  | 
        |                    | <=50K         |    >50K       |
        | <=50K              | 0.61          |    0.15       |
        | >50K               | 0.04          |    0.20       |
        
        Prediction accuracy for instances label <= 50K is `0.80`.
        
        Prediction accuracy for instances label > 50K is `0.84`.
        
        Overall Test Accuracy is `0.81`. 
        

      * Ensemble: Majority Vote for 7 Learned Classifiers
   
        Confusion matrix:
    
        | Confusion Matrix   | Truth <=50K   |  Truth >50K   | 
        | ----------------   |-------------  |-------------  |
        | Prediction <=50K   | 0.63          |    0.14       |
        | Prediction >50K    | 0.04          |    0.20       |
        
        Prediction accuracy for instances label <= 50K is `0.82`.
        
        Prediction accuracy for instances label > 50K is `0.84`.
        
        Overall Test Accuracy is `0.83`.  
