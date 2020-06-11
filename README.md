Utilizing Timestamps of Longitudinal Electronic Health Record Data to Predict Clinical Deterioration Events
=========================

Li-Heng Fu, MD, MA1, Chris Knaplund1, Kenrick Cato, RN, PhD2, Adler Perotte1, MD, MA, Min- Jeoung Kang, RN, PhD3,4 , Patricia C. Dykes, RN, PhD3,4 , David Albers, PhD1,5, Sarah Collins Rossetti RN, PhD1,2

*1Department of Biomedical Informatics, Columbia University, New York, NY
*2School of Nursing, Columbia University, New York, NY
*3Division of General Internal Medicine and Primary Care, Brigham and Women’s Hospital, Boston, MA 
*4Harvard Medical School, Boston, MA 
*5Department of Pediatrics, Section of Informatics and Data Science, University of Colorado, Aurora, CO


##Motivation
We propose an algorithm that utilizes only timestamps of longitudinal electronic health record (EHR) data (i.e., time and co-occurrence of vital sign measurements, flowsheets comments, order entry, and nursing notes) to predict clinical deterioration events. These time-series data reflect nurses’ decision-making related to patient surveillance[16 18]. We emphasize that our data for analysis does not include any measurement values (i.e., heart rate = 90mHg). This study aims to 1) validate the proposed prediction models built on the time series of data entry timestamps that reflect the healthcare process, 2) and evaluate the impact of including a variable representing time in the model.

##Structure
This github folder contains the original source codes for the original article by Fu et al. 
- The Create_dataset.ipynb file converts original healthcare data into formated dataset  ready for machine learning 
- The Model Derivation and Validation.ipynb provides all models that were built in this manucript. 

##Models
- Logistic Regression with L2 Regularization
- Deep Neural Network
- LSTM
- GRU
