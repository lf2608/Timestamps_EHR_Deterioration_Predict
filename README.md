Utilizing Timestamps of Longitudinal Electronic Health Record Data to Predict Clinical Deterioration Events
=========================

Li-Heng Fu, MD, MA1, Chris Knaplund1, Kenrick Cato, RN, PhD2, Adler Perotte1, MD, MA, Min- Jeoung Kang, RN, PhD3,4 , Patricia C. Dykes, RN, PhD3,4 , David Albers, PhD1,5, Sarah Collins Rossetti RN, PhD1,2

*1Department of Biomedical Informatics, Columbia University, New York, NY
*2School of Nursing, Columbia University, New York, NY
*3Division of General Internal Medicine and Primary Care, Brigham and Women’s Hospital, Boston, MA 
*4Harvard Medical School, Boston, MA 
*5Department of Pediatrics, Section of Informatics and Data Science, University of Colorado, Aurora, CO


##Motivation

We propose an algorithm that utilizes only timestamps of longitudinal electronic health record (EHR) data (i.e., time and co-occurrence of vital sign measurements, flowsheets comments, order entries, and nursing notes) to predict clinical deterioration events. These time-series data reflect nurses’ decision-making related to patient surveillance. We emphasize that our data for analysis does not include any measurement values (i.e., heart rate = 90mHg). This study aims to 1) validate the proposed models built on sequences of timestamps of underlying clinical data that reflect the healthcare process, and 2) evaluate the impact of including time-of-day and time-to-outcome information in the model.

##Structure

This github folder contains the original source codes for the original article by Fu et al. 
- The data_preprocessing_pipeline.py converts original healthcare data into formated dataset for machine learning 
- The modeling_pipeline_single_point.py contains pipelines for model selection, model traininng and validation using logistic regression and random forest classifier. 
- The modeling_pipeline_rnn.py contains pipelines for model selection, model traininng and validation for RNNs

