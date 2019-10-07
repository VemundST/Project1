
This repository is for Project 1 in fys-stk 4155.

The authors of the code and report are:
- Astrid Tesaker
- Vemund Stenbekk Thorkildsen
- Sofie Tunes

The script functions.py includes the main algorithms. The functions included are:

- DesignDesign --> for creating design matrix 
- OridinaryLeastSquares --> OLS regression
- RidgeRegression
- k_fold_cv --> K-fold cross validation
- reshaper --> Used for sorting inside k_fold_cv
- N_bootstraps --> Bootstrap resampling and prediction.
- Bootstrap --> The actual resampling employed by N_bootstraps.
- MSE, r2, confidence_interval --> Error metrics.


The jupyter notebook main.ipynb is used for running the codes in functions.py.
The jupyter notebook test.ipynb is used for benchmarking the regression functions.

The folder /figures includes all the plots that is included in the written report, with addition to some extra figures.
The file SRTM_data_Norway_1.tif is the digital terrain dataset.
The PDF file Project1_AT_VST_SMT.pdf is the written report.

