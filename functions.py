## Functions
import numpy as np
from scipy.stats import t
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import scipy.linalg as scl


''' Franke Function and Design Matrix '''
def FrankeFunction(x, y, noise_level=0):
    '''
    Usage: Computes the Franke function with user defined spatial paramters and noise
    Input:  x,y = meshgrid of spatial vectors
            noise_level = scalar value
    Output: Franke function = matrix of size [x]
    '''
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    noise = noise_level*np.random.randn(len(x),len(y))
    return term1 + term2 + term3 + term4 + noise


def DesignDesign(x, y, power,ravel=False):
    '''
    This function employs the underlying pattern governing a design matrix
    on the form [1,x,y,x**2,x*y,y**2,x**3,(x**2)*y,x*(y**2),y**3 ....]

    x_power=[0,1,0,2,1,0,3,2,1,0,4,3,2,1,0,...,n,n-1,...,1,0]
    y_power=[0,0,1,0,1,2,0,1,2,3,0,1,2,3,4,...,0,1,...,n-1,n]

    input:  x,y = 1D spatial vectors, or raveled mesh if ravel=True
            power = polynomial degree
            ravel = False (input vectors will not be meshed and raveled)
                  = True (input vecotors wil be kept as is)
    output: DesignMatrix = The design matrix
    '''

    concat_x   = np.array([0,0])
    concat_y   = np.array([0,0])


    for i in range(power):
        toconcat_x = np.arange(i+1,-1,-1)
        toconcat_y = np.arange(0,i+2,1)
        concat_x   = np.concatenate((concat_x,toconcat_x))
        concat_y   = np.concatenate((concat_y,toconcat_y))

    concat_x     = concat_x[1:len(concat_x)]
    concat_y     = concat_y[1:len(concat_y)]

    if ravel:
        X = x
        Y = y
    else:
        X,Y          = np.meshgrid(x,y)
        X            = np.ravel(X)
        Y            = np.ravel(Y)
    DesignMatrix = np.empty((len(X),len(concat_x)))
    for i in range(len(concat_x)):
        DesignMatrix[:,i]   = (X**concat_x[i])*(Y**concat_y[i])
    return DesignMatrix


''' Regression Algorithms'''
def OridinaryLeastSquares(design, data, test):
    '''
    Usage: Performs OLS  regression employing matrix inversion
    input:  design = user defined design matrix
            data = training data
            test = test design matrix (can be equal to design if no splitting)
    output: beta = beta parameters
            pred = prediction
    '''
    inverse_term   = np.linalg.inv(design.T.dot(design))
    beta           = inverse_term.dot(design.T).dot(data)
    pred           = test @ beta
    return beta, pred

def ols_svd(design, data, test):
    '''
    Usage: Performs OLS regression employing SVD
    input:  design = user defined design matrix
            data = training data
            test = test design matrix (can be equal to design if no splitting)
    output: beta = beta parameters
            pred = prediction
    '''
    u, s, v = np.linalg.svd(design)
    beta = v.T @ scl.pinv(scl.diagsvd(s, u.shape[0], v.shape[0])) @ u.T @ data
    return beta, test @ beta

def RidgeRegression(design, data, test, _lambda=0):
    '''
    Usage: Performs Ridge regression employing matrix inversion
    input:  design = user defined design matrix
            data = training data
            test = test design matrix (can be equal to design if no splitting)
            _lambda = hyperparameter/penalty paramter/tuning parameter
    output: beta = beta parameters
            pred = prediction
    '''
    inverse_term   = np.linalg.inv(design.T.dot(design)+ _lambda*np.eye((design.shape[1])))
    beta           = inverse_term.dot(design.T).dot(data)
    pred           = test @ beta
    return beta, pred

''' Error Metrics'''

def MSE(y, ytilde):
    '''
    Usage: calculates the Mean Squared Error
    Input:  y = observed data
            ytilde = predicted data
    output: Mean Squared Error
    '''
    return (np.sum((y-ytilde)**2))/y.size


def R2Score(y, ytilde):
    '''
    Usage: calculates the R2-score
    Input:  y = observed data
            ytilde = predicted data
    output: R2score
    '''
    return 1 - np.sum((y - ytilde) ** 2) / np.sum((y - np.mean(ytilde)) ** 2)


def confidence_interval(design, sigma, confidence, _lambda=0):
    '''
    Usage: calculates the confidence interval for OLS and Lasso if noise is known
    Input:  design = user defined design matrix
            sigma = a-priori known noise level
            confidence = 0.95 = 95 perecent confidence level
            _lambda = if lambda = 0, calculates confidence for OLS
                      if lambda ~=0, calculates confidence for Ridge
    output: Confidence interval for all beta values.
    '''
    if _lambda != 0:
        I=np.eye(design.shape[1])
        inverse_term   = np.linalg.inv(design.T.dot(design) + _lambda*I)
        variance_mat   = sigma**2*(inverse_term)*(design.T.dot(design))*(inverse_term)
    else:
        inverse_term   = np.linalg.inv(design.T.dot(design))
        variance_mat   = inverse_term*sigma**2

    standard_dev   = np.sqrt(np.diag(variance_mat))
    return standard_dev*norm.ppf(confidence+(1-confidence)/2)


''' Resampling methods'''

def reshaper(k, data):
    '''
    Usage: Manages the data for k_fold_cv
    Input: k = number of folds
           data = shuffled input data or input design matrix
    output: Splitted data
    '''
    output = []
    j = int(np.ceil(len(data)/k))
    for i in range(k):
        if i<k:
            output.append(data[i*j:(i+1)*j])
        else:
            output.append(data[i*j:])
    return np.asarray(output)


def k_fold_cv(k, indata, indesign, predictor, _lambda=0, shuffle=False):

    '''
    Usage: k-fold cross validation employing either RidgeRegression, OridinaryLeastSquares or ols_svd
    Input: k = number of folds
           indata = datapoints
           indesign = user defined design matrix
           predictor = RidgeRegression, OridinaryLeastSquares or ols_svd
           _lambda = hyperparameter/penalty paramter/tuning parameter for RidgeRegression
           shuffle = False, input data will not be shuffled
                     True, input data will be shuffled
    output: r2_out/k = averaged out sample R2-score
            mse_out/k = averaged out sample MSE
            r2_in/k = averaged in sample R2-Score
            mse_in/k = averaged in sample MSE
    '''
    mask = np.arange(indata.shape[0])
    if shuffle:
        np.random.shuffle(mask)
    data = reshaper(k, indata[mask])
    design = reshaper(k, indesign[mask])
    r2_out = 0
    r2_in = 0
    mse_out = 0
    mse_in = 0
    bias = 0
    variance = 0
    for i in range(k):
        train_design = design[np.arange(len(design))!=i]      # Featch all but the i-th element
        train_design = np.concatenate(train_design,axis=0)
        train_data   = data[np.arange(len(data))!=i]
        train_data   = np.concatenate(train_data,axis=0)
        test_design  = design[i]
        test_data    = data[i]

        if _lambda != 0:
            beta, pred = predictor(train_design, train_data, test_design, _lambda)
        else:
            beta, pred = predictor(train_design, train_data, test_design)

        r2_out += R2Score(test_data, pred)
        r2_in +=R2Score(train_data,train_design @ beta)
        mse_out += MSE(test_data, pred)
        mse_in += MSE(train_data,train_design @ beta)

    return r2_out/k, mse_out/k, r2_in/k, mse_in/k





def N_bootstraps(data, design,predictor,n,_lambda=0,test_size=0.2):
    '''
    Usage: Bootstrap resampling for bias-varaince tradeoff analysis.
    Input: data = input data
           design = user defined design matrix
           predictor = RidgeRegression, OridinaryLeastSquares or ols_svd
           n = number of bootstraps
           _lambda = hyperparameter/penalty paramter/tuning parameter for RidgeRegression
           test_size = amount of input data held out of bootstrapping.
    output: mse = MSE averaged over n bootstraps
            bias = bias averaged over n bootstraps
            variance = variance averaged over n bootstraps
    '''
    design_train, design_test, data_train, data_test = train_test_split(design, data, test_size=test_size)
    prediction = np.empty((data_test.shape[0], n))
    data_test=data_test.reshape(data_test.shape[0],1)
    if _lambda != 0:

        for i in range(n):
            design_resamp,data_resamp = bootstrap(design_train, data_train, i)
            beta, prediction[:,i] = predictor(design_resamp,data_resamp, design_test, _lambda)
    else:
        for i in range(n):
            design_resamp,data_resamp = bootstrap(design_train, data_train, i)
            beta, prediction[:,i] = predictor(design_resamp,data_resamp, design_test)

    mse = (np.mean(np.mean((data_test - prediction) ** 2, axis=1, keepdims=True)))
    bias = np.mean((data_test - np.mean(prediction, axis=1, keepdims=True)) ** 2)
    variance = np.mean(np.var(prediction, axis=1, keepdims=True))

    return mse, bias, variance



def bootstrap(design, data, random_state):
    '''
    Usage: Single bootstrap resampling
    Input: design = user defined design matrix
           data = input data
    output: design_subset = bootstrap resampled design matrix
            data_subset = bootstrap resampled data
    '''


    # For random randint
    rgen = np.random.RandomState(random_state)

    nrows, ncols = np.shape(design)

    selected_rows = np.random.randint(
        low=0, high=nrows, size=nrows
    )

    data_subset = data[selected_rows]
    design_subset = design[selected_rows, :]

    return design_subset, data_subset
