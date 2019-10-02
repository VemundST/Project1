## Functions
import numpy as np
from scipy.stats import t
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import scipy.linalg as scl

def FrankeFunction(x, y, noise_level=0):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    noise = noise_level*np.random.randn(len(x),len(y))
    return term1 + term2 + term3 + term4 + noise


def OridinaryLeastSquares(design, data, test):
    inverse_term   = np.linalg.inv(design.T.dot(design))
    beta           = inverse_term.dot(design.T).dot(data)
    pred           = test @ beta
    return beta, pred

def ols_svd(design, data, test):
    u, s, v = scl.svd(design)
    beta = v.T @ scl.pinv(scl.diagsvd(s, u.shape[0], v.shape[0])) @ u.T @ z
    return beta, test @ beta


def RidgeRegression(design, data, test, _lambda=0):
    inverse_term   = np.linalg.inv(design.T.dot(design)+ _lambda*np.eye((design.shape[1])))
    beta           = inverse_term.dot(design.T).dot(data)
    pred           = test @ beta
    return beta, pred


def VarianceBeta(y, n):
    n = np.size(y)
    mean = (1/n ) * np.sum(y)
    return (1/n)* np.sum((y-mean)**2)


def MSE(y, ytilde):
    return (np.sum((y-ytilde)**2))/y.size


def R2Score(y, ytilde):
    return 1 - ((np.sum((y-ytilde)**2))/(np.sum((y-((np.sum(y))/y.size))**2)))


def MAE(y, ytilde):
    return (np.sum(np.abs(y-ytilde)))/y.size


def MSLE(y, ytilde):
    return (np.sum((np.log(1+y)  -  np.log(1+ytilde))**2))/y.size


def DesignDesign(x, y, power):
    '''
    This function employs the underlying pattern governing a design matrix
    on the form [1,x,y,x**2,x*y,y**2,x**3,(x**2)*y,x*(y**2),y**3 ....]

    x_power=[0,1,0,2,1,0,3,2,1,0,4,3,2,1,0,...,n,n-1,...,1,0]
    y_power=[0,0,1,0,1,2,0,1,2,3,0,1,2,3,4,...,0,1,...,n-1,n]
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

    X,Y          = np.meshgrid(x,y)
    X            = np.ravel(X)
    Y            = np.ravel(Y)
    DesignMatrix = np.empty((len(X),len(concat_x)))
    for i in range(len(concat_x)):
        DesignMatrix[:,i]   = (X**concat_x[i])*(Y**concat_y[i])

    #DesignMatrix = np.concatenate((np.ones((len(X),1)),DesignMatrix), axis = 1)
    return DesignMatrix


def reshaper(k, data):
    output = []
    j = int(np.ceil(len(data)/k))
    for i in range(k):
        if i<k:
            output.append(data[i*j:(i+1)*j])
        else:
            output.append(data[i*j:])
    return np.asarray(output)


def k_fold_cv(k, indata, indesign, predictor, _lambda=0, shuffle=False, scikit=False):
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

        if scikit:
            r2_out += r2_score(test_data, pred)
            r2_in += r2_score(train_data,train_design @ beta)
            mse_out += mean_squared_error(test_data, pred)
            mse_in += mean_squared_error(train_data,train_design @ beta)
        else:
            r2_out += R2Score(test_data, pred)
            r2_in +=R2Score(train_data,train_design @ beta)
            mse_out += MSE(test_data, pred)
            mse_in += MSE(train_data,train_design @ beta)


        #bias += np.mean((test_data-np.mean(pred))**2)
        #variance += np.mean((pred-np.mean(pred))**2)

    return r2_out/k, mse_out/k, r2_in/k, mse_in/k, #bias/k, variance/k


def confidence_interval(design, sigma, confidence, _lambda=0):
    inverse_term   = np.linalg.inv(design.T.dot(design))
    if _lambda != 0:
        I=np.eye(inverse_term.shape[0])
        variance_mat   = sigma**2*(inverse_term + _lambda*I)*(inverse_term)*np.transpose(inverse_term + _lambda*I)
    else:
        variance_mat   = inverse_term*sigma**2
    standard_dev   = np.sqrt(np.diag(variance_mat))
    return standard_dev*norm.ppf(confidence+(1-confidence)/2)

def confidence_interval_est_sigma(design, confidence, data, prediction, _lambda=0):
    '''
    As sigma is unknown it must be extracted from the data,
    using equation below (it has no number) 3.8 from Hastie et al - Elements of statistical learning
    '''
    sigma_sqrd = MSE(data,prediction)/(design.shape[0] - design.shape[1] - 1)

    inverse_term   = np.linalg.inv(design.T.dot(design))
    if _lambda != 0:
        I=np.eye(inverse_term.shape[0])
        variance_mat   = sigma_sqrd*(inverse_term + _lambda*I)*(inverse_term)*np.transpose(inverse_term + _lambda*I)
    else:
        variance_mat   = inverse_term*sigma_sqrd
    standard_dev   = np.sqrt(np.diag(variance_mat))
    return standard_dev*norm.ppf(confidence+(1-confidence)/2)


def N_bootstraps(data, model,predictor,n,_lambda=0,test_size=0.2):
    design_train, design_test, data_train, data_test = train_test_split(model, data, test_size=test_size)
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



def bootstrap(X, z, random_state):

    # For random randint
    rgen = np.random.RandomState(random_state)

    nrows, ncols = np.shape(X)

    selected_rows = np.random.randint(
        low=0, high=nrows, size=nrows
    )

    z_subset = z[selected_rows]
    X_subset = X[selected_rows, :]

    return X_subset, z_subset
