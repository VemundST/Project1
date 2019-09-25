## Functions
import numpy as np


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


def OridinaryLeastSquares_SVD(design, data, test):
    U,_sigma,V     = np.linalg.svd(design.T.dot(design))
    inverse_term   = 23
    beta           = inverse_term.dot(design.T).dot(data)
    pred           = test @ beta
    return beta, pred


def RidgeRegression(design, data, test, _lambda=0):
    inverse_term   = np.linalg.inv(design.T.dot(design)+ _lambda*np.eye((design.shape[1])))
    beta           = inverse_term.dot(design.T).dot(data)
    pred           = test @ beta
    return beta, pred


def VarianceBeta(design, _lambda=0):
    return np.diag(np.linalg.inv(design.T.dot(design) + _lambda*np.eye((design.shape[1]))))


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


def k_fold_cv(k, indata, indesign, predictor, _lambda=0, shuffle=False):
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
        tmp_design = design[np.arange(len(design))!=i]      # Featch all but the i-th element
        #tmp_design = tmp_design.reshape(tmp_design.shape[0]*tmp_design.shape[1], tmp_design.shape[2]) #reshape from 3D to 2D matrix
        tmp_design=np.concatenate(tmp_design,axis=0)
        tmp_data = data[np.arange(len(data))!=i]
        #tmp_data = tmp_data.reshape(tmp_data.shape[0]*tmp_data.shape[1])    # reshape from 2D to 1D
        tmp_data = np.concatenate(tmp_data,axis=0)
        if _lambda != 0:
            beta, pred = predictor(tmp_design, tmp_data, design[i], _lambda)
        else:
            beta, pred = predictor(tmp_design, tmp_data, design[i])
        r2_out += R2Score(data[i], pred)
        r2_in +=R2Score(tmp_data,tmp_design @ beta)
        mse_out += MSE(data[i], pred)
        mse_in += MSE(tmp_data,tmp_design @ beta)

        bias += np.mean((data[i]-np.mean(pred))**2)
        variance += np.mean((pred-np.mean(pred))**2)

    return r2_out/k, mse_out/k, r2_in/k, mse_in/k, bias/k, variance/k
