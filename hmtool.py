#########################################
###date 13/10/18
###It is going to be up to date at:
###< https://github.com/LeonardoLeano333 >
#########################################
### change lsm to lsm_

import numpy as np

# it is not that important
def w_average(x,sx):
    '''calculate the weighted average

    x:np.array # values
    sx:np.array # associeted deviation
    '''
    lx= len(x)    
    p = np.array([1/(sx[i]*sx[i]) for i in range(lx)])
    w_avg = np.dot(x,p)/sum(p)
    return w_avg

#  it is not that important
def s_w_average(sx):
    lx=len(sx)
    p = np.array([1/(sx[i]*sx[i]) for i in range(lx)])    
    s_w_avg = np.sqrt(1/(sum(p)))
    return s_w_avg

#####################################
### chi-square calculated by data and estimated data y
#####################################
#it is not that important
def chi2_calculator(data,sdata,y):
    '''chi-square calculated by data and estimated data y

    data:np.array
    sdata:np.array # data desviations, same lenght of data
    y:np.array # estimated data, same lenght of data
    '''
    delta = (data-y)/sdata 
    chi2  = sum(delta**2)
    return chi2

#####################################
###slope fit by least square method
#####################################
def lsm_slope_fit(x,y,sy=[0]):
    '''slope fit by least square method
    
    problem definition: y = ax
    
    x:np.array 
    y:np.array # data same length of x 
    sy:np.array # same length of x
    return a,sigma_a
    
    you can use it for standard desviation by linear fit

    ### example1: lsm_linear_fit(x,y,sy)
    y = np.array([ 0.5*i +np.random.rand(1)[0] for i in range(0,100)])
    sy = np.ones(100)
    x = np.arange(0,100)
    A,cov_matrix,_ = lsm_linear_fit(x,y,sy)
    
    ### example2: lsm_linear_fit(x,y)
    y = np.array([ 0.5*i +np.random.rand(1)[0] for i in range(0,100)])
    x = np.arange(0,100)
    A,cov_matrix,sy_estimate = lsm_linear_fit(x,y)
    
    ############
    ###coments for developping
    # id you dont know the deviation of the data you have to do it recursively in order to estimate it

    
    '''
    lx = len(x)
    if len(sy) == lx: 
        b = sum([y[i]*x[i]/(sy[i]*sy[i]) for i in range(lx) ])
        m = sum([x[i]*x[i]/(sy[i]*sy[i]) for i in range(lx) ])
        a = b/m
        sy_estimate =0
    
    else:
        print('Warning!!! The program is estimating the data deviation')
        sy_estimate= 1        
        for i in range(10):
            sy = sy_estimate*np.ones(lx)                
            b = sum([y[i]*x[i]/(sy[i]*sy[i]) for i in range(lx) ])
            m = sum([x[i]*x[i]/(sy[i]*sy[i]) for i in range(lx) ])
            a = b/m
            f_i = a*x            
            sy_estimate = np.sqrt(sum((y-f_i)**2)/(lx-1))                
        print('\n',str(sy_estimate),'\n')
    sigma_a = 1/m        

    return a, sigma_a, sy_estimate

def lsm_linear_fit(x,y,sy=[0]):
    ''' linear fit by least square method

    problem definition: y = ax +b
    
    #parameters
    x:np.array 
    y:np.array; data same length of x 
    sy = np.array; same length of x
    
    return A,cov_matrix
    
    A = parameters
    A[0] = a
    A[1] = b
    
    return A,cov_matrix,sy_estimate
    cov_matrix = covariance matrix

    ###example1: lsm_linear_fit(x,y,sy)
    y = np.array([ 0.5*i+7 +np.random.rand(1)[0] for i in range(0,100)])
    sy = np.ones(100)
    x = np.arange(0,100)
    A,cov_matrix,_ = lsm_linear_fit(x,y,sy)

    ###example2: lsm_linear_fit(x,y)
    #y = np.array([ 0.5*i+7 +np.random.rand(1)[0] for i in range(0,100)])
    #x = np.arange(0,100)
    #A,cov_matrix,sy_estimate = lsm_linear_fit(x,y)


    '''
    lx = len(x)
    if len(sy) == lx:    
        #linear problem definition
        B = [0,0]               #
        M = [[0,0],[0,0]]       #design matrix
        
        M[0][0] = sum([x[i]*x[i]/(sy[i]*sy[i]) for i in range(lx) ])
        M[1][0] = sum([x[i]/(sy[i]*sy[i])   for i in range(lx)])
        M[0][1] = M[1][0]
        M[1][1] = sum([1/(sy[i]*sy[i])   for i in range(lx)])
        
        B[0] = sum([y[i]*x[i]/(sy[i]*sy[i]) for i in range(lx) ])
        B[1] = sum([y[i]/(sy[i]*sy[i]) for i in range(lx) ])
        
        A = np.linalg.solve(M,B)
        if A[0] == 'nan':
            print('Warning!!! \n \n this problem could not have a solution')
            A = np.dot(np.linalg.pinv(M),B)
        cov_matrix = np.linalg.inv(M)
        sy_estimate=0
    else:
        print('Warning!!! The program is estimating the data deviation')
        sy_estimate= 1        
        for i in range(10):        
            sy = sy_estimate*np.ones(lx)                
            B = [0,0]               #
            M = [[0,0],[0,0]]       #design matrix
            
            M[0][0] = sum([x[i]*x[i]/(sy[i]*sy[i]) for i in range(lx) ])
            M[1][0] = sum([x[i]/(sy[i]*sy[i])   for i in range(lx)])
            M[0][1] = M[1][0]
            M[1][1] = sum([1/(sy[i]*sy[i])   for i in range(lx)])
            
            B[0] = sum([y[i]*x[i]/(sy[i]*sy[i]) for i in range(lx) ])
            B[1] = sum([y[i]/(sy[i]*sy[i]) for i in range(lx) ])
            
            A = np.linalg.solve(M,B)
            if A[0] == 'nan':
                print('Warning!!! \n \n this problem could not have a solution')
                A = np.dot(np.linalg.pinv(M),B)
            f_i = A[0]*x +A[1]            
            sy_estimate = np.sqrt(sum((y-f_i)**2)/(lx-2))            
        
        print('data deviation estimated:\n')
        print(str(sy_estimate)+'\n')
        cov_matrix = np.linalg.inv(M)
        
    #covariance matrix
    
    return A,cov_matrix,sy_estimate


#####################################
###exponential decai parameter estimator
#####################################

def exponential_decai_p_estimator(tt,yy,sy,pp,n_gauss=0):
    '''parameter estimator for exponential decay model
    
    #model yy = I0*exp(-p*tt)+BG
    
    entries:
    
    tt=array
    yy=array same length tt
    sy=deviation of yy
    pp = trie this pp parameters they need to be positives
    
    return
    [p,I0,BG],QQ,cov_matrix,chi2_refined
    [p,I0,BG] = fitted parameters
    QQ = chi2 for each p tried
    cov_matrix = covariance matrix of parameters
    chi2_refined = chi2 of the fitted curve
    
    #####################################
    ### example:
    import matplotlib.pyplot as plt
    tt = np.arange(0,60)
    p = 0.206
    BG = 40
    I0=1000
    pp = [i/1000 for i in range(1,1001)]
    
    sy = np.sqrt(I0*np.exp(-p*tt) + BG)
    yy = np.array([I0*np.exp(-p*tt[i]) +BG + np.random.normal(0,sy[i]) for i in range(len(tt))])
    params,chi2,cov_matrix,chi2_refined = exponential_decai_p_estimator(tt,yy,sy,pp,n_gauss=10)
    [p,I0,BG]=params
    
    plt.figure(1)
    plt.errorbar(tt,yy,yerr=sy, marker='o',linestyle=' ')
    plt.plot(tt,I0*np.exp(-p*tt) +BG )
    
    residue = [(I0*np.exp(-p*tt[i]) + BG - yy[i]) for i in range(len(yy))]
    
    plt.figure(2)
    plt.errorbar(tt,residue,yerr=sy,marker='o',linestyle=' ')
    plt.show()
    #####################################

    '''
    QQ = np.zeros(len(pp))
    #looking p,I0,BG first aproximation that minimizes chi2
    for iter_p in range(len(pp)):
        p=pp[iter_p]
        xx = np.exp(-p*tt)
        #linear fit
        parameters, cov_matrix,_ = lsm_linear_fit(xx,yy,sy)
        #first guess of the parameters
        I0=parameters[0]
        BG=parameters[1]
        QQ[iter_p] = sum([(yy[i]-I0*np.exp(-p*tt[i])-BG)**2/(sy[i]*sy[i]) for i in range(len(tt))])
        
    #calculation of the minimum chi2 properties
    p = pp[np.argmin(QQ)]
    xx = np.exp(-p*tt)
    [I0,BG], cov_matrix,_ = lsm_linear_fit(xx,yy,sy)
    params=[p,I0,BG]
    min_chi2 = sum([(yy[i]-I0*np.exp(-p*tt[i])-BG)**2/(sy[i]*sy[i]) for i in range(len(tt))])
    #cov_matrix calculation for [p,I0,BG] by Gauss method,
    #admiting the data are not correlated
    x_matrix = np.array([[-I0*tt[i]*np.exp(-p*tt[i]),\
                          np.exp(-p*tt[i]),\
                            1] for i in range(len(tt))])
    V = (sy*sy)*np.eye(len(sy))
    inverse_V = np.linalg.inv(V)
    planning_matrix = np.dot(x_matrix.T,np.dot(inverse_V,x_matrix))
    #cov_matrix = (Xt V-1 X)-1
    cov_matrix = np.linalg.inv(planning_matrix)
        
    #Gauss method correction
    chi2_refined = min_chi2
    for i_correction in range(n_gauss):
        yy_line = yy -I0*np.exp(-p*tt)-BG
        #d_params = (Xt V-1 X)-1 Xt V-1 yy_refined
        d_params = np.dot(cov_matrix,np.dot(x_matrix.T,np.dot(inverse_V,yy_line)))
        params= params+d_params
        [p,I0,BG]=params
        chi2_refined = sum([(yy[i]-I0*np.exp(-p*tt[i])-BG)**2/(sy[i]*sy[i]) for i in range(len(tt))])
    
    #calculation of cov_matrix from refined parameters 
    x_matrix = np.array([[-I0*tt[i]*np.exp(-p*tt[i]),np.exp(-p*tt[i]),1] for i in range(len(tt))])
    V = (sy*sy)*np.eye(len(sy))
    inverse_V = np.linalg.inv(V)
    planning_matrix = np.dot(x_matrix.T,np.dot(inverse_V,x_matrix))
    #cov_matrix = (Xt V-1 X)-1
    cov_matrix = np.linalg.inv(planning_matrix)
    
    ###SSS Verificar aqui : tem uma redundancia na rotina SSS###
    ### Acho que esse nao leva em conta a correcao de gauss
    #DESCRIPTION: this method use a mapping in the parameter p for the first guess in p, and make an correction with the gauss method.
    #Obs.: it is admited that the data are not correlated
    #WARNING: Look at the p vs chi_square(QQ) map in order to evaluate the convergence of gauss method, it should be a parabola in the minimum value of chi_square.
    #REFERENCE:
    
    return [p,I0,BG],QQ,cov_matrix,chi2_refined

#####################################
### least square method 3 parameter estimator
#####################################
###SSS Verifica aqui SSS###
### mudar de nome essa funcao para ingles; 
### Lembrar que ela esta sendo usada no outro metodo
### double_exponential_decai_p_estimator
#DESCRIPTION:
#mudar de nome


def lsm_linear_fit_3d(yy,sy,xx1,xx2):
    '''parameter estimator for 2 indenpendent values

    model of data entry:
       yy= a0+a1*x1+a2*x2
    entries:
    yy  = numpy array 
    sy  = numpy array, same length yy
    xx1 = numpy array, same length yy
    xx2 = numpy array, same length yy
    
    return a,cov_matrix,chi2
    a = numpy array with parameters
    cov_matrix = flout covariation matrix
    chi2 = flout chisquare

    #####################################
    example:
    import matplotlib.pyplot as plt
    tt = np.arange(0,60)
    p1 = 0.2
    p2 = 0.1
    BG = 40
    I10 = 1000
    I20 = 200
    pp1 = [i/1000 for i in range(1,1001)]
    
    sy = np.sqrt(I10*np.exp(-p1*tt)+ I20*np.exp(-p2*tt) + BG)
    yy = np.array([I10*np.exp(-p1*tt[i])+ I20*np.exp(-p2*tt[i]) + BG + np.random.normal(0,sy[i]) for i in range(len(tt))])
    
    pp1 = [i/10 for i in range(1,11)]
    pp2 = [i/10 for i in range(1,11)]
    QQ = np.zeros([len(pp1),len(pp2)])
    
    p1 = pp1[iter_1]
    p2 = pp2[iter_2] 
    xx1 = np.exp(-p1*tt) 
    xx2 = np.exp(-p1*tt)
    a,cov_matrix,chi2=lsm_linear_fit_3d(yy,sy,xx1,xx2)        
    [BG,I01,I02] = a
    print([BG,I01,I02])
    print(chi2)
    
    plt.figure(1)
    plt.errorbar(tt,yy,yerr=sy,marker='o',linestyle=' ')
    plt.plot(tt,I10*np.exp(-p1*tt)+ I20*np.exp(-p2*tt) + BG)      
    
    QQ[iter_1][iter_2] = chi2
    print(np.argmin()) 
    residue = (yy - I10*np.exp(-p1*tt) - I20*np.exp(-p2*tt) - BG  )
    plt.figure(2)
    plt.errorbar(tt,residue,yerr=sy,marker='o',linestyle=' ')
    plt.plot([0,60],[0,0],color='k')
    #####################################

    '''
    

    ly = len(yy)
    D = np.zeros(3)     #vector for solving lsm
    M = np.zeros([3,3]) #design matrix
    #parameters

    #D = vector for solving
    D[0] = np.sum([yy[i]/(sy[i]*sy[i]) for i in range(ly)])
    D[1] = np.sum([yy[i]*xx1[i]/(sy[i]*sy[i]) for i in range(ly) ])
    D[2] = np.sum([yy[i]*xx2[i]/(sy[i]*sy[i]) for i in range(ly) ])

    #M = design matrix
    M[0][0] = np.sum([1/(sy[i]*sy[i]) for i in range(ly)])
    M[0][1] = np.sum([xx1[i]/(sy[i]*sy[i]) for i in range(ly)])
    M[0][2] = np.sum([xx2[i]/(sy[i]*sy[i]) for i in range(ly)])
    M[1][1] = np.sum([xx1[i]*xx1[i]/(sy[i]*sy[i]) for i in range(ly)])
    M[1][2] = np.sum([xx1[i]*xx2[i]/(sy[i]*sy[i]) for i in range(ly)])
    M[2][2] = np.sum([xx2[i]*xx2[i]/(sy[i]*sy[i]) for i in range(ly)])
    M[1][0] = M[0][1]
    M[2][0] = M[0][2]
    M[2][1] = M[1][2]
    
    #calculating the parameters end covariation matrix        
    if np.linalg.det(M)!= 0:    
        a = np.linalg.solve(M,D)
        cov_matrix = np.linalg.inv(M)
        chi2 = sum([(yy[i]-a[0]-a[1]*xx1[i]-a[2]*xx2[i])**2/(sy[i]*sy[i]) for i in range(len(xx1))])
    else:
        #print('DEU RUIM')
        a= np.array([0,0,0])
        cov_matrix= np.zeros([3,3])
        #chi2 = 0        
        #print(M)        
        chi2 = 10*ly
    #verificar isso com calculo do minimo chi2
    
    
    return a,cov_matrix,chi2


#########################################
### double exponential decai parameter estimator
#########################################
def double_exponential_decai_p_estimator(tt,yy,sy,pp1,pp2,n_gauss=0):
    ''' estimate 2 exponential decay
    
    obs: It is admited that the data entries are not correlated

    data model:
    yy = BG+I01*exp(-p1*t)+I02*exp(-p2*t)
    
    entries:
    tt = numpy array
    yy = numpy array, same length tt
    sy = numpy array, same length tt
    pp1 = numpy array, tries this p1 values 
    pp2 = numpy array, tries this p2 values
    n_gauss = int, iterate gauss method n_gauss times
    return:
    params = [BG,I01,I02,p1,p2] = vector, stimated parameters 
    cov_matrix = 2D numpy array, covariation matrix
    chi2 = float, chi-square
    residue = numpy array, vector to plot the residues
    QQ = 2D numpy array, length(pp1) X length(pp2), to plot 3D the guessed region

   ###example:
    #########################################
    import matplotlib.pyplot as plt
    tt = np.arange(0,60)
    p1 = 0.3
    I10 = 1000
    p2 = 0.1
    I20 = 200
    BG = 40
    
    sy = np.sqrt(I10*np.exp(-p1*tt)+ I20*np.exp(-p2*tt) + BG)
    yy = np.array([I10*np.exp(-p1*tt[i])+ I20*np.exp(-p2*tt[i]) + BG + np.random.normal(0,sy[i]) for i in range(len(tt))])
    lpp = 100
    pp1 = [i/lpp for i in range(1,lpp+1)]
    pp2 = [i/lpp for i in range(1,lpp+1)]
    
    a,cov_matrix,chi2,residue,QQ = double_exponential_decai_p_estimator(tt,yy,sy,pp1,pp2)
    [BG,I10,I20,p1,p2] = a
    
    print(a)
    print(cov_matrix)
    print(chi2)
    
    plt.figure(1)
    plt.title('random data and fit')
    plt.errorbar(tt,yy,yerr=sy,marker='o',linestyle=' ',label='random data',color='b')
    plt.plot(tt,a[0]+a[1]*np.exp(-p1*tt)+ a[2]*np.exp(-p2*tt),label='fit',color='r')      
    plt.legend()
    
    plt.figure(2)
    plt.title('residue')
    plt.errorbar(tt,residue,yerr=sy,marker='o',linestyle=' ')
    plt.plot([0,60],[0,0],color='k')
    
    
    plt.figure(2)
    plt.title('reduced residue')
    plt.errorbar(tt,residue,yerr=sy,marker='o',linestyle=' ',color='blue')
    plt.plot([0,60],[0,0],color='k')
    
    from mpl_toolkits.mplot3d import axes3d
    
    fig = plt.figure(3)
    
    X = pp1
    Y = pp2
    X, Y = np.meshgrid(X, Y)
    Z = np.array(QQ)
    Z[Z>chi2_refined+3*np.sqrt(chi2_refined)] = chi2_refined+2*np.sqrt(chi2_refined)
    V = np.array([chi2_refined+i*np.sqrt(chi2_refined) for i in range(0,10)])
    
    cp = plt.contourf(X, Y, Z,cmap='jet')
    cp.levels = [chi2_refined+i**2 for i in range(0,5)]
    plt.colorbar(cp)
    plt.title('Contours Plot')
    plt.xlim(p1-3*np.sqrt(cov_matrix[3][3]),p1+3*np.sqrt(cov_matrix[3][3]))
    plt.ylim(p2-3*np.sqrt(cov_matrix[4][4]),p2+3*np.sqrt(cov_matrix[4][4])) 
    plt.xlabel('p1')
    plt.ylabel('p2')
    plt.show()
    #########################################

    
    '''
    QQ = np.zeros([len(pp1),len(pp2)])
    for iter_1 in range(len(pp1)):
        for iter_2 in range(len(pp2)):
            p1 = pp1[iter_1]
            p2 = pp2[iter_2] 
            xx1 = np.exp(-p1*tt) 
            xx2 = np.exp(-p2*tt)
            a,cov_matrix,chi2=lsm_linear_fit_3d(yy,sy,xx1,xx2)        
            QQ[iter_1][iter_2]=chi2

    best = np.unravel_index(QQ.argmin(),QQ.shape)
    
    p1 = pp1[best[0]]
    p2 = pp2[best[1]] 
    xx1 = np.exp(-p1*tt) 
    xx2 = np.exp(-p2*tt)
    a,cov_matrix,chi2=lsm_linear_fit_3d(yy,sy,xx1,xx2)   
    [BG,I01,I02] = a
    params = [BG,I01,I02,p1,p2]
    print(params)      
    #gauss method correction
    x_matrix = np.array([[ 1,\
                          np.exp(-p1*tt[i]),\
                          np.exp(-p2*tt[i]),\
                          -I01*tt[i]*np.exp(-p1*tt[i]),\
                          -I02*tt[i]*np.exp(-p2*tt[i])]\
                          for i in range(len(tt)) ])

    #data not correlated
    V = (sy*sy)*np.eye(len(sy))
    inverse_V = np.linalg.pinv(V)
    planning_matrix = np.dot(x_matrix.T,np.dot(inverse_V,x_matrix))
    cov_matrix = np.linalg.pinv(planning_matrix)
    #Gauss method correction fails i dont know why
    #problably it is because of the parameters mapped     
    #for i_correction in range(n_gauss):
    #    yy_line = yy - BG - I01*np.exp(-p1*tt) - I20*np.exp(-p2*tt)   
    #    #d_params = (Xt V-1 X)-1 Xt V-1 yy_refined
    #    d_params = np.dot(cov_matrix,np.dot(x_matrix.T,np.dot(inverse_V,yy_line)))
    #    params= params+d_params
    #    [BG,I01,I02,p1,p2] = params
    #    print(params)    
    #    chi2_refined = sum([(yy - BG - I10*np.exp(-p1*tt) - I20*np.exp(-p2*tt))**2/(sy[i]*sy[i]) \
    #                     for i in range(len(tt))])
    
    residue = ( yy - BG - I01*np.exp(-p1*tt) - I02*np.exp(-p2*tt) )
    print('\n\n\n')
    return [BG,I01,I02,p1,p2],cov_matrix,chi2,residue,QQ

#########################################
###derivate with precision
#########################################
def derivate(func,par,x,dx):
    '''derivate a function with triangle method
    #func= model function
    #par = function parameters
    #x   = derivate point 
    #dx  = precision
    '''
    df = (func(par,x+dx)-func(par,x-dx))/(2*dx)
    return df



### COMENT
#########################################
###derivate in parameter with precision dpar is a vector to derivate
#########################################
def derivate_in_par(func,x,par,dpar):
    '''derivate a function in parameters
    #func= model function
    #par = function parameters
    #x   = derivate point 
    #dx  = precision
    '''
    df = (func(x,par+dpar)-func(x,par-dpar))/(2*np.linalg.norm(dpar))
    return df
    
##############################
###exponential decay stretching
##############################
def decay_stretching(x,par):
    ''' y = (e**(p*x))**b 
    
    par = np.array([1,1])
    x = np.arange(0,121)
    y = decay_stretching(x,par)
    
    plt.plot(x,y)
    plt.show()
    
    dpar = np.array([0,0,0.0001])
    
    dy = derivate_in_par(decay_stretching,x,par,dpar)
    
    plt.plot(x,dy)
    plt.show()
    ##############################
    
    '''
    
    p = par[0]
    b= par[1]
    y = np.exp(   - np.power((p*x),b) )
    y = np.array(y)
    return y

##############################
###General Two Nonlinear Parameter Estimator - Mapping & Gauss ###
##############################
def two_coupled_nonlinear_parameter_estimator(func,xx,yy,sy,pp1,pp2,delta,n_gauss=0):
    '''estimate non linear parameter in a function with two nonlinear parameters
    
    model:
        y = A*f(x;a,b)

    func: function with par where para is 2D
    pp1: np.array for mapping the parameter 
    pp2: np.array for mapping the parameter
    
    #######################
    ####example:
    I0 = 1000
    p=0.555
    b=0.455
    F = 45
    par = np.array([p,b])
    
    xx = np.arange(0,121)
    sy = np.sqrt(I0*decay_stretching(xx,par) + F)
    yy = np.array([I0*decay_stretching(xx[i],par) + F + np.random.normal(0,sy[i]) for i in range(len(xx))])
    
    
    func = decay_stretching
    delta= 0.0001
    ll=100
    pp1=np.array([i/ll for i in range(1,ll)])
    pp2=np.array([i/ll for i in range(1,ll)])
    
    par_estimated,cov_matrix,chi2_min,residue,QQ = two_coupled_nonlinear_parameter_estimator(func,xx,yy,sy,pp1,pp2,delta,n_gauss=0)
    
    
    plt.figure(1)
    par = [par_estimated[0],par_estimated[1]]
    A = [par_estimated[2],par_estimated[3]]
    plt.plot(xx,yy,label='dados',color='b')
    plt.plot(xx,A[0]*decay_stretching(xx,par)+A[1],label='fit',color='r')
    plt.xlabel('tempo(s)')
    plt.ylabel('sinal OSL(C/s)')
    plt.show()
    
    
    plt.figure(2)
    plt.plot(residue/sy)
    plt.show()
    #contour plot
    
    plt.figure(3)
    x = pp1
    y = pp2
    X, Y = np.meshgrid(x, y)
    Z = QQ
    Z_min = np.min(QQ)
    V = np.array([Z_min+i**2 for i in range(10)])
    
    
    p1 = par_estimated[0]
    p2 = par_estimated[1]
    sp1 = np.sqrt(cov_matrix[0][0])
    sp2 = np.sqrt(cov_matrix[1][1])
    plt.figure(4)
    CS = plt.contour(X, Y, Z, V)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.plot([p2],[p1],label='Min chisquare'+str(int(Z_min)),marker='o'\
    ,linestyle=' ',color='r',markersize=3)
    plt.xlabel('$B$')    
    plt.ylabel('$p(s^{-1})$')    
    plt.xlim(p2-3*sp2,p2+3*sp2)
    plt.ylim(p1-3*sp1,p1+3*sp1)
    
    plt.legend(loc='best')
    plt.show()
    #######################
    '''
    par =[0,0]
    
    
    QQ = np.zeros([len(pp1),len(pp2)])
    for iter_1 in range(len(pp1)):
        for iter_2 in range(len(pp2)):
            par[0] = pp1[iter_1]
            par[1] = pp2[iter_2] 
            xx1 = func(xx,par)
            A,cov_matrix,_=lsm_linear_fit(xx1,yy,sy)
            #plt.plot(xx,xx1)
            ff = A[0]*xx1+A[1]
            QQ[iter_1][iter_2]=chi2_calculator(yy,sy,ff)
            
    best = np.unravel_index(QQ.argmin(),QQ.shape)
    
    p1 = pp1[best[0]]
    p2 = pp2[best[1]]
    par = [p1,p2]
    xx1 = decay_stretching(xx,par)
    A,cov_matrix,_=lsm_linear_fit(xx1,yy,sy)
    ff = A[0]*xx1+A[1]
    chi2_min = chi2_calculator(yy,sy,ff)
    par_nonlinear_estimated = np.array([p1,p2])
    ### preciso agora fazer a tiracao das propriedades
    ### straction of properties
    x_matrix = np.array([[ A[0]*derivate_in_par(func,xx[i],par_nonlinear_estimated,[delta,0]),\
                  A[0]*derivate_in_par(func,xx[i],par_nonlinear_estimated,[0,delta]),\
                  func(xx[i],par_nonlinear_estimated),\
                  1]for i in range(len(xx))])
    V = (sy*sy)*np.eye(len(sy))
    inverse_V = np.linalg.pinv(V)
    planning_matrix = np.dot(x_matrix.T,np.dot(inverse_V,x_matrix))
    cov_matrix = np.linalg.pinv(planning_matrix)

    par_estimated = [p1,p2,A[0],A[1]]
    residue = ( yy -( A[0]*func(xx,par)+A[1]) )
    
    return par_estimated,cov_matrix,chi2_min,residue,QQ


