#author: Zhiqin Xu 许志钦
#email: xuzhiqin@sjtu.edu.cn
#2019-09-24
# coding: utf-8


import os
import matplotlib
matplotlib.use('Agg')   
import pickle
import time  
import shutil 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt   
#from BasicFunc import mySaveFig, mkdir

from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


Leftp=0.18
Bottomp=0.18
Widthp=0.88-Leftp
Heightp=0.9-Bottomp
pos=[Leftp,Bottomp,Widthp,Heightp]

def mkdir(fn):
    if not os.path.isdir(fn):
        os.mkdir(fn)
def mySaveFig(pltm, fntmp,fp=0,ax=0,isax=0,iseps=0,isShowPic=0):
    if isax==1:
        #pltm.legend(fontsize=18)
        # plt.title(y_name,fontsize=14)
#        ax.set_xlabel('step',fontsize=18)
#        ax.set_ylabel('loss',fontsize=18)
        pltm.rc('xtick',labelsize=18)
        pltm.rc('ytick',labelsize=18)
        ax.set_position(pos, which='both')
    fnm='%s.png'%(fntmp)
    pltm.savefig(fnm)
    if iseps:
        fnm='%s.eps'%(fntmp)
        pltm.savefig(fnm, format='eps', dpi=600)
    if fp!=0:
        fp.savefig("%s.pdf"%(fntmp), bbox_inches='tight')
    if isShowPic==1:
        pltm.show() 
    elif isShowPic==-1:
        return
    else:
        pltm.close()
        

R_variable={}  ### used for saved all parameters and data
 
### mkdir a folder to save all output
BaseDir = 'fitnd/'
subFolderName = '%s'%(int(np.absolute(np.random.normal([1])*100000))//int(1)) 
FolderName = '%s%s/'%(BaseDir,subFolderName)
mkdir(BaseDir)
mkdir(FolderName)
mkdir('%smodel/'%(FolderName))
R_variable['FolderName']=FolderName 
R_variable['input_dim']=2
R_variable['epsion']=0.1
def get_y_func(xs):
    tmp=0
    for ii in range(R_variable['input_dim']):
        tmp+=np.cos(4*xs[:,ii:ii+1])
    return tmp


R_variable['output_dim']=1
R_variable['ActFuc']=1   ###  0: ReLU; 1: Tanh; 3:sin;4: x**5,, 5: sigmoid  6 sigmoid derivate

R_variable['hidden_units']=[1500,1500,1500]
#R_variable['hidden_units']=[500,500,500]
R_variable['hidden_units']=[200,200,200]
### initialization standard deviation
R_variable['astddev']=np.sqrt(1/20) # for weight
R_variable['bstddev']=np.sqrt(1/20)# for bias terms2

R_variable['ASI']=0
R_variable['learning_rate']=1e-6
R_variable['learning_rateDecay']=2e-7

### setup for activation function
R_variable['seed']=0

plotepoch=500
R_variable['train_size']=1000;  ### training size
R_variable['batch_size']=R_variable['train_size'] # int(np.floor(R_variable['train_size'])) ### batch size
R_variable['test_size']=R_variable['train_size']  ### test size
R_variable['x_start']=-np.pi/2  #math.pi*3 ### start point of input
R_variable['x_end']=np.pi/2  #6.28/4 #math.pi*3  ### end point of input


R_variable['tol']=-1e10
R_variable['Total_Step']=600000  ### the training step. Set a big number, if it converges, can manually stop training 

R_variable['FolderName']=FolderName   ### folder for save images

print(R_variable) 
if R_variable['input_dim']==1:
    R_variable['test_inputs'] =np.reshape(np.linspace(R_variable['x_start'], R_variable['x_end'], num=R_variable['test_size'],
                                                      endpoint=True),[R_variable['test_size'],1])
    #n_size=R_variable['train_size']
    R_variable['train_inputs']=np.reshape(np.linspace(R_variable['x_start'], R_variable['x_end'], num=R_variable['train_size'],
                                                      endpoint=True),[R_variable['train_size'],1])
else:
    R_variable['test_inputs']=np.random.rand(R_variable['test_size'],R_variable['input_dim'])*(R_variable['x_end']-R_variable['x_start'])+R_variable['x_start']
    R_variable['train_inputs']=np.random.rand(R_variable['train_size'],R_variable['input_dim'])*(R_variable['x_end']-R_variable['x_start'])+R_variable['x_start']



test_inputs=R_variable['test_inputs']
train_inputs=R_variable['train_inputs']
R_variable['y_true_test']= get_y_func(test_inputs)
R_variable['y_true_train']=get_y_func(train_inputs)

#plt.figure()
#plt.plot(test_inputs,R_variable['y_true_test'])

t0=time.time() 

def add_layer2(x,input_dim = 1,output_dim = 1,isresnet=0,astddev=0.05,
               bstddev=0.05,ActFuc=0,seed=0, name_scope='hidden'):
    if not seed==0:
        tf.set_random_seed(seed)
    
    with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
        ua_w = tf.get_variable(name='ua_w', initializer=astddev)
        ua_b = tf.get_variable(name='ua_b', initializer=bstddev) 
        z=tf.matmul(x, ua_w) + ua_b
        
        
        if ActFuc==1:
            output_z = tf.nn.tanh(z)
            print('tanh')
        elif ActFuc==3:
            output_z = tf.sin(z)
            print('sin')
        elif ActFuc==0:
            output_z = tf.nn.relu(z)
            print('relu')
        elif ActFuc==4:
            output_z = z**50
            print('z**50')
        elif ActFuc==5:
            output_z = tf.nn.sigmoid(z)
            print('sigmoid')
            
        L2Wight= tf.nn.l2_loss(ua_w) 
        if isresnet and input_dim==output_dim:
            output_z=output_z+x
        return output_z,ua_w,ua_b,L2Wight

def getWini(hidden_units=[10,20,40],input_dim = 1,output_dim_final = 1,astddev=0.05,bstddev=0.05):
    
    hidden_num = len(hidden_units)
    #print(hidden_num)
    add_hidden = [input_dim] + hidden_units;
    
    w_Univ0=[]
    b_Univ0=[]
    
    for i in range(hidden_num):
        input_dim = add_hidden[i]
        output_dim = add_hidden[i+1]
        ua_w=np.float32(np.random.normal(loc=0.0,scale=astddev,size=[input_dim,output_dim]))
        ua_b=np.float32(np.random.normal(loc=0.0,scale=bstddev,size=[output_dim]))
        w_Univ0.append(ua_w)
        b_Univ0.append(ua_b)
    ua_w=np.float32(np.random.normal(loc=0.0,scale=astddev,size=[hidden_units[hidden_num-1], output_dim_final]))
    ua_b=np.float32(np.random.normal(loc=0.0,scale=bstddev,size=[output_dim_final]))
    w_Univ0.append(ua_w)
    b_Univ0.append(ua_b)
    return w_Univ0, b_Univ0

w_Univ0, b_Univ0=getWini(hidden_units=R_variable['hidden_units'],
                                         input_dim = R_variable['input_dim'],
                                         output_dim_final = R_variable['output_dim'],
                                         astddev=R_variable['astddev'],
                                         bstddev=R_variable['bstddev'])

def univAprox2(x0, hidden_units=[10,20,40],input_dim = 1,output_dim_final = 1,
               isresnet=0,astddev=0.05,bstddev=0.05,
               ActFuc=0,seed=0,ASI=1):
    if seed==0:
        seed=time.time()
    # The simple case is f: R -> R 
    hidden_num = len(hidden_units)
    #print(hidden_num)
    add_hidden = [input_dim] + hidden_units;
    
    w_Univ=[]
    b_Univ=[] 
    L2w_all=0
    
    w_Univ0, b_Univ0=getWini(hidden_units=hidden_units,input_dim = input_dim,output_dim_final = output_dim_final,astddev=astddev,bstddev=bstddev)
    

    output=x0
    
    
    for i in range(hidden_num):
        input_dim = add_hidden[i]
        output_dim = add_hidden[i+1]
        print('input_dim:%s, output_dim:%s'%(input_dim,output_dim))
        name_scope = 'hidden' + np.str(i+1)
            
        output,ua_w,ua_b,L2Wight_tmp=add_layer2(output,input_dim,output_dim,isresnet=isresnet,
                                               astddev=w_Univ0[i],bstddev=b_Univ0[i], ActFuc=ActFuc,
                                               seed=seed, name_scope= name_scope)
        w_Univ.append(ua_w)
        b_Univ.append(ua_b)
        L2w_all=L2w_all+L2Wight_tmp
    
    ua_we = tf.get_variable(
            name='ua_we'
            #, shape=[hidden_units[hidden_num-1], output_dim_final]
            , initializer=w_Univ0[-1]
        )
    ua_be = tf.get_variable(
            name='ua_be'
            #, shape=[1,output_dim_final]
            , initializer=b_Univ0[-1]
        )
    
    z1 = tf.matmul(output, ua_we)+ua_be
    w_Univ.append(ua_we)
    b_Univ.append(ua_be)
    
    # you can ignore this trick for now. Consider ASI=False
    if ASI:
        output=x0
        for i in range(hidden_num):
            input_dim = add_hidden[i]
            output_dim = add_hidden[i+1]
            print('input_dim:%s, output_dim:%s'%(input_dim,output_dim))
            name_scope = 'hidden' + np.str(i+1+hidden_num)
            output,ua_w,ua_b,L2Wight_tmp=add_layer2(output,input_dim,output_dim,isresnet=isresnet,
                                               astddev=w_Univ0[i],bstddev=b_Univ0[i], ActFuc=ActFuc,
                                               seed=seed, name_scope= name_scope)
        ua_we = tf.get_variable(
                name='ua_wei2'
                #, shape=[hidden_units[hidden_num-1], output_dim_final]
                , initializer=-w_Univ0[-1]
            )
        ua_be = tf.get_variable(
                name='ua_bei2'
                #, shape=[1,output_dim_final]
                , initializer=-b_Univ0[-1]
            )
        z2 = tf.matmul(output, ua_we)+ua_be
    else:
        z2=0
    z=z1+z2
    return z,w_Univ,b_Univ,L2w_all

tf.reset_default_graph()
with tf.variable_scope('Graph',reuse=tf.AUTO_REUSE) as scope:
    # Our inputs will be a batch of values taken by our functions
    x = tf.placeholder(tf.float32, shape=[None, R_variable['input_dim']], name="x")
    
    
    y_true = tf.placeholder_with_default(input=[[0.0]], shape=[None, R_variable['output_dim']], name="y")
    in_learning_rate= tf.placeholder_with_default(input=1e-3,shape=[],name='lr')
    y,_,_,_ = univAprox2(x, R_variable['hidden_units'],input_dim = R_variable['input_dim'],
                                            astddev=R_variable['astddev'],bstddev=R_variable['bstddev'],
                                            ActFuc=R_variable['ActFuc'],
                                            seed=R_variable['seed'],ASI=R_variable['ASI'])
    
    loss=tf.reduce_mean(tf.square(y_true-y))
    # We define our train operation using the Adam optimizer
    adam = tf.compat.v1.train.AdamOptimizer(learning_rate=in_learning_rate)
    train_op = adam.minimize(loss)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()  
        
class model():
    def __init__(self): 
        R_variable['y_train']=[]
        R_variable['y_test']=[]
        R_variable['loss_test']=[]
        R_variable['loss_train']=[]
    
        nametmp='%smodel/'%(FolderName)
        mkdir(nametmp)
        w_Univ0, b_Univ0=getWini(hidden_units=R_variable['hidden_units'],
                                 input_dim = R_variable['input_dim'],
                                 output_dim_final = R_variable['output_dim'],
                                 astddev=R_variable['astddev'],
                                 bstddev=R_variable['bstddev'])
        sess.run(tf.global_variables_initializer())
        saver.save(sess, "%smodel.ckpt"%(nametmp))
        y_test, loss_test_tmp= sess.run([y,loss],feed_dict={x: test_inputs, y_true: R_variable['y_true_test']})
        y_train,loss_train_tmp= sess.run([y,loss],feed_dict={x: train_inputs, y_true: R_variable['y_true_train']})
        R_variable['y_train']=y_train
        R_variable['y_test']=y_test
        R_variable['loss_test'].append(loss_test_tmp)
        R_variable['loss_train'].append(loss_train_tmp)
        
        self.ploty(name='ini') 
    def run_onestep(self):
        y_test, loss_test_tmp= sess.run([y,loss],feed_dict={x: test_inputs, y_true: R_variable['y_true_test']})
        
        y_train,loss_train_tmp= sess.run([y,loss],feed_dict={x: train_inputs, y_true: R_variable['y_true_train']})
            
        if R_variable['train_size']>R_variable['batch_size']:
            indperm = np.random.permutation(R_variable['train_size'])
            nrun_epoch=np.int32(R_variable['train_size']/R_variable['batch_size'])
            
            for ijn in range(nrun_epoch):
                ind = indperm[ijn*R_variable['batch_size']:(ijn+1)*R_variable['batch_size']] 
                _= sess.run(train_op, feed_dict={x: train_inputs[ind], y_true: R_variable['y_true_train'][ind],
                                                  in_learning_rate:R_variable['learning_rate']})
        else:
            _ = sess.run(train_op, feed_dict={x: train_inputs, y_true: R_variable['y_true_train'],
                                                  in_learning_rate:R_variable['learning_rate']})
        R_variable['learning_rate']=R_variable['learning_rate']*(1-R_variable['learning_rateDecay'])
            
        R_variable['y_train']=y_train
        R_variable['y_test']=y_test
        R_variable['loss_test'].append(loss_test_tmp)
        R_variable['loss_train'].append(loss_train_tmp)
    def run(self,step_n=1):
        nametmp='%smodel/model.ckpt'%(FolderName)
        saver.restore(sess, nametmp)
        for ii in range(step_n):
            self.run_onestep()
            if R_variable['loss_train'][-1]<R_variable['tol']:
                print('model end, error:%s'%(R_variable['lossu_train'][-1]))
                self.plotloss()
                self.ploty()
                self.savefile()
                nametmp='%smodel/'%(FolderName)
                shutil.rmtree(nametmp)
                saver.save(sess, "%smodel.ckpt"%(nametmp))
                break
            if ii==0:
                print('initial %s'%(np.max(R_variable['y_train'])))
                
            if ii%plotepoch==0:
                print('time elapse: %.3f'%(time.time()-t0))
                print('model, epoch: %d, test loss: %f' % (ii,R_variable['loss_test'][-1]))
                print('model, epoch: %d, train loss: %f' % (ii,R_variable['loss_train'][-1]))
                self.plotloss()
                self.ploty(name='%s'%(ii))
                self.savefile()
                nametmp='%smodel/'%(FolderName)
                shutil.rmtree(nametmp)
                saver.save(sess, "%smodel.ckpt"%(nametmp))
            
                
    def plotloss(self):
        plt.figure()
        ax = plt.gca()
        y1 = R_variable['loss_test']
        y2 = R_variable['loss_train']
        plt.plot(y1,'ro',label='Test')
        plt.plot(y2,'g*',label='Train')
        ax.set_xscale('log')
        ax.set_yscale('log')                
        plt.legend(fontsize=18)
        plt.title('loss',fontsize=15)
        fntmp = '%sloss'%(FolderName)
        mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
        
            
    def ploty(self,name=''):
        
        if R_variable['input_dim']==2:
            # Make data.
            X = np.arange(R_variable['x_start'], R_variable['x_end'], 0.1)
            Y = np.arange(R_variable['x_start'], R_variable['x_end'], 0.1)
            X, Y = np.meshgrid(X, Y)
            xy=np.concatenate((np.reshape(X,[-1,1]),np.reshape(Y,[-1,1])),axis=1)
            Z = np.reshape(get_y_func(xy),[len(X),-1])
            fp = plt.figure()
            ax = fp.gca(projection='3d')
            # Plot the surface.
            surf = ax.plot_surface(X, Y, Z-np.min(Z), cmap=cm.coolwarm,linewidth=0, antialiased=False)
            # Customize the z axis.
            #ax.set_zlim(-2.01, 2.01)
            ax.zaxis.set_major_locator(LinearLocator(5))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            # Add a color bar which maps values to colors.
            fp.colorbar(surf, shrink=0.5, aspect=5)
            ax.scatter(train_inputs[:,0], train_inputs[:,1], R_variable['y_train']-np.min(R_variable['y_train']))
            fntmp = '%s2du%s'%(FolderName,name)
            mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
        if R_variable['input_dim']==1:
            plt.figure()
            ax = plt.gca()
            y1 = R_variable['y_test']
            y2 = R_variable['y_train']
            y3 = R_variable['y_true_test']
            plt.plot(test_inputs,y1,'ro',label='Test')
            plt.plot(train_inputs,y2,'g*',label='Train')
            plt.plot(test_inputs,y3,'b*',label='True')
            plt.title('g2u',fontsize=15)        
            plt.legend(fontsize=18) 
            fntmp = '%su_m%s'%(FolderName,name)
            mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
            
    def savefile(self):
        with open('%s/objs.pkl'%(FolderName), 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(R_variable, f, protocol=4)
         
        text_file = open("%s/Output.txt"%(FolderName), "w")
        for para in R_variable:
            if np.size(R_variable[para])>20:
                continue
            text_file.write('%s: %s\n'%(para,R_variable[para]))
        
        text_file.close()
        
                
            
model1=model()
model1.run(100000)


