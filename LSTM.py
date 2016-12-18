import numpy as np
from utils import initw
from utils import softmax
import sys
from datetime import datetime
from utils import randi

class LSTM:

    @staticmethod
    def init(input_size, hidden_size, output_size):

        model = {}

        model['WLSTM'] = initw(input_size + hidden_size + 1, 4 * hidden_size)
        model['Wd'] = initw(hidden_size, output_size)
        model['bd'] = np.zeros((1, output_size))

        update = ['WLSTM', 'Wd', 'bd']

        return { 'model' : model, 'update' : update}

    @staticmethod
    def forward(Xs, model, **kwargs):

        X = Xs

        WLSTM = model['WLSTM']
        Wd = model['Wd']
        bd = model['bd']
        #n = X.shape[0]
        n = len(X)
        d = model['Wd'].shape[0]

        Hin = np.zeros((n, WLSTM.shape[0]))
        Hout = np.zeros((n, d))
        IFOG = np.zeros((n,d * 4))
        IFOGf = np.zeros((n, d * 4))
        C = np.zeros((n, d))

        for t in xrange(n):
            prev = np.zeros(d) if t == 0 else Hout[t-1]
            
            Hin[t, 0] = 1
            #Hin[t, 1:1+d] = X[t]
            Hin[t, 1:1+d] = prev
            Hin[t, 1+d+X[t]] = 1 # one hot representation

            IFOG[t] = Hin[t].dot(WLSTM)

            IFOGf[t, :3*d] = 1.0/(1.0 + np.exp(-IFOG[t, :3*d]))
            IFOGf[t, 3*d:] = np.tanh(IFOG[t, 3*d:])

            C[t] = IFOGf[t, :d] * IFOGf[t, 3*d:]
            if t>0: C[t] += IFOGf[t, d:2*d] * C[t-1]

            Hout[t] = IFOGf[t, 2*d:3*d] * np.tanh(C[t])

        #Y = Hout[1:, :].dot(Wd) + bd
        Y = Hout.dot(Wd) + bd
        Y = softmax(Y)

        cache = {}

        cache['WLSTM'] = WLSTM
        cache['Hout'] = Hout
        cache['Wd'] = Wd
        cache['IFOGf'] = IFOGf
        cache['IFOG'] = IFOG
        cache['C'] = C
        cache['X'] = X
        cache['Hin'] = Hin

        return Y, cache

    @staticmethod
    def fuck(Xs, model):
        y, c = LSTM.forward(Xs, model)
        model['WLSTM'] += 100
        y2, c = LSTM.forward(Xs,model)
        print 'y1 ', y
        print 'y2 ', y2

    @staticmethod
    def backword(dY, cache):

        Wd = cache['Wd']
        Hout = cache['Hout']
        IFOG = cache['IFOG']
        IFOGf = cache['IFOGf']
        C = cache['C']
        Hin = cache['Hin']
        WLSTM = cache['WLSTM']

        dWd = Hout.transpose().dot(dY)
        dbd = np.sum(dY, axis=0, keepdims=True)
        dHout = dY.dot(Wd.transpose())

        dIFOG = np.zeros(IFOG.shape)
        dIFOGf = np.zeros(IFOGf.shape)
        dWLSTM = np.zeros(WLSTM.shape)
        dHin = np.zeros(Hin.shape)
        dC = np.zeros(C.shape)
        #dX = np.zeros(X.shape)
        #dX = np.zeros((len(X)+1, Wd.shape[1]))
        n, d = Hout.shape

        for t in reversed(xrange(n)):
            tanhC = np.tanh(C[t])
            dIFOGf[t, 2*d:3*d] = tanhC * dHout[t]
            dC[t] += (1 - tanhC**2) * (IFOGf[t, 2*d:3*d] * dHout[t])

            #forget gate
            if t > 0:
                dIFOGf[t, d:2*d] = C[t-1] * dC[t]
                dC[t-1] += IFOGf[t, d:2*d] * dC[t]

            #input gate
            dIFOGf[t, :d] = IFOGf[t, 3*d:] * dC[t]
            dIFOGf[t, 3*d:] = IFOGf[t, :d] * dC[t]

            dIFOG[t, 3*d:] = (1 - IFOGf[t, 3*d:] ** 2) * dIFOGf[t, 3*d:]
            y = IFOGf[t, :3*d]
            dIFOGf[t, :3*d] = (y*(1.0-y)) * dIFOGf[t, :3*d]

            dWLSTM +=  np.outer(Hin[t], dIFOG[t])
            dHin[t]  = dIFOG[t].dot(WLSTM.transpose())

            #dX[t] = dHin[t,:d]
            if t > 0:
                dHout[t-1] += dHin[t, 1:1+d]
                #dHout[t-1] += dHin[t, d:]

        #return {'WLSTM' : dWLSTM, 'Wd' : dWd, 'bd' : dbd, 'dX' : dX }
        return {'WLSTM' : dWLSTM, 'Wd' : dWd, 'bd' : dbd}

    @staticmethod
    def predict(Xs, model, **kwargs):
        
        Y, cache = LSTM.forward(Xs, model)

        return np.argmax(Y, axis=1)

    @staticmethod
    def calc_total_loss(Xs, y, model):
        L = 0
        N = np.sum(len(y_i) for y_i in y)
        
        for i in xrange(len(y)):
            #print 'shuchu', i
            #print len(Xs), len(y)
            py, cache = LSTM.forward(Xs[i], model)
            #print 'input ', Xs[i]
            #print 'output ', np.argmax(softmax(py), axis=1)
            correct_word_prediction = py[np.arange(len(y[i])), y[i]]
            #print correct_word_prediction
            L += -1 * np.sum(np.log(correct_word_prediction))

        return L / N


    @staticmethod
    def sgd_step(Xs, y, learning_rate, model):
        py, cache = LSTM.forward(Xs, model)
       # print len(py), len(y)
        y = np.reshape(y,(len(y),-1))
        dY = py
        dY[np.arange(len(y)), y] -= 1
        #print 'len!! ', len(y)
        #print 'y!!, ', y
        #print 'dy!!, ', dY
        #print 'dY: ', dY

        bp = LSTM.backword(dY, cache)
        dWLSTM = bp['WLSTM']
        dWd = bp['Wd']
        dbd = bp['bd']

        #print 'ddddd', model['bd'], dbd
        model['WLSTM'] -= learning_rate * dWLSTM
        model['Wd'] -= learning_rate * dWd
        model['bd'] -= learning_rate *dbd

    @staticmethod
    def train_with_sgd(model, Xs, y, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
        losses = []
        num_examples_seen = 0
        #print 'cjjc', len(Xs), len(y)
        for epoch in xrange(nepoch):
            if(epoch % evaluate_loss_after ==0):
                loss = LSTM.calc_total_loss(Xs, y, model)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                print '%s loss %f %d' %(time, loss, num_examples_seen)
                if(len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate *= 0.5
                    print "Setting learning rate to %f" % learning_rate
                sys.stdout.flush()

            
            for i in xrange(len(y)):
                LSTM.sgd_step(Xs[i], y[i], learning_rate, model)
                num_examples_seen += 1

    @staticmethod
    def grad_check(X, Y, model):
        n = len(Y)

        for num in xrange(n):

            Xs = X[num]
            y = Y[num]

            #print '**************888'
            #LSTM.fuck(Xs, model)
            #print '**************888'

            py, cache = LSTM.forward(Xs, model)
            y = np.reshape(y,(len(y),-1))
            dY = py
            dY[np.arange(len(y)), y] -= 1
            bp = LSTM.backword(dY, cache)

            num_checks = 10
            #delta = 1e-5
            delta = 1
            rel_error_thr_warning = 1e-1
            rel_error_thr_error = 1

            num_checks = 100
            for p in model.keys():
                mat = model[p]

                for i in xrange(num_checks):
                    ri = randi(mat.size)

                    old_val = mat.flat[ri]
                    mat.flat[ri] = old_val + delta
                    #old_val = mat
                    #mat = old_val + delta
                    y1, c = LSTM.forward(Xs,model)
                    cost0 = LSTM.calc_total_loss(X, Y, model)
                    print 'FIFIF ', model[p].flat[ri]
                    mat.flat[ri] = old_val - delta
                    #mat = old_val - delta
                    y2, c = LSTM.forward(Xs,model)
                    print 'ENENE ', model[p].flat[ri]
                    #print 'y1 ',y1,
                    #print 'y2 ',y2
                    cost1 = LSTM.calc_total_loss(X, Y, model)
                    mat.flat[ri] = old_val
                    #mat = old_val
                    print 'cost ', cost0, cost1

                    grad_analytic = bp[p].flat[ri]
                    grad_numerical = (cost0 - cost1) / (2 * delta)


                    # compare them
                    if grad_numerical == 0 and grad_analytic == 0:
                        rel_error = 0 # both are zero, OK.
                        status = 'OK'
                    elif abs(grad_numerical) < 1e-7 and abs(grad_analytic) < 1e-7:
                        rel_error = 0 # not enough precision to check this
                        status = 'VAL SMALL WARNING'
                    else:
                        rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
                        status = 'OK'
                        if rel_error > rel_error_thr_warning: status = 'WARNING'
                        if rel_error > rel_error_thr_error: status = '!!!!! NOTOK'

                    # print stats
                    print '%s checking param %s index %8d (val = %+8f), analytic = %+8f, numerical = %+8f, relative error = %+8f' \
                    % (status, p, ri, old_val, grad_analytic, grad_numerical, rel_error)









