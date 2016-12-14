import numpy as np
from utils import initw

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
    def forward(Xs, model, params, **kwargs):

        X = Xs

        WLSTM = model['WLSTM']
        Wd = model['Wd']
        bd = model['bd']
        n = X.shape[0]
        d = model['Wd'].shape[0]

        Hin = np.zeros((n, WLSTM.shape[0]))
        Hout = np.zeros((n, d))
        IFOG = np.zeros((n * 4))
        IFOGf = np.zeros((n * 4))
        C = np.zeros((n, d))

        for t in xrange(n):
            prev = np.zeros(d) if t == 0 else Hout[t-1]

            Hin[t, 0] = 1
            Hin[t, 1:1+d] = X[t]
            Hin[t, 1+d:] = prev

            IFOG[t] = Hin[t].dot(WLSTM)

            IFOGf[t, :3*d] = 1.0/(1.0 + np.exp(-IFOG[t, :3*d]))
            IFOGf[t, 3*d:] = np.tanh(IFOG[t, 3*d:])

            C[t] = IFOGf[t, :d] * IFOGf[t, 3*d:]
            if t>0: C[t] += IFOGf[t, d:2*d] * C[t-1]

            Hout[t] = IFOGf[t, 2*d:3*d] * np.tanh(C[t])

        #Y = Hout[1:, :].dot(Wd) + bd
        Y = Hout.dot(Wd) + bd

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
    def backword(dY, cache):

        Wd = cache['Wd']

        Hout = cache['Hout']
        IFOG = cache['IFOG']
        IFOGf = cache['IFOGf']
        C = cache['C']
        Hin = cache['Hin']
        WLSTM = cache['WLSTM']
        X = cache['X']

        dWd = Hout.transpose().dot(dY)
        dbd = np.sum(dY, axis=0, keepdims=True)
        dHout = dY.dot(Wd.transpose())

        dIFOG = np.zeros(IFOG.shape)
        dIFOGf = np.zeros(IFOGf.shape)
        dWLSTM = np.zeros(WLSTM.shape)
        dHin = np.zeros(Hin.shape)
        dC = np.zeros(C.shape)
        dX = np.zeros(X.shape)

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

            dX[t] = dHin[t,:d]
            if t > 0:
                dHout[t-1] += dHin[t, d:]

        return {'WLSTM' : dWLSTM, 'Wd' : dWd, 'bd' : dbd, 'dX' : dX }

