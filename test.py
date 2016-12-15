from data_provider import Data_provider
import numpy as np

data_provider = Data_provider()
X, y = data_provider.getdata(2)
X = np.array(X)
y = np.array(y)

