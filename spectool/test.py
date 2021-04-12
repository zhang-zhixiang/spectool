import numpy as np
import matplotlib.pyplot as plt

import liblogccf
print('liblogccf =', liblogccf)

# data = [1., 2., 3., 4., 5., 6., 7.]
data = np.arange(0, 13, 0.1)
print('data =', data)
Smodel = liblogccf.Shift_spec(data)
newdata = Smodel.get_shift_spec_arr(1.0)
print('newdata =', newdata)

plt.plot(data)
for shift in range(1, 30, 4):
    newdata = Smodel.get_shift_spec_arr(shift)
    plt.plot(newdata)
plt.show()