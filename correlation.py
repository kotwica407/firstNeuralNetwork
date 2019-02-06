from functions import readdata
from functions import valuesOfAttribute
from functions import words, chars
import numpy as np


def Round_To_n(x, n):
    return round(x, -int(np.floor(np.sign(x) * np.log10(abs(x)))) + n)


inp = []
out = []

readdata(inp, out)

corr = np.zeros(shape=(len(inp[0]), len((inp[0]))))

for w in range(len(corr)):
    for k in range(len(corr[0])):
        x = valuesOfAttribute(inp, w)
        y = valuesOfAttribute(inp, k)
        ccf = np.corrcoef(np.transpose(x), np.transpose(y))
        corr[w][k] = ccf[0][1]
        if abs(corr[w][k]) > 0.2:
            if w != k:
                if w <= 47:
                    pierwsze = words[w]
                else:
                    if w <= 53:
                        pierwsze = chars[w - 47]
                    else:
                        pierwsze = w
                if k <= 47:
                    drugie = words[k]
                else:
                    if k <= 53:
                        drugie = chars[k - 47]
                    else:
                        drugie = k
                print(pierwsze, '_', drugie, 'corr = ', corr[w][k])

np.savetxt('korelacja.txt', corr, delimiter=" ", fmt='%1.2f')

