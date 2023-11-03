import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# d = pd.DataFrame({'水果':['苹果','梨','草莓'],
#        '数量':[3,2,5],
#        'date':[datetime.strptime('2242022 21:54:32', '%m%d%Y %H:%M:%S').timestamp(),datetime.strptime('2242022 21:54:34', '%m%d%Y %H:%M:%S').timestamp(),datetime.strptime('2242022 21:54:37', '%m%d%Y %H:%M:%S').timestamp()]})
# print(d)
# d['tmp'] = d['date'].shift()
# d['add'] = abs(d['date'] - d['tmp'])
# d['session_id'] = (d['add']>=2).cumsum()
# print(d)

# a = datetime.strptime('2242022 21:54:32', '%m%d%Y %H:%M:%S').timestamp()
# print('a:',a)
# b =  datetime.strptime('2242022 21:54:33', '%m%d%Y %H:%M:%S').timestamp()
# print('b:',b)
# c = abs(a-b)
# print('c:',c)

# a = [[1,2],[3,4],[4,5,6]]
# b = [[2],[4],[8]]
# print((a,b))
a = torch.range(1,24).resize(2,3,4)
b = a
print(a)
print(torch.einsum('ijk,ijk->ij', a, b))




