import numpy as np
from collections import Counter


data = np.array(['a', 'b', 'c', 'a', 'e', None,
                'g', 'h', 'i', 'j'], dtype=object)
mode_value = Counter(data).most_common(1)[0][0]
print(mode_value)

data = np.array(
    [x if x is not None else mode_value for x in data], dtype=object)

print(data)
