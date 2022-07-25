import random
import pandas as pd

list = random.sample(range(0,1136),1136)
dataFrame = pd.DataFrame({'randomSampleId':list})
dataFrame.to_csv('./randomSample.csv')
print(list)

so = list.sort()

print(list)