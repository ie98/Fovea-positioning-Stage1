import matplotlib.pyplot as plt

x1=[31,29,27,25,23,21]
y_subset1=[6.53,6.41,6.49,5.83,6.07,6.22]
y_subset2=[7.69,7.83,7.12,6.88,6.84,6.92]


plt.plot(x1,y_subset1,label='train in subset1',linewidth=2,color='b',marker='o',
markerfacecolor='aqua',markersize=8)
plt.plot(x1,y_subset2,label='train in subset2',linewidth=2,color='g',marker='o',
markerfacecolor='lime',markersize=8)

plt.xlabel('r')
plt.ylabel('dist')
plt.title('')
plt.legend()
plt.savefig('./r.png')
plt.close()

