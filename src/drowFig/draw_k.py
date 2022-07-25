import matplotlib.pyplot as plt
y9=[257,270,264,261,249]
x1=[0.2,0.3,0.4,0.5,0.6]
y11=[263,275,266,261,252]
y13=[266,273,264,260,250]
y15=[256,270,265,260,252]

y9_4=[392,396,396,395,385]
y11_4=[390,395,399,393,387]
y13_4=[396,396,401,394,388]
y15_4=[396,397,398,395,387]


plt.plot(x1,y9,label='k = 9,R/8',linewidth=2,color='b',marker='o',
markerfacecolor='aqua',markersize=8)
plt.plot(x1,y11,label='k = 11,R/8',linewidth=2,color='g',marker='o',
markerfacecolor='lime',markersize=8)
plt.plot(x1,y13,label='k = 13,R/8',linewidth=2,color='m',marker='o',
markerfacecolor='violet',markersize=8)
plt.plot(x1,y15,label='k = 15,R/8',linewidth=2,color='y',marker='o',
markerfacecolor='yellow',markersize=8)
plt.xlabel('p')
plt.ylabel('R/8 sample')
plt.title('kernel size and p')
plt.legend()
plt.savefig('./123.png')
plt.close()
plt.plot(x1,y9_4,label='k = 9,R/4',linewidth=2,color='b',marker='o',
markerfacecolor='aqua',markersize=8,linestyle='dashed')
plt.plot(x1,y11_4,label='k = 11,R/4',linewidth=2,color='g',marker='o',
markerfacecolor='lime',markersize=8,linestyle='dashed')
plt.plot(x1,y13_4,label='k = 13,R/4',linewidth=2,color='m',marker='o',
markerfacecolor='violet',markersize=8,linestyle='dashed')
plt.plot(x1,y15_4,label='k = 15,R/4',linewidth=2,color='y',marker='o',
markerfacecolor='yellow',markersize=8,linestyle='dashed')
plt.xlabel('p')
plt.ylabel('R/4 sample')
plt.title('kernel size and p')
plt.legend()
plt.savefig('./456.png')
