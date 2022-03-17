final = []
NX=64
NY=(int)(NX/2)
for i in range(NY):
    cols = []
    for j in range(NX):
        cols.append(0)
    final.append(cols)


# for k in final:
#     print(k) 

for i in range(NY):
    for j in range(NX):
        if j<NX/8:
            if j==0:
                if i<NY/4 or i>NY/2:
                    final[i][j] = 4
                else:
                    final[i][j] = 2
            else:
                if i<NY/4 or i>NY/2 :
                    final[i][j] = 4
                elif i == NY/4 or i == NY/2:
                    final[i][j] = 3
                else:
                    final[i][j] = 1
                
        elif j<NX/4:
            if j==NX/8:
                if i>(NY-NY/4) or i<NY/4:
                    final[i][j] = 4
                elif i==(NY-NY/4) or i == NY/2 or i== NY/4:
                    if i==(NY-NY/4) :
                        final[i][j] = 5
                    elif i == NY/2:
                        final[i][i] = 6
                    else:
                        final[i][j] = 3
                elif i>NY/2 and i<(NY-NY/4):
                    final[i][j] = 3
                else:
                    final[i][j] = 1    
            else:
                # final[i][j] = 0
                if i>(NY-NY/4) or i<NY/4 :
                    final[i][j] = 4
                elif i==(NY-NY/4) or i==NY/4:
                    final[i][j] = 3
                else:
                    final[i][j] =1
        elif j<(NX/2- NX/8):
            if j==NX/4:
                if i<NY/4 or i==NY-1 :
                    final[i][j] = 4
                elif i==(NY-NY/4) or i == NY/4 or i == NY - 2 or i==NY/2:
                    if i==(NY-NY/4):
                        final[i][j] = 6
                    elif i == NY/4:
                        final[i][j] = 5
                    elif i == NY/2:
                        final[i][j] = 6
                    else:
                        final[i][j] = 5
                elif i>(NY-NY/4) and i<(NY):
                    final[i][j] = 3
                elif i<(NY-NY/4) and i>NY/2:
                    final[i][j] = 1
                else:
                    final[i][j] = 3
            else:
                if i == NY/2 or i == NY-2:
                    final[i][j] = 3
                elif i>NY/2 and i<NY-1:
                    final[i][j] = 1
                else:
                    final[i][j] = 4
        elif j<NX/2:
            if j == (NX/2- NX/8):
                if i<NY/2 or i==NY-1:
                    final[i][j] = 4
                elif i<(NY - NY/4):
                    final[i][j] = 3
                    if i==NY/2:
                        final[i][j] = 5
                elif i == (NY-NY/4):
                        final[i][j] = 6
                elif i==NY-2:
                    final[i][j] = 3
                else:
                    final[i][j] = 1 
            else:
                if i<(NY - NY/4) or i==NY-1:
                    final[i][j] = 4
                elif i==(NY - NY/4) or i == NY-2:
                    final[i][j] =3
                else:
                    final[i][j] = 1
        elif j == NX/2:
            if i<=(NY - NY/4) or i == NY-2 or i <= 1:
                final[i][j] =3
                if( i == (NY - NY/4)):
                    final[i][j] = 6
                elif(i == 1):
                    final[i][j] = 5
                elif(i==0):
                    final[i][j] = 4
                else:
                    final[i][j] = 3
            elif i==NY-1 :
                final[i][j]  =4
            else:
                final[i][j] =1
        else:
            if(i == 0 or i == NY -1):
                final[i][j] = 4
            elif(i == 1 or i == NY -2):
                final[i][j] = 3
            else:
                final[i][j] = 1
            if(j == NX-1 and i<NY-1 and i>0):
                final[i][j] = 7

for k in final:
    print(k)
print("")

num = [[]]
for i in range(NX):
    num[0].append(i%10)

for i in num:
    print(i, end='')
print("")