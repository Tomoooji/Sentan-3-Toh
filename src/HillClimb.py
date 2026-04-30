import numpy as np
import matplotlib.pyplot as plt

rng= np.random.default_rng()
size = (15,15)
xrange = (4,9)
yrange = (5,10)
hillheight = 2.5
pop_old = np.array([[x,y,0] for x in [5,7,11] for y in [5,7,11] if not (x==7 and y==7)]).astype(float)#
gen_num = 100
#fits = np.zeros(8)

def calc_fitness(x,y):
    if xrange[0]<x<xrange[1] and yrange[0]<y<yrange[1]:
        if y>=x+1 and y<-x+14:
            return x-4
        elif y>=x+1 and y>=-x+14:
            return 10-y
        elif y<x+1 and y>=-x+14:
            return 9-x
        elif y<x+1 and y<-x+14:
            return y-5
        else:
            return "e"
    else:
        return 0

def mutate(x,y,xr,yr):
    if 0.5>=xr>0.25:
        x+=0.1 if x+0.1<=15 else 0
    elif 0.25>=xr:
        x-=0.1 if x-0.1>=0 else 0
    if 0.5>=yr>0.25:
        y+=0.1 if y+0.1<=15 else 0
    elif 0.25>=yr:
        y-=0.1 if y-0.1>=0 else 0
    return x,y

def recombinate(pop):
    pop_=rng.permutation(pop)
    for i in range(4):#len(pop)//2):
        if rng.random()>0.5:
            pop_[2*i][0],pop_[2*i+1][0] = pop_[2*i+1][0],pop_[2*i][0]
            pop_[2*i][1],pop_[2*i+1][1] = pop_[2*i+1][1],pop_[2*i][1]
    return pop_

def select(pop,old):
    pop_=np.concatenate([[old[np.argsort(old[:,2])[-1]]],pop])#[:8]
    pop_=pop_[np.argsort(pop_[:,2][::-1])][:8]
    return pop_

pop_old[:,2]=np.array([calc_fitness(pop[0],pop[1]) for pop in pop_old])
print(pop_old)

for g in range(gen_num):
    print("-----------")
    pop_new=np.zeros_like(pop_old)
    xrand=rng.random(len(pop_old))
    yrand=rng.random(len(pop_old))
    pop_new=recombinate(pop_old)
    #print(pop_new)
    for i in range(8):
        pop_new[i][:2] = list(mutate(pop_new[i][0],pop_new[i][1],xrand[i],yrand[i]))
        pop_new[i,2] = calc_fitness(pop_new[i][0],pop_new[i][1])#+1
    #print(pop_new)
    pop_new=select(pop_new,pop_old)
    print(pop_new)
    pop_old=pop_new

#pop_old[:,2]=np.array([calc_fitness(pop[0],pop[1])+1 for pop in pop_old])
    
        