import pandas as pd  #se usa?


# Aplica un listado de métodos
def eval_measure_list(y,fname,mlist):
    n = len(mlist)
    vm = []
    vp = []
    va1 = []
    va2 = []
    ea2 = False
    for val in mlist:
        if isinstance(val, list):
            method = val[0]
            arg = val[1:]        
            p = fname(y,method,*arg)  # ver arg
            arg1 = arg[0]
            if len(arg)==2:
                arg2 = arg[0]
                ea2 = True
            else:
                arg2 = None
        else:
            method = val
            arg1 = None
            arg2 = None
            p = fname(y,method)            
        vm.append(method)     
        va1.append(arg1)   
        va2.append(arg2)   
        vp.append(p)     
       # print(method + ': ' + str(p)) 
    if ea2:
        d = {'method': vm, 'par1':va1,'par2':va2,'measure':vp}        
    else:
        d = {'method': vm, 'par1':va1,'measure':vp}
    df = pd.DataFrame(d) 
    return df

flatten = lambda l: [item for sublist in l for item in sublist]

def joinpar(x,y):
    if isinstance(x, list):
        return flatten([[x[0]],[y],x[1:]])
    else:
        return [x,y]

