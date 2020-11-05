#install.packages("ineq")
#install.packages("affluenceIndex")

#=========================== ineq ============================
library('ineq') 

n = 100 # par

y1 <- rep(10,n)
y2 <- seq(1,n)
y3 <- y2^2
y4 <- c(rep(0,n-1),10)

n1 = n/2
a = seq(1,n1)^1.2
b = n1^2
y5 = sort(c(b - a,b + a))
y5 = y5-min(y5)+1

a = seq(1,n1)^0.5;
y6 = sort(c(b - a,b + a))
y6 = y6-min(y6)+1



r1 <- c(Gini(y1), Gini(y2), Gini(y3), Gini(y4), Gini(y5), Gini(y6))


b <- 0.5
r2 <- c(Atkinson(y1, parameter = b),Atkinson(y2, parameter = b),
       Atkinson(y3, parameter = b),Atkinson(y4, parameter = b),
       Atkinson(y5, parameter = b),Atkinson(y6, parameter = b))


b <- 1
r3 <- c(Atkinson(y1, parameter = b),Atkinson(y2, parameter = b),
        Atkinson(y3, parameter = b),Atkinson(y4, parameter = b),
        Atkinson(y5, parameter = b),Atkinson(y6, parameter = b))


b <- 2
r4 <- c(Atkinson(y1, parameter = b),Atkinson(y2, parameter = b),
        Atkinson(y3, parameter = b),Atkinson(y4, parameter = b),
        Atkinson(y5, parameter = b),Atkinson(y6, parameter = b))



b <- 0
r5 <- c(entropy(y1, parameter = b),entropy(y2, parameter = b),
        entropy(y3, parameter = b),entropy(y4, parameter = b),
        entropy(y5, parameter = b),entropy(y6, parameter = b))

b <- 1
r6 <- c(entropy(y1, parameter = b),entropy(y2, parameter = b),
        entropy(y3, parameter = b),entropy(y4, parameter = b),
        entropy(y5, parameter = b),entropy(y6, parameter = b))

b <- 2
r7 <- c(entropy(y1, parameter = b),entropy(y2, parameter = b),
        entropy(y3, parameter = b),entropy(y4, parameter = b),
        entropy(y5, parameter = b),entropy(y6, parameter = b))


b <- 0.5
r8 <- c(Kolm(y1, parameter = b),Kolm(y2, parameter = b),
        Kolm(y3, parameter = b),Kolm(y4, parameter = b),
        Kolm(y5, parameter = b),Kolm(y6, parameter = b))


r9 <- c(var.coeff(y1), var.coeff(y2), var.coeff(y3),
        var.coeff(y4), var.coeff(y5), var.coeff(y6))


r10 <- c(Rosenbluth(y1), Rosenbluth(y2), Rosenbluth(y3),
        Rosenbluth(y4), Rosenbluth(y5), Rosenbluth(y6))

# Herfindahl es diferente en ineq


"============================== affluenceIndex ================================="
library('affluenceIndex') 

q1 <- c(polar.aff(y1)$p.scalar, polar.aff(y2)$p.scalar, polar.aff(y3)$p.scalar,
        polar.aff(y4)$p.scalar, polar.aff(y5)$p.scalar, polar.aff(y6)$p.scalar)


"============================================================================"

df <- data.frame(gini=r1,atkinson05=r2,atkinson1=r3,atkinson2=r4,
                 entropy0=r5,entropy1=r6,entropy2=r7,
                 kolm05 = r8, varcoeff = r9, rosenbluth=r10,
                 wolfson = q1)

write.csv(x=df, file="test_ineq.csv",row.names = FALSE)


