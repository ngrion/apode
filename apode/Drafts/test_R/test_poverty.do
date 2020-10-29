ssc install poverty

cd "D:\GitHub\apode\apode\Drafts\test_R"

import delimited income.csv

// headcount, gap, watts, fgt2, fgt15, cuh01, cuh05, sen, thon, takayama 
quietly  poverty y1, all 
di "$S_6, $S_8, $S_10, $S_13, $S_12, $S_20, $S_22, $S_25, $S_26, $S_27"
quietly poverty y2, all 
di "$S_6, $S_8, $S_10, $S_13, $S_12, $S_20, $S_22, $S_25, $S_26, $S_27"
quietly poverty y3, all 
di "$S_6, $S_8, $S_10, $S_13, $S_12, $S_20, $S_22, $S_25, $S_26, $S_27"
quietly poverty y4, all 
di "$S_6, $S_8, $S_10, $S_13, $S_12, $S_20, $S_22, $S_25, $S_26, $S_27"
quietly poverty y5, all 
di "$S_6, $S_8, $S_10, $S_13, $S_12, $S_20, $S_22, $S_25, $S_26, $S_27"
quietly poverty y6, all 
di "$S_6, $S_8, $S_10, $S_13, $S_12, $S_20, $S_22, $S_25, $S_26, $S_27"
 



/*
    S_1  = total number of observations in the data
    S_2  = number of observations used to compute the indices
    S_3  = weighted number of observations
    S_4  = value of the poverty line
    S_5  = weighted number of observations identified as poor
    S_6  = headcount ratio [FGT(0)]     
    S_7  = aggregate poverty gap
    S_8  = poverty gap ratio [FGT(1)]
    S_9  = income gap ratio 
    S_10 = Watts index
    S_11 = FGT(0.5)
    S_12 = FGT(1.5)
    S_13 = FGT(2)
    S_14 = FGT(2.5)
    S_15 = FGT(3)
    S_16 = FGT(3.5)
    S_17 = FGT(4)
    S_18 = FGT(4.5)
    S_19 = FGT(5)
    S_20 = Clark et al. index (0.10)
    S_21 = Clark et al. index (0.25)
    S_22 = Clark et al. index (0.5)
    S_23 = Clark et al. index (0.75)
    S_24 = Clark et al. index (0.90)
    S_25 = Sen index
    S_26 = Thon index
    S_27 = Takayama index
*/
