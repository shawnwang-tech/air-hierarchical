
'air_impute_p5_4'
    O3
    72-24

'air_impute_p5-6'
    O3
    72-24
    batch 16

'air_impute_p5-7'
    O3
    24-72
    batch 16

air_impute_p5-8
    stgcn
    24-24

    use_met: False/True
    use_time: False

    在stgcn做MT得是dilated

air_impute_p5-9
    比较GC_GRU PM25GNN