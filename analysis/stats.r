library("readxl")
sink("stats-output.txt", append=FALSE)

total_comparisons=0

baseline = function(tidnet_file, baseline_file, skip_first=FALSE){
    deep = read_excel(tidnet_file)
    shallow = read_excel(baseline_file)
    if (skip_first){
        deep = deep[-c(1),]
        shallow = shallow[-c(1),]
    }

    print(wilcox.test(deep$...2, shallow$...2, exact=TRUE, paired=TRUE))
    print(wilcox.test(deep$'_mixup', shallow$...2, exact=TRUE, paired=TRUE))
    print(wilcox.test(deep$'_ea', shallow$...2, exact=TRUE, paired=TRUE))
    print(wilcox.test(deep$'_ea_mixup', shallow$...2, exact=TRUE, paired=TRUE))

    ## Increase comparisons count
    total_comparisons <<- total_comparisons + 4
}

baseline_dg = function(tidnet_file, baseline_file, skip_first=FALSE){
    deep = read_excel(tidnet_file)
    shallow = read_excel(baseline_file)
    if (skip_first){
        deep = deep[-c(1),]
        shallow = shallow[-c(1),]
    }

    print(wilcox.test(deep$'_mixup', shallow$'_mixup', exact=TRUE, paired=TRUE))
    print(wilcox.test(deep$'_ea', shallow$'_ea', exact=TRUE, paired=TRUE))
    print(wilcox.test(deep$'_ea_mixup', shallow$'_ea_mixup', exact=TRUE, paired=TRUE))

    ## Increase comparisons count
    total_comparisons = total_comparisons + 4
}

result_set = function(tidnet_template, baseline_template, skip_first=FALSE){
    baseline(sprintf(tidnet_template, ""), sprintf(baseline_template, ""), skip_first=skip_first)
    baseline_dg(sprintf(tidnet_template, ""), sprintf(baseline_template, ""), skip_first=skip_first)

    baseline(sprintf(tidnet_template, "_mdl"), sprintf(baseline_template, "_mdl"), skip_first=skip_first)
    baseline_dg(sprintf(tidnet_template, "_mdl"), sprintf(baseline_template, "_mdl"), skip_first=skip_first)
}

ft_versus_mdl = function(ft, mdl, length){
    data_ft = data.matrix(read.csv(ft)[, 2:length])
    data_mdl = data.matrix(read.csv(mdl)[, 2:length])

    # target performance
    print(sprintf("Fine Tuning Target Performance: %f", mean(diag(data_ft))))
    print(sprintf("MDL Target Performance: %f", mean(diag(data_mdl))))
    print(wilcox.test(diag(data_ft), diag(data_mdl), paired=TRUE))

    # General performance
    diag(data_ft) <- NA
    diag(data_mdl) <- NA
    print(sprintf("Fine Tuning Target Performance: %f", mean(data_ft, na.rm=TRUE)))
    print(sprintf("MDL Target Performance: %f", mean(data_mdl, na.rm=TRUE)))
    print(wilcox.test(c(data_ft), c(data_mdl), paired=TRUE))
}

print("BCI_2a")
result_set('aggregated/BCI_2a_TIDNet%s.xlsx', 'aggregated/BCI_2a_ShallowConvNet%s.xlsx', skip_first=TRUE)

print("P300")
result_set('aggregated/P300_TIDNet%s.xlsx', 'aggregated/P300_EEGNet%s.xlsx')

# MMI
print("MMI 2")
result_set('aggregated/MMI_2_TIDNet%s.xlsx', 'aggregated/MMI_2_EEGNet%s.xlsx')

print("MMI 3")
result_set('aggregated/MMI_3_TIDNet%s.xlsx', 'aggregated/MMI_3_EEGNet%s.xlsx')

print("MMI 4")
result_set('aggregated/MMI_4_TIDNet%s.xlsx', 'aggregated/MMI_4_EEGNet%s.xlsx')

print("ERN")
result_set('aggregated/ERN_TIDNet%s.xlsx', 'aggregated/ERN_EEGNet%s.xlsx')

print(sprintf("\n%d Total Comparisons\n", total_comparisons))
print(sprintf("\nSignificance after bonferroni: %f\n", 0.05 / total_comparisons))


print("Fine-tuning versus MDL")
print('MMI')
ft_versus_mdl('mmi_2_ft.csv', 'mmi_2_mdl.csv', 106)
ft_versus_mdl('mmi_3_ft.csv', 'mmi_3_mdl.csv', 106)
ft_versus_mdl('mmi_4_ft.csv', 'mmi_4_mdl.csv', 106)
print('ERN')
ft_versus_mdl('ern_ft.csv', 'ern_mdl.csv', 11)

