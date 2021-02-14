import csv

with open('./files/impressions_files.txt', 'r', encoding='utf-8') as r:
    data = r.read().lower().split('\n')

    files = [s.partition(', ')[0] for s in data]
    impressions = [s.partition(', ')[2] for s in data]

    no_imp = []
    neg_imp = []
    neg_2_imp = []
    abnorm_imp = []
    normal_imp = []
    clear_imp = []
    hyper_imp = []
    bi_imp = []
    opac_imp = []
    pul_imp = []
    pneu_imp = []
    bas_imp = []
    emp_imp = []
    copd_imp = []
    frac_imp = []
    or_imp = []
    stable_imp = []
    chr_imp = []
    low_imp = []
    rem_imp = []
    acute_imp = []
    no_evi_imp = []
    rest_imp = []




    #impressions = [x.replace('1.', '') for x in impressions]


    for i, imp in enumerate(impressions):
        if 'no ' in imp[:4]:
            no_imp.append((i, imp))
            continue
        elif 'negative ' in imp[:9]:
            neg_imp.append((i, imp))
            continue
        elif 'clear' in imp: #[:7]:
            clear_imp.append((i, imp))
            continue
        elif 'negative' in imp:
            neg_2_imp.append((i, imp))
            continue
        elif 'acute' in imp:
            acute_imp.append((i, imp))
            continue
        elif 'no evidence' in imp:
            no_evi_imp.append((i, imp))
            continue
        elif 'normal' in imp:
            normal_imp.append((i, imp))
            continue
        elif 'abnormal' in imp or 'borderline' in imp or 'cardio' in imp:
            abnorm_imp.append((i, imp))
            continue
        elif 'hyper' in imp:
            hyper_imp.append((i, imp))
            continue
        elif 'pneu' in imp:
            pneu_imp.append((i, imp))
            continue
        elif 'bi' in imp:
            bi_imp.append((i, imp))
            continue
        elif 'basilar' in imp:
            bas_imp.append((i, imp))
            continue
        elif 'opac' in imp:
            opac_imp.append((i, imp))
            continue
        elif 'pulmo' in imp:
            pul_imp.append((i, imp))
            continue
        elif 'emphy' in imp:
            emp_imp.append((i, imp))
            continue
        elif 'copd' in imp:
            copd_imp.append((i, imp))
            continue
        elif 'scar' in imp or 'frac' in imp:
            frac_imp.append((i, imp))
            continue
        elif 'without' in imp or 'otherwise' in imp:
            or_imp.append((i, imp))
            continue
        elif 'stable' in imp:
            stable_imp.append((i, imp))
            continue
        elif 'chronic' in imp:
            chr_imp.append((i, imp))
            continue
        elif 'low' in imp or 'mild' in imp:
            low_imp.append((i, imp))
            continue
        elif 'remarkable' in imp:
            rem_imp.append((i, imp))
            continue
        else:
            rest_imp.append((i, imp))

norm = [acute_imp, clear_imp, neg_imp, neg_2_imp, no_imp, normal_imp, or_imp, rem_imp, stable_imp, no_evi_imp]
abnorm = [abnorm_imp, bas_imp, bi_imp, chr_imp, emp_imp, frac_imp, hyper_imp, low_imp, opac_imp, pneu_imp, pul_imp, copd_imp, rest_imp]


labels = {}

for lst in norm:
    for impr in lst:
        labels.update({files[impr[0]]: (impr[1],0)})
for lst in abnorm:
    for impr in lst:
        labels.update({files[impr[0]]: (impr[1],1)})



with open('./files/labels.csv', 'w') as csvfile:
    csvWriter = csv.writer(csvfile, delimiter='\t',
                           quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvWriter.writerow(['Filename', 'Impression', 'Label'])
    for key, value in labels.items():
        csvWriter.writerow([key, value[0], value[1]])

with open('./files/normal_updated.txt', 'w', encoding='utf-8') as f:
    for lst in norm:
        for impr in lst:
            f.write(f'{files[impr[0]]}\t{impr[1]}\n')

with open('./files/abnormal_updated.txt', 'w', encoding='utf-8') as f:
    for lst in abnorm:
        for impr in lst:
            f.write(f'{files[impr[0]]}\t{impr[1]}\n')
