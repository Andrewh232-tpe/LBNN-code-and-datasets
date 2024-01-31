AAA=(abalone19.csv glass4.csv segment0.csv
abalone9-18.csv  page-blocks0.csv yeast4.csv
ecoli4.csv   pima.csv  yeast5.csv
glass2.csv  yeast6.csv)
for i in ${AAA[@]}; do
        python3 OC-SVM.py $i
done