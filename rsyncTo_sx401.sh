#/bin/bash 

rsync -v -r -t -u simple.run sx401:/net/home/m/mabdelra/amplopt 
rsync -v -r -t -u bat.dat sx401:/net/home/m/mabdelra/amplopt 
rsync -v -r -t -u period.dat sx401:/net/home/m/mabdelra/amplopt 
rsync -v -r -t -u input/energycharts_downloads_price_2018_en_15min.dat_24.0_12.0_H_subFiles sx401:/net/home/m/mabdelra/amplopt/input/energycharts_downloads_price_2018_en_15min.dat_24.0_12.0_H_subFiles 
