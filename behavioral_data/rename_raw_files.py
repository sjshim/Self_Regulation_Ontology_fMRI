# script to fix naming of raw data files

tasks = ['ANT', 'CCTHot', 'discountFix',
         'DPX', 'motorSelectiveStop',
         'stopSignal', 'stroop', 'surveyMedley',
         'twoByTwo', 'WATT3']

import glob
from os import path, rename

raw_dir = '/home/ian/Downloads/'

def lower(string):
    return string[0].lower() + string[1:]

problems = 0
fixes = 0
for filey in glob.glob(path.join(raw_dir,'*','*')):
    base = filey.split('_')[0]
    task = filey.split('_')[1][0:-4]
    if task not in tasks:
        problems += 1
        print(filey)
        new_file = None
        if lower(task) in tasks:
            new_file = base + '_%s.csv' % lower(task)
        if task == 'motorStop' or task == 'MotorStop':
            new_file = base + '_%s.csv' % 'motorSelectiveStop'
        if task == 'WATT':
            new_file = base + '_%s.csv' % 'WATT3'
        if task == 'CCT':
            new_file = base + '_%s.csv' % 'CCTHot'
        if lower(task) == 'DiscountFixed':
            new_file = base + '_%s.csv' % 'DiscountFix'
        if new_file:
            fixes += 1
            print('can Fix! newfile: %s\n' % new_file)
            rename(filey, new_file)
print('Problems: %s, Fixes: %s' % (problems, fixes))
    
            
            