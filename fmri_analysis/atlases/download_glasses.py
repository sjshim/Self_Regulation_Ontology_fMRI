import os
import urllib2

# get atlas nii.gz image from https://neurovault.org/collections/1549/
if not os.path.exists('HCPMMP1_on_MNI152_ICBM2009a_nlin.txt'):
    response = urllib2.urlopen('https://ndownloader.figshare.com/files/5534027')
    with open('HCPMMP1_on_MNI152_ICBM2009a_nlin.txt','wb') as output:
        output.write(response.read())