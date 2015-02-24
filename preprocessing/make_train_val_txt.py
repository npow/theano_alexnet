'''
For generating caffe style train and validation label txt files
'''
import os
import yaml
from sklearn.cross_validation import train_test_split

with open('paths.yaml', 'r') as f:
    paths = yaml.load(f)

train_img_dir = paths['train_img_dir']
misc_dir = paths['misc_dir']

valtxt_filename = os.path.join(misc_dir, 'val.txt')
traintxt_filename = os.path.join(misc_dir, 'train.txt')

train_dirs = [name for name in os.listdir(train_img_dir) if os.path.isdir(os.path.join(train_img_dir, name))]

def is_orig(filename):
  filename = os.path.basename(filename)
  return filename.find('_') == -1

def add_rotations(fnames):
  L = []
  for x in fnames:
    fname = x[0]
    klass = x[1]
    L.append((fname,klass))
    bname = os.path.splitext(fname)[0]
    for r in [90, 180, 270]:
      L.append(("%s_%d.jpg" % (bname, r), klass))
  return L

fnames = []
for i,folder in enumerate(train_dirs):
    fnames += [(os.path.join(train_img_dir, folder, x), i) for x in filter(lambda x: is_orig(x), os.listdir(os.path.join(train_img_dir, folder)))]
train_fnames, val_fnames = train_test_split(fnames, test_size=0.1, random_state=42)

with open(traintxt_filename, 'w') as f:
    for x in add_rotations(train_fnames):
      path = x[0]
      klass = x[1]
      f.write("%s %d\n" % (path, klass))
    
with open(valtxt_filename, 'w') as f:
    for x in add_rotations(val_fnames):
      path = x[0]
      klass = x[1]
      f.write("%s %d\n" % (path, klass))
    
