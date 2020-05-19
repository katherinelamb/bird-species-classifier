import os
import random
from shutil import copyfile

#'.DS_Store' in train directory - skip this
species = os.listdir('data/100-bird-species/train')
for s in species:
    if s == '.DS_Store': continue 
    species_dir = os.path.join('data/100-bird-species/train_small/', s)
    if os.path.isdir(species_dir) is False:
        os.mkdir(species_dir)
    images =  os.listdir('data/100-bird-species/train/' + s)
    sample_num = min(25, len(images))
    random.seed(16)
    random_files = random.sample(images, sample_num)
    for f in random_files:
        src = os.path.join('data/100-bird-species/train/', s, f)
        dest = os.path.join(species_dir, f)
        copyfile(src, dest)