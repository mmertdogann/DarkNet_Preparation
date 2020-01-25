import os
from PIL import Image

def create_labels(src, tar):
   if not os.path.exists(tar):
       os.makedirs(tar)
   if not os.path.isfile(tar + '/labels.txt'):
       labels = open(tar + '/labels.txt', 'w+')
       categories = os.listdir(src)
       for category in categories:
           labels.write(category + '\n')
       labels.close()

def generate_dataset(src, tar, size=None, split=0.8):
   if not os.path.exists(tar + 'backup/'):
       os.makedirs(tar + 'backup/')
   if not os.path.exists(tar + 'train/'):
       os.makedirs(tar + 'train/')
   if not os.path.exists(tar + 'test/'):
       os.makedirs(tar + 'test/')
   trainlist = open(tar+"train.list", "w+")
   testlist = open(tar+"test.list", "w+")
   exceptions = []
   for category in os.listdir(src):
       i = 0
       filenames = os.listdir(src + category)
       length = len(filenames)
       print(category, '...')
       for filename in filenames:
           s = src + category + '/' + filename
           if i / length < split:
               d = tar + 'train/' + str(i) + '_' + category + '.jpg'
               trainlist.write(d + '\n')
           else:
               d = tar + 'test/' + \
                   str(int(i - length*split)) + '_' + category + '.jpg'
               testlist.write(d + '\n')
           try:
               img = Image.open(s)
               if size is not None:
                   img = img.resize(size, Image.ANTIALIAS)
               img.convert('RGB').save(d)
           except:
               exceptions.append(s)
               continue
           i += 1
   trainlist.close()
   testlist.close()
   print('EXCEPTIONS\n', exceptions)

def create_data_file(src, tar, model_name='Senior', top=3):
   with open(tar + model_name + '.data', 'w+') as data:
       numOfClasses = len(os.listdir(src))
       data.write('classes=' + str(numOfClasses) + '\n' +
                  'train = ' + tar + 'train.list' + '\n' +
                  'valid = ' + tar + 'test.list' + '\n' +
                  'labels = ' + tar + 'labels.txt' + '\n' +
                  'backup = ' + tar + 'backup/' + '\n' +
                  'top = ' + str(top))

if __name__ == '__main__':
   SRC = '/home/boxx-gpuserver/Mert_Dogan/Dataset/'
   TAR = '/home/boxx-gpuserver/Mert_Dogan/'
   MODEL_NAME = 'Senior'
   DARKNET_PATH = '~/darknet/'
   WEIGHTS_FILE = DARKNET_PATH + 'darknet53.conv.74'  # can be replaced with ''
   SIZE = (350, 350)
   SPLIT = 0.90
   create_labels(SRC, TAR)
   create_data_file(SRC, TAR, MODEL_NAME, 5)
   generate_dataset(SRC, TAR, SIZE, SPLIT)
   train = input('Egitimle devam etmek ister misiniz? ([Y]/[n])')
   train_command = DARKNET_PATH + './darknet classifier train ' + TAR + \
       MODEL_NAME + '.data ' + TAR + MODEL_NAME + '.cfg ' + WEIGHTS_FILE + ' -gpus 0,1,2,3,4,5,6,7'
   if train == '' or train == 'Y' or train == 'y':
       print('=>  ', train_command)
       os.system(train_command)
   else:
       print('Manuel olarak egitmek isterseniz : \n', train_command)
