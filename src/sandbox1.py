import numpy as np
import os
import time
from glob import glob
from PIL import Image


start = time.time()

PATH = '/Users/fabienfluro/Documents/MS_BGD/Fil_Rouge/Work/Gender_CNN/data/raw2/'



f = open('data/dataset2.txt', 'wb')
test = np.array([2,3])
#test.tofile(f)

btest = bytes(test)
f.write(btest)
f.close()

# f = open('data/dataset2.txt', 'rb')
# test = np.fromfile(f)
# f.close()
# print(test)
# print(int.from_bytes(test, byteorder='big'))

with open('data/dataset2.txt','rb') as f: 
    content = f.read()
    print(content)
    print(int.from_bytes(content, byteorder='big'))

# count = 0
# dataset = []
# for root, dirs, files in os.walk(PATH):
#     count += 1
#     dataset.extend([np.asarray(Image.open(f)) for f in glob(os.path.join(root, '*.jpg'))])
#     print('=====================[' + str(count) + ']=====================')

# print('Total identies : ' + str(count))

# end = time.time()

# elapsed = end - start

# print('time elapsed : ' + str(elapsed))

# f = open('data/dataset.txt', 'wb')
# dataset = np.array(dataset)
# print(dataset)
# #print(dataset.tostring())
# #f.write(dataset.tobytes()) 
# dataset.tofile(f)
# f.close()



# # #np.savetxt('data/dataset2.txt', dataset)

# # print("===================")


# # #with open('data/dataset.txt','rb') as f: print(f.read())

# # # f = open('data/dataset.txt', 'rb')
# # # test = np.fromfile(f)
# # # f.close()
# # # print(test)


# # # H x W x RGB
