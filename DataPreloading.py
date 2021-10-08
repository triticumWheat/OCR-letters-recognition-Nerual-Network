# encoding=utf-8
from scipy import misc
import imageio
import numpy as np

def main(src, dst):
    with open(src, 'r') as f:
        list = f.readlines()
    data = []
    labels = []
    for i in list:
        name, label = i.strip('\n').split(' ')
        name = "C:\\Users\\13794\\Desktop\\python\\" + name
        print(name + ' processed')
        img = imageio.imread(name)
        img = img / 255
        img.resize((img.size, 1))
        data.append(img)
        labels.append(int(label))

    print('write to npy')
    np.save(dst, [data, labels])
    print('completed')

if __name__ == '__main__':
    main("./validate.txt", "./validate")
