import matplotlib.image as mping
import numpy as np
import os
import scipy.misc


##class Data_process:


# 遍历处理数据集
def read_image(type1 = 'train',type2 = 'label'):
    base_path = os.getcwd()
    for i in os.listdir(base_path):
        path_1 = os.path.join(base_path,str(i))
        if os.path.isdir(path_1):
            
            # 处理train数据集
            if i == type1 == 'train' or i == type1 == 'val':
                for j in os.listdir(path_1):
                    path_2 = os.path.join(path_1,str(j))
                    
                    # 处理image数据集
                    if j == type2 == 'image':
                        filenames = []
                        for file in os.listdir(path_2):
                            filenames.append(file)

                        # 遍历读取数据
                        for num in range(len(filenames)):
                            img = mping.imread(os.path.join(path_2,filenames[num]))
                            print(os.path.join(path_2,filenames[num]))

                            # 创建路径
                            dir_name = os.path.join(path_1,'image_clip')
                            if not os.path.exists(dir_name):
                                os.mkdir(dir_name)

                            # 遍历裁剪
                            targetSize = 256
                            height ,width= img.shape[0], img.shape[1]
                            rows , cols = height//targetSize+1, width//targetSize+1
                            subImg_num = 0
                            for i in range(rows):
                                for j in range(cols):
                                    if (i+1)*targetSize < height and (j+1)*targetSize < width:
                                        temp_img = img[targetSize * i : targetSize * i + targetSize, targetSize * j : targetSize * j + targetSize, :]
                                    elif (i+1)*targetSize < height and (j+1)*targetSize > width:
                                        temp_img = img[targetSize * i : targetSize * i + targetSize, width - targetSize: width, :]
                                    elif (i+1)*targetSize > height and (j+1)*targetSize < width:
                                        temp_img = img[height - targetSize: height, targetSize * j : targetSize * j + targetSize, :]
                                    else:
                                        temp_img = img[height - targetSize: height,  width - targetSize : width, :]
                                    subImg_num_3 = str(subImg_num).zfill(3)
                                    tempName = os.path.join(dir_name,filenames[num][0:-4] + subImg_num_3 +'_.npy')
                                    np.save(tempName,temp_img)
                                    subImg_num +=1


##                            # 新文件名字
##                            new_filename = os.path.join(dir_name,filenames[num][0:-4] + '_new.tif')
##                            # 保存归一化文件
##                            mping.imsave(new_filename,image/255)


                    # 处理label数据
                    if j == type2 == 'label':
                        filenames = []
                        for file in os.listdir(path_2):
                            filenames.append(file)

                        # 遍历读取数据
                        for num in range(len(filenames)):
                            label = mping.imread(os.path.join(path_2,filenames[num]))
                            print(os.path.join(path_2,filenames[num]))
                            new_label = np.zeros((label.shape[0],label.shape[1]))
                            

                            # 处理label数据
                            R = [0,150,150,200,150,150,250,200,200,250,200,250,0,0,0,0]
                            G = [200,250,200,0,0,150,200,200,0,0,150,150,0,150,200,0]
                            B = [0,0,150,200,250,250,0,0,0,150,150,150,200,200,250,0]
                            r = []
                            for one_band in range(len(R)):
                                r.append(R[one_band] + G[one_band]/10 + B[one_band]/100)

                            gray_label = label[::,::,0] + label[::,::,1]/10 + label[::,::,2]/100

                            # 为label数据做标记（0-15）
##                            0-水田；    1-水浇地；  2-旱耕地；  3-园地；    4-乔木林地；5灌木林地；6-天然草地
##                            7-人工草地；8-工业用地；9-城市住宅；10-村镇住宅；11-运输交通；12-河流；13-湖泊；14-坑塘；15-其他类别
                            for class_num in range(len(r)):
                                new_label[gray_label == r[class_num]] += class_num
                            

                            # 创建路径
                            dir_name = os.path.join(path_1,'pro_label')
                            if not os.path.exists(dir_name):
                                os.mkdir(dir_name)



                            # 遍历裁剪
                            targetSize = 256
                            height ,width= new_label.shape[0], new_label.shape[1]
                            rows , cols = height//targetSize+1, width//targetSize+1
                            subImg_num = 0
                            for i in range(rows):
                                for j in range(cols):
                                    if (i+1)*targetSize < height and (j+1)*targetSize < width:
                                        temp_img = new_label[targetSize * i : targetSize * i + targetSize, targetSize * j : targetSize * j + targetSize]
                                    elif (i+1)*targetSize < height and (j+1)*targetSize > width:
                                        temp_img = new_label[targetSize * i : targetSize * i + targetSize, width - targetSize: width]
                                    elif (i+1)*targetSize > height and (j+1)*targetSize < width:
                                        temp_img = new_label[height - targetSize: height, targetSize * j : targetSize * j + targetSize]
                                    else:
                                        temp_img = new_label[height - targetSize: height,  width - targetSize : width]
                                    subImg_num_3 = str(subImg_num).zfill(3)
                                    tempName = os.path.join(dir_name,filenames[num][0:-4] + subImg_num_3 +'_.npy')
                                    np.save(tempName,temp_img)
                                    subImg_num +=1

##                            # 新文件名字
####                            new_filename = os.path.join(dir_name,filenames[num][0:-4] + '_new.tif')
##                            new_filename = os.path.join(dir_name,filenames[num][0:-4] + '_new.npy')
##                            # 保存处理后label文件
####                            scipy.misc.imsave(new_filename, new_label)
##                            # 数据保存为npy格式
##                            np.save(new_filename,new_label)
                            





# 归一化train/image
##a = read_image(type1 = 'train',type2 = 'image')
# 对label处理
##b = read_image(type1 = 'train',type2 ='label')


# 处理val/image
##a = read_image(type1 = 'val',type2 = 'image')
# 对label处理
b = read_image(type1 = 'val',type2 ='label')


