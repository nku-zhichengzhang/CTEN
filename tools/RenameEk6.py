from glob import glob
import os

filelist=glob("/home/ubuntu6/wlj/dataset/Ekman6-jpg/*/*")

cnt=1

filelist.sort()

print(filelist)

class_dict={
    'anger':1,
    'disgust':2,
    'fear':3,
    'joy':4,
    'sadness':5,
    'surprise':6
}

with open("/home/ubuntu6/wlj/code/Video/VAANet-master/tools/annotations/ek6/trainlist01.txt",'w') as f:

    for file in filelist:
        name=file.split('/')[-2]+"/"+file.split('/')[-1]
        if cnt%2==1:
            f.write(name+" "+str(class_dict[file.split('/')[-2]])+"\n")
        cnt+=1