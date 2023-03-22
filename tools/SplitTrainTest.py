import xlrd
import os
import glob

def excel_data(file='/home/ubuntu6/wlj/Youtube-8/VideoEmotionDataset-TrainTestSplits.xlsx'):
    try:
        # 打开Excel文件读取数据
        data = xlrd.open_workbook(file)
        # 获取第一个工作表
        table = data.sheet_by_index(1)
        # 获取行数
        nrows = table.nrows
        # 获取列数
        ncols = table.ncols
        # 定义excel_list
        excel_list = []
        for row in range(2, nrows):
            file_name=[]
            for col in range(2,4):
                # 获取单元格数据
                cell_value = table.cell(row, col).value
                if isinstance(cell_value,float):
                    cell_value=str(int(cell_value))
                # 把数据追加到excel_list中
                file_name.append(cell_value)
            excel_list.append(file_name)
        return excel_list
    except:
        print("error")

def GetFileList():
    filelist=glob.glob("/home/ubuntu6/wlj/VideoEmotion/VAANet-master/data/Youtube-8-jpg/*/*")
    file_dict={}
    for file in filelist:
        #print(file)
        class_name=file.strip().split("/")[-2]
        file_name=file.strip().split("/")[-1]
        file_dict[file_name]=class_name+"/"+file_name
        #print(class_name,file_name)
    return file_dict

def GetClassDict(path="/home/ubuntu6/wlj/VideoEmotion/VAANet-master/tools/annotations/ve8/classInd.txt"):
    class_dict={}
    with open(path,'r') as f:
        lines=f.readlines()
        for line in lines:
            label_num=line.strip().split(' ')[0]
            label_name=line.strip().split(" ")[-1]
            class_dict[label_name]=label_num
    return class_dict

def GetTrainList(train_list,file_dict,csv_dir_path,split_index):
    train_csv_path = os.path.join(csv_dir_path, 'trainlist0{}.txt'.format(split_index))
    class_dict=GetClassDict()
    print(class_dict)
    with open(train_csv_path, 'w') as f:
        for i,_ in train_list:
            if i!="":
                print(file_dict[i]+" ",class_dict[file_dict[i].split('/')[0]]+"\n")
                f.write(file_dict[i]+" ")
                f.write(class_dict[file_dict[i].split('/')[0]]+"\n")

def GetTestList(test_list,csv_dir_path,split_index):
    test_csv_path = os.path.join(csv_dir_path, 'testlist0{}.txt'.format(split_index))
    class_dict = GetClassDict()
    print(class_dict)
    with open(test_csv_path, 'w') as f:
        for _, i in test_list:
            if i != "":
                print(file_dict[i] + " ", class_dict[file_dict[i].split('/')[0]] + "\n")
                f.write(file_dict[i] + "\n")

if __name__ == "__main__":
    list = excel_data()
    print(len(list))
    print(list[0])
    cnt=2
    file_dict=GetFileList()
    csv_dir_path="/home/ubuntu6/wlj/VideoEmotion/VAANet-master/data"
    GetTrainList(list,file_dict,csv_dir_path,cnt)
    GetTestList(list,csv_dir_path,cnt)
    #parent_path="/home/ubuntu6/wlj/Youtube-8/"