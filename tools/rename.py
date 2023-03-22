import xlrd
import os

def excel_data(file='/home/ubuntu7/wlj/dataset/Youtute-8/VideoEmotionDataset-TrainTestSplits.xlsx'):
    try:
        # 打开Excel文件读取数据
        data = xlrd.open_workbook(file)
        # 获取第一个工作表
        table = data.sheet_by_index(0)
        # 获取行数
        nrows = table.nrows
        # 获取列数
        ncols = table.ncols
        # 定义excel_list
        excel_list = []
        for row in range(2, nrows):
            file_name=[]
            for col in range(0,2):
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

def rename_file(parent_path,file_list):
    print(parent_path)
    for new_name,old_name in file_list:
       old_name=os.path.join(parent_path,old_name)
       new_name=old_name.split('/')[-3]+"/"+old_name.split('/')[-2]+"/"+new_name+".mp4"
       if os.path.exists(old_name):
           new_name=os.path.join(parent_path,new_name)
           print(old_name,new_name)
           os.rename(old_name,new_name)
       else:
           pass
           #print(old_name)

if __name__ == "__main__":
    list = excel_data()
    print(len(list))
    print(list[0])
    parent_path="/home/ubuntu6/wlj/dataset/Ekman6"
    rename_file(parent_path,list)
    '''for i,j in list:
        print(i,j)'''