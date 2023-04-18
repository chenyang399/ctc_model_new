# ```python
# /path/to/audio/file0.wav,我爱你
# /path/to/audio/file1.wav,你爱我吗
# ...
# ```


# 生成index文件
index=0
train_w=open("data/data_aishell/train-sort.manifest","a")
dev_w=open("data/data_aishell/dev.manifest","a")
with open("data/data_aishell/transcript/aishell_transcript_v0.8.txt","r") as f:
        for line in f:
            print('运行到第'+str(index)+'行')
            index+=1
            if index<130398:
                continue

            parts = line.split()
            file_index=' '.join(parts[0:1])
            text = ' '.join(parts[1:])
            # print(text)
            # 消除空格
            text_noblank=str()
            for i in text:
                if i !=' ':
                    text_noblank+=i
            # print(text_noblank)
            # 接下来要找到文件路径
            # 遍历所有的sph。list寻找

            # 排除已经处理过的

            sphlist=open("/home/chenyang/chenyang_space/ctc_model_new/conf/sph.flist","r")
            for line_x in sphlist:
                #  print("        "+line_x)
                 parts_x = line_x.split("/")
                #  print("        "+str(parts_x))
                 file_index_x=(parts_x[-1])
                #  print("        the file index in sphlist is "+file_index_x[:-5])
                #  print("        the directory is "+parts_x[-3])
                 if parts_x[-3]=="train":
                    if file_index_x[:-5]==file_index:
                        # print("           找到了路径"+file_index_x)
                        #   放到需要保存的index文件里面去，这里暂时不区分train和dev
                        train_w.write(line_x[:-1]+','+text_noblank+'\n')
                        train_w.flush()
                        break
                
                 if  parts_x[-3]=="dev":
                    if file_index_x[:-5]==file_index:
                        # print("           找到了路径"+file_index_x)
                        #   放到需要保存的index文件里面去，这里暂时不区分train和dev
                        dev_w.write(line_x[:-1]+','+text_noblank+'\n')
                        dev_w.flush()
                        break
            
                 
            sphlist.close()
        dev_w.close()
        train_w.close()


                 

