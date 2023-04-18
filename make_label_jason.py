# coding=utf-8
import json
def remove_duplicate_chinese_chars(s):
    """
    去除字符串s中的重复中文字符
    """
    # 将字符串s转换为UTF-8编码的字节数组
    s_bytes = s.decode('utf-8')

    # 将字节数组转换为set集合
    s_set = set(s_bytes)

    # 将set集合转换为字符串
    s_new = b''.join(s_set).encode('utf-8')

    return s_new


result=str()
index=0
with open("data/data_aishell/transcript/aishell_transcript_v0.8.txt","r") as f:
        for line in f:
            print('运行到第'+str(index)+'行')
            index+=1
            parts = line.split()
            text = ' '.join(parts[1:])
            # text.replace(" ","")
            # print(text)
            result+=text
            # print(result)
            # 接下来把这些文字放到一个里面，然后去重



# 去重
result_new = remove_duplicate_chinese_chars(result)
#去空格
# print(result_new)
b=str()

for i in result_new:
    if i !=' ':
        b+=i
# print(b)

# 把结果写到jason文件之中
            
          
file = open('data/data_aishell/labels.json','w+')
#向文件中输入字符串


# file.close()

b='#'+b     
print(b)
b=json.dumps(b)
print(b)
file.write(b)
file.close()    