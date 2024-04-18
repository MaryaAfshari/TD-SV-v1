#In the name of GOD
#Function Definition ---------
def split_and_print(line):
    # Split the line into different parts/columns using spaces as the delimiter
    parts = line.split()

    # Extract and print each part
    for part in parts:
        print(part)
#30-march-2023 - shanbe-- 11-farvardin-1403
#4-4-2024 - panjshanbe - 16-farvardin-1403
#7-4-2024 - Yekshanbe -
#8-4-2024 - Doshanbe -
#10-4-2024- Chaharshanbe-22-Farvardin
#15-4-2024- Dooshanbe - 27-farvardin-1403
#17-4-2024 - Chaharshanbe -29-farvardin-1403
#18-4-2024 - Panjshanbe - 30-farvardin-1403
print("Hello Practical Speech word date :4-April-2024")
print("Hello Practical Speech word date :10-April-2024")
print("Hello Practical Speech word date :15-April-2024")
print("Hello Practical Speech word date :17-April-2024")
print("Hello Practical Speech word date :18-April-2024 = 30-farvardin-1403")
#code address : /code1-29-3-2024/TD-SV-v1/ECAPA-TDNN-main
# trn address file: /mnt/disk1/data/DeepMine/key/text-dependent/trn/ENG/male/100-spk/3-sess
# ndx address file: /mnt/disk1/data/DeepMine/key/text-dependent/ndx/ENG/male/100-spk/3-sess
trn_dev_addr = "../../../../../mnt/disk1/data/DeepMine/key/text-dependent/trn/ENG/male/100-spk/3-sess/dev.trn"
ndx_dev_TC_addr = "../../../../../mnt/disk1/data/DeepMine/key/text-dependent/ndx/ENG/male/100-spk/3-sess/dev_TC.ndx" #Target Correct

with open(trn_dev_addr, 'r') as file:
    line_count = 0
    for line in file:
        print(line)
        split_and_print(line)
        line_count += 1
        if line_count == 2:
            break


with open(ndx_dev_TC_addr, 'r') as file:
    line_count = 0
    for line in file:
        print(line)
        split_and_print(line)
        line_count += 1
        if line_count == 2:
            break