#In the name of GOD
#30-march-2023 - shanbe-- 11-farvardin-1403
#4-4-2024 - panjshanbe - 16-farvardin-1403
#7-4-2024 - Yekshanbe -
#8-4-2024 - Doshanbe -
#10-4-2024- Chaharshanbe-22-Farvardin
#15-4-2024- Dooshanbe - 27-farvardin-1403
#17-4-2024 - Chaharshanbe -29-farvardin-1403
print("Hello Practical Speech word date :4-April-2024")
print("Hello Practical Speech word date :10-April-2024")
print("Hello Practical Speech word date :15-April-2024")
print("Hello Practical Speech word date :17-April-2024")
#code address : /code1-29-3-2024/TD-SV-v1/ECAPA-TDNN-main
# trn address file: /mnt/disk1/data/DeepMine/key/text-dependent/trn/ENG/male/100-spk/3-sess
# ndx address file: /mnt/disk1/data/DeepMine/key/text-dependent/ndx/ENG/male/100-spk/3-sess
trn_dev_addr = "../../../../../mnt/disk1/data/DeepMine/key/text-dependent/trn/ENG/male/100-spk/3-sess/dev.trn"
with open(trn_dev_addr, 'r') as file:
    line_count = 0
    for line in file:
        print(line)
        line_count += 1
        if line_count == 2:
            break