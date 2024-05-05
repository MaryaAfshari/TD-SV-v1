output_file_path = '../../../ResultFile1-24-4-2024/train_labels.txt'
#check the number of lines written in 
line_count = 0
with open(output_file_path, 'r') as file:
    for line in file:
        line_count += 1

print(f"The file in {output_file_path} has {line_count} lines.")
