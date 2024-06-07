def check_phrase_labels(train_list):
    lines = open(train_list).read().splitlines()
    lines = lines[1:]  # Skip the header row
    all_integers = True
    for line in lines:
        try:
            phrase_label = int(line.split()[2])  # Trying to convert to integer
        except ValueError:
            all_integers = False
            print(f"Non-integer phrase ID found: {line.split()[2]}")
            break
    if all_integers:
        print("All phrase labels are integers.")
    else:
        print("There are non-integer phrase labels in the file.")

if __name__ == "__main__":
    train_list = "../../../../../mnt/disk1/data/TdSVC2024/task1/docs/train_labels.txt"  # Replace with the path to your train list file
    check_phrase_labels(train_list)
