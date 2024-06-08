#Ya Rabal alamin
#Date: 18.3.1403 Khordad mah
#Date: 7.6.2024 June
#Date: 8.6.2024 June/ 19.3.1403 Khordad 
#Author: Maryam Afshari -Iranian


def check_phrase_labels(train_list):
    """
    This function checks if all phrase labels in the train list are integers.
    If it finds a non-integer phrase label, it prints the label and exits the check.
    """
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

def inspect_train_list_for_ft(train_list):
    """
    This function inspects the train list and prints lines that contain the phrase ID "FT".
    """
    # Inspect the train list entries
    with open(train_list, 'r') as f:
        lines = f.readlines()

    # Print lines containing "FT"
    for line in lines:
        if "FT" in line:
            print(line)

if __name__ == "__main__":
    train_list = "../../../../../mnt/disk1/data/TdSVC2024/task1/docs/train_labels.txt"  # Replace with the path to your train list file
    check_phrase_labels(train_list)
    inspect_train_list_for_ft(train_list)
