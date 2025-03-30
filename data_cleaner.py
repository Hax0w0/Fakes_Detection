# Class: Northwestern CS 461 Winter 2025
# ---------------------------------------

# Professor: David Demeter
# ---------------------------------------

# Contributers:
# ---------------------------------------
#   Raymond Gu
#   Maanvi Sarwadi

import re
import pandas as pd

def extract_initial_description(bio_text):
    
    pattern = r'= (.*?) =\s*([^=].*?)\s*(?=(?:==|<end_bio>))'
    match = re.search(pattern, bio_text, re.DOTALL)

    if match:
        name = match.group(1).strip()
        description = match.group(2).strip()
        description = description.replace('\n', ' ')

        if not name or '\n' in name:
            return None
        else:
            return description
    
    return None

def parse_file(file_name, is_fake):

    with open(file_name, 'r', encoding='utf-8') as file:
        bios = file.read().split('<start_bio>')
    
    parsed_data = []
    for bio in bios:
        description = extract_initial_description(bio)
        label = "fake" if is_fake else "real"
        if description:
            parsed_data.append((description, label))
    
    return parsed_data

def format_data():

    real_data = [r"./Original Data/real.train.txt",
                r"./Original Data/real.test.txt",
                r"./Original Data/real.valid.txt"]

    fake_data = [r"./Original Data/fake.train.txt",
                r"./Original Data/fake.test.txt",
                r"./Original Data/fake.valid.txt"]

    real = []
    for file in real_data:
        real.extend(parse_file(file, False))

    fake = []
    for file in fake_data:
        fake.extend(parse_file(file, True))

    real_df = pd.DataFrame(real, columns=["Description", "Label"])
    fake_df = pd.DataFrame(fake, columns=["Description", "Label"])

    print("  Format_Data Output:")
    print("\tReal Entries: ", len(real))
    print("\tFake Entries: ", len(fake))
    print("")

    return real_df, fake_df

def main():

    real_df, fake_df = format_data()

    train_size = 7824
    valid_size = 978
    test_size = 978

    # Shuffle dataframes before splitting
    real_df = real_df.sample(frac=1, random_state=42).reset_index(drop=True)
    fake_df = fake_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Splitting real data
    train_real_df = real_df.iloc[:train_size]
    valid_real_df = real_df.iloc[train_size:train_size + valid_size]
    test_real_df = real_df.iloc[train_size + valid_size:train_size + valid_size + test_size]

    # Splitting fake data
    train_fake_df = fake_df.iloc[:train_size]
    valid_fake_df = fake_df.iloc[train_size:train_size + valid_size]
    test_fake_df = fake_df.iloc[train_size + valid_size:train_size + valid_size + test_size]

    # Join the datasets
    train_df = pd.concat([train_real_df, train_fake_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    valid_df = pd.concat([valid_real_df, valid_fake_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.concat([test_real_df, test_fake_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Print the data about each dataset:
    print("  Files Produced By Main:")
    print("\tTraining File: ", len(train_df))
    print("\t   Number of Real Entries: ", len(train_real_df))
    print("\t   Number of Fake Entries: ", len(train_fake_df))
    print("")

    print("\tValidation File: ", len(valid_df))
    print("\t   Number of Real Entries: ", len(valid_real_df))
    print("\t   Number of Fake Entries: ", len(valid_fake_df))
    print("")

    print("\tTest File: ", len(valid_df))
    print("\t   Number of Real Entries: ", len(test_real_df))
    print("\t   Number of Fake Entries: ", len(test_fake_df))

    # Save the datasets into EXCEL files
    train_df.to_excel(r"./Cleaned Data/train.xlsx", index=False)
    valid_df.to_excel(r"./Cleaned Data/valid.xlsx", index=False)
    test_df.to_excel(r"./Cleaned Data/test.xlsx", index=False)

main()