import os

logs_path = "testing_logs"
for filename in os.listdir(logs_path):
    if filename[-8:] == "_err.log":
        f_path = os.path.join(logs_path, filename)
        with open(f_path, 'r') as f:
            file_content = f.read().lower()
            if "error" in file_content:
                print("Error. See file {}".format(filename))


