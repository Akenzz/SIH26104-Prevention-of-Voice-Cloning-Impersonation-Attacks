with open("pipeline.py", "r") as f:
    lines = f.readlines()

process_func = lines[217:273]
class_start = lines[:37]
class_end = lines[37:217] + lines[273:]

with open("pipeline.py", "w") as f:
    f.writelines(class_start + process_func + class_end)
