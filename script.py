from os import system
execution_dict = {
                    # "afraid_6":["MS 0", "MS 0,1", "MS 1"],
                    # "braunfels_1":["MS 0", "MS 0,1", "MS 0,1,2", "MS 0,2", "MS 1", "MS 1,2", "MS 2"],
                    "carol_2":["MS 0", "MS 0,1", "MS 0,1,2", "MS 0,2", "MS 1", "MS 1,2", "MS 2"]
                 }

for video, shot_specs in execution_dict.items():
    for shot_spec in shot_specs:
        print("*** %s %s started ***" % (video, shot_spec))
        system("python multi-clip.py  %s \"%s\"" % (video, shot_spec))