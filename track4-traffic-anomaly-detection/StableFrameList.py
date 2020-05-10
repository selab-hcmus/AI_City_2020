import json
import numpy
import Config

class StableFrameList:
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.stableList = json.load(f)

        self.preprocessing()

    def preprocessing(self):
        #join interval
        pass

    def __getitem__(self, key):
        return self.stableList[str(key)]

if __name__ == '__main__':
    list = StableFrameList(Config.data_path + '/unchanged_scene_periods.json')
    for i in range(1, 101):
        print(i, len(list[i]))