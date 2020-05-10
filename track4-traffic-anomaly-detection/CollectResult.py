import numpy as np
import Config

g = open(Config.output_path + '/result_all.txt', 'w')
for i in range(1, 101):
    f = open(Config.output_path + '/' + str(i) + '/anomaly_events.txt', 'r')
    c = f.read()
    g.write(c)
    f.close()
g.close()
