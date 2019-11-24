import csv
from os import path
from datetime import datetime

class Metrics:
    def __init__(self):
        self.filename = 'metrics-' + str(datetime.now()).replace(' ', '-').replace(':', '-') + '.csv'
        with open(self.filename, mode='w') as metrics_file:
            self.metrics_writer = csv.writer(metrics_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            self.metrics_writer.writerow(['datetime', 'step', 'x', 'y', 'angle', 'speed', 'steering',
                                          'center_dist', 'center_angle', 'reward', 'total_reward'])

    def record(self, step, x, y, angle, speed, steering, center_dist, center_angle, reward, total_reward):
        now = str(datetime.now())
        #print({now, step, speed, steering, center_dist, center_angle, reward, total_reward})

        with open(self.filename, mode='a') as metrics_file:
            self.metrics_writer = csv.writer(metrics_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            self.metrics_writer.writerow([now, step, x, y, angle, speed, steering,
                                          center_dist, center_angle, reward, total_reward])
