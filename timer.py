import time
import platform


class Timer:
    def __init__(self):
        self.time = []
        self.name = []

    def initialize(self):
        self.time = [time.time()]
        self.name = ['start']

    def check(self, name):
        self.time.append(time.time())
        self.name.append(name)

    def get_result_as_text(self):
        total_time = self.time[-1] - self.time[0]

        # fps
        result = f'system: {platform.system()} / {platform.processor()}'
        result += f'\nfps: {1/total_time:.3f} (time: {total_time*1000:.2f}ms)'

        # remaining times
        for i in range(1, len(self.time)):
            time_now = self.time[i]
            time_prev = self.time[i - 1]
            interval = (time_now - time_prev) * 1000  # in ms
            name = self.name[i]

            result += f'\n{name}: {interval:.1f}ms'

        return result