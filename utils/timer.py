import time
import platform


class Timer:
    def __init__(self):
        self.time = []
        self.name = []
        self.history = {}

    def initialize(self):
        for i in range(1, len(self.time)):
            name = self.name[i]
            interval = self.time[i] - self.time[i - 1]

            try:
                self.history[name].append(interval)
            except KeyError:
                self.history[name] = [interval]

        self.time = [time.time()]
        self.name = ["start"]

    def check(self, name):
        self.time.append(time.time())
        self.name.append(name)

    def get_result_as_text(self):
        interval = self.time[-1] - self.time[0]

        # fps
        result = f"system: {platform.system()} / {platform.processor()}"
        result += f"\nfps: {1/interval:.3f} (time: {interval*1000:.2f}ms)"

        # remaining times
        for i in range(1, len(self.time)):
            name = self.name[i]
            time_now = self.time[i]
            time_prev = self.time[i - 1]
            interval = (time_now - time_prev) * 1000  # in ms

            result += f"\n{name}: {interval:.1f}ms"

        return result
