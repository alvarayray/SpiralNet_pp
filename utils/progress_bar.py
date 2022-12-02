from __future__ import print_function
import time, sys
# utils is a custom module in /home/ben/Developer
# make sure this path is included in PYTHONPATH when using this module

class ProgressBarWrapper():
    '''
    Example:
    >>> import time
    >>> for i in ProgressBarWrapper(range(10)):
    >>>     time.sleep(0.2)

    Example:
    >>> import time
    >>> for i in ProgressBarWrapper(range(10), 'Blue'):
    >>>     time.sleep(0.2)

    Available Color:\033[95m PURPLE \033[0m, \033[96m DARKCYAN \033[0m, \033[36m CYAN \033[0m, \033[94m BLUE \033[0m, \033[92m GREEN \033[0m, \033[93m YELLOW \033[0m, \033[91m RED \033[0m, \033[1m BOLD \033[0m, \033[4m UNDERLINE \033[0m

    Note: Don't print anything in the loop. Or the progress bar will be meshed up.

    IMPORTANT:
        DEPENDENCY: utils is a custom module in /home/ben/Developer.
        Make sure this path is included in PYTHONPATH when using this module.
    '''

    MAX_UPDATE_RATE = 1000

    def __init__(self, a_list, style='BLUE', require_timing=True, dense_update=False):
        self.my_list = a_list
        self.current = 0
        self.require_timing = require_timing
        self.dense_update = dense_update
        if self.require_timing:
            self.previous_loop_start_time = None
            self.first_loop_start_time = None
        self.style = style

        # only calculate once to save computation
        self.my_list_length = len(a_list)
        self._update_interval = max(1,int(self.my_list_length/self.MAX_UPDATE_RATE))

    def __iter__(self):
        return self

    def next(self):

        if self.dense_update or self.current % self._update_interval == 0 or self.current==self.my_list_length:
            self.update_ui()

        if self.current >= self.my_list_length:
            raise StopIteration
        self.current += 1
        return self.my_list[self.current-1]

    def __next__(self):
        return self.next()

    def update_ui(self):
        info = '{}/{}'.format(self.current, self.my_list_length)

        if self.require_timing:
            now = time.time()
            if self.current == 0:
                self.first_loop_start_time = now
                self.previous_loop_start_time = now
            else:
                time_passed = now - self.first_loop_start_time
                time_to_go = (now - self.first_loop_start_time)/self.current*(self.my_list_length - self.current)
                time_of_current_iter = (now - self.previous_loop_start_time)/self._update_interval
                info += ' [{}<{}, {:.2f}it/s]'.format(
                    sec_in_float2time_str(time_passed),
                    sec_in_float2time_str(time_to_go),
                    1/time_of_current_iter
                )

        printProgressBar(self.current, self.my_list_length, prefix = 'Progress:', suffix = info, length = 50, style=self.style)
        self.previous_loop_start_time = time.time()


# Print iterations progress
styles = {
    'PURPLE' : '\033[95m',
    'DARKCYAN' : '\033[96m',
    'CYAN' : '\033[36m',
    'BLUE' : '\033[94m',
    'GREEN' : '\033[92m',
    'YELLOW' : '\033[93m',
    'RED' : '\033[91m',
    'BOLD' : '\033[1m',
    'UNDERLINE' : '\033[4m',
    'END' : '\033[0m'
}
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = u"\u2588", style=None):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)

    if style is not None:
        bar = styles[style] + bar + styles['END']

    sys.stdout.write('\r')
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()

    # Print New Line on Complete
    if iteration == total:
        print()

def sec_in_float2time_str(sec):
    m = int(sec/60)
    sec = sec%60
    return "{:02}:{:02.0f}".format(m, sec)

