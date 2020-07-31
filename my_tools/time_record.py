# stephen add to record get_item time
import time


class Time_record:
    def __init__(self, key_name,  record_file = None, is_print = True):
        '''
        key_name(list of string):  contain record field in turns    
        
        '''
        self._field_len = len(key_name)
        self._key_name = key_name
        self._record_file = record_file
        self.is_print = is_print
        self.reset()

        
        
    def reset(self):
        self._process_time = [0 for i in range(self._field_len)]
        self._time_stamp = [0 for i in range(self._field_len + 1)]
        self._index = 0
        self._total_time = 0


    def start_time(self):
        self._time_stamp[0] = time.time()
        self._index = 0

    def record(self):
        self._index = self._index + 1
        assert self._index <= self._field_len
        self._time_stamp[self._index] = time.time()
    
    def accumulate(self):
        for i in range(self._field_len):
            self._process_time[i] = self._process_time[i] + (self._time_stamp[i + 1] - self._time_stamp[i])
        # self._total_time = self._total_time + (self._time_stamp[self._field_len] - self._time_stamp[0])
    def single_time(self):
        print('single time: ')
        for i in range(self._field_len):
            print('\033[1;37;41m {:15}:\t{:6.4f} \033[0m'.format(self._key_name[i], self._time_stamp[i + 1] - self._time_stamp[i]))
    def statistic(self, is_print = True):
        for i in range(self._field_len):
            self._total_time = self._total_time + self._process_time[i]
            if self.is_print:
                print('\033[1;37;41m {:15}:\t{:6.4f} \033[0m'.format(self._key_name[i], self._process_time[i]))
        if self.is_print:
            print('\033[1;37;41m total_time:\t{:6.4f}\033[m'.format(self._total_time))
        if(self._record_file is None):
            return
        with open(self._record_file, 'a+') as f:
            f.write("record time:\t{0}\n".format(time.ctime()))
            for i in range(self._field_len):
                f.write('{:15}:\t{:6.4f}\n'.format(self._key_name[i], self._process_time[i]))
            f.write('total_time:\t{:6.4f}\n'.format(self._total_time))
            f.write(''.center(70, '-') + '\n')

        
    def record_to_file(self):
        if(self._record_file is None):
            return
        with open(self._record_file, 'a+') as f:
            f.write("record time:\t{0}\n".format(time.ctime()))
            for i in range(self._field_len):
                f.write('{:15}:\t{:6.4f}\n'.format(self._key_name[i], self._process_time[i]))
            f.write('total_time:\t{:6.4f}\n'.format(self._total_time))
            f.write(''.center(70, '-') + '\n')
    

if __name__ == '__main__':
    time_name = ['start', 'first', 'second', 'last']
    t1 = Time_record(time_name)
    t1.start_time()
    for j in range(10000):
        t1.start_time()
        for i in range(len(time_name)):
            t1.record()
        t1.accumulate()
    t1.statistic()
    t1.record_to_file()
    