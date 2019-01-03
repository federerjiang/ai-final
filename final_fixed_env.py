import numpy as np
import random
import datetime
import copy

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42

BITRATE_LEVELS = 2

lamda = 1
default_quality = 0
latency_threshold = 7
skip_add_frame = 100
ADD_FRAME = 0 


class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, random_seed=RANDOM_SEED, logfile_path='./', VIDEO_SIZE_FILE ='./video_size_', Debug = True):
        assert len(all_cooked_time) == len(all_cooked_bw)
         
        if Debug:
            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') 
            self.log_file = open(logfile_path +"log." +current_time, "w")

        self.video_size_file = VIDEO_SIZE_FILE
        self.Debug = Debug

        #np.random.seed(random_seed)

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.time = 0
        self.down_time = 0
        self.play_time = 0
        self.newest_frame = [0,0]
        self.all_time0 = 0
        self.all_time1 = 0
        self.last_video_chunk0 = 0
        self.last_video_chunk1 = 0

        self.video_chunk_counter = [0,0]
        self.last_bit_rate = 0

        self.buffer_size = 0

        # pick a random trace file
        self.trace_idx = 0
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]
        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.decision = False
        self.buffer_status = True

        self.skip_time_frame = 100000000
        
        #self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]
        self.video_size = {}  # in bytes
        self.cdn_arrive_time = {}
        self.frame_time_len = {}
        self.gop_flag = {}

        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            self.cdn_arrive_time[bitrate] = []
            self.frame_time_len[bitrate] = []
            self.gop_flag[bitrate] = []
            cnt = 0
            with open(self.video_size_file + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(float(line.split()[1]))
                    self.frame_time_len[bitrate].append(1.0/float(line.split()[3]))
                    self.gop_flag[bitrate].append(int(float(line.split()[2])))
                    self.cdn_arrive_time[bitrate].append(float(line.split()[0]))
                    cnt += 1
        self.latency = self.frame_time_len[0][0] 

    def get_trace_id(self):
        return self.trace_idx

    def set_trace_id(self,TRACE_ID=1):
        self.trace_idx = TRACE_ID 

    def get_video_trace(self):
        return self.video_size_file

    def set_video_trace(self,VIDEO_SIZE_FILE="./video_size_"):
        self.video_size_file = VIDEO_SIZE_FILE 

    def get_video_frame(self, bit_rate, target_buffer):

        assert bit_rate >= 0
        assert bit_rate < BITRATE_LEVELS
        
        # get to quick play and slow play
        quick_play_bound = target_buffer + 0.1
        slow_play_bound = target_buffer * 2.0 / 3

        #  self.down_time ---->  self.video_chunk_counter
        self.all_time0 += sum(self.frame_time_len[0][self.last_video_chunk0:self.video_chunk_counter[0]])
        self.all_time1 += sum(self.frame_time_len[1][self.last_video_chunk1:self.video_chunk_counter[1]])
        while 1:
            if self.all_time0 < self.down_time:
                self.all_time0 += self.frame_time_len[0][self.video_chunk_counter[0]]
                self.video_chunk_counter[0] += 1
            if self.all_time1 < self.down_time:
                self.all_time1 += self.frame_time_len[1][self.video_chunk_counter[1]]
                self.video_chunk_counter[1] += 1
            if self.all_time0 >= self.down_time and self.all_time1 >= self.down_time:
                break
        self.last_video_chunk0 = copy.copy(self.video_chunk_counter[0])
        self.last_video_chunk1 = copy.copy(self.video_chunk_counter[1])
          
        #print(self.video_chunk_counter ,self.down_time, sum(self.frame_time_len[0][:self.video_chunk_counter[0] ]))
        # calculate the quality
        if self.gop_flag[bit_rate][self.video_chunk_counter[bit_rate] ] == 1 and self.last_bit_rate != bit_rate:
             quality = bit_rate 
             #print("yao qie huan le ", self.last_bit_rate, quality)
             self.last_bit_rate = bit_rate
             switch = 1
        else:
             quality = self.last_bit_rate
             switch = 0
        # 
        if quality == 0:
             other_quality = 1
        else:
             other_quality = 0
        # Initial the Settings
        self.decision = False                                                 # GOP_flag
        video_frame_size = self.video_size[quality][self.video_chunk_counter[quality]] # Data_size
        FRAME_TIME_LEN = self.frame_time_len[quality][self.video_chunk_counter[quality]] # frame time len
        cdn_rebuf_time = 0                                                    # CDN_rebuf_time
        rebuf = 0                                                             # rebuf_time
        end_of_video = False                                                  # is trace time end
        duration = 0                                                          # this loop 's time len
        current_new = 0
        global ADD_FRAME
 
        # This code is check the quick play or slow play
        # output is the play_weight
        if self.buffer_size > quick_play_bound :
            #quick play
            play_duration_weight = 1.05
        elif  self.buffer_size < slow_play_bound :
            #slow play
            play_duration_weight = 0.95
        else:
            play_duration_weight = 1
        # This code is check Is the cdn has the frame in this time
        # self.time means the real time
        # self.cdn_arrive_time means the time the frame came
        if self.time < self.cdn_arrive_time[quality][self.video_chunk_counter[quality]] and not end_of_video: # CDN can't get the frame
            cdn_rebuf_time = self.cdn_arrive_time[quality][self.video_chunk_counter[quality]] - self.time
            self.newest_frame[quality] = self.video_chunk_counter[quality]
            self.newest_frame[other_quality] = self.video_chunk_counter[other_quality] 
            duration = cdn_rebuf_time
            # if the client means the buffering
            if not self.buffer_status:
                # not buffering ,you can't get the frame ,but you must play the frame

                # the buffer is enough
                if self.buffer_size > cdn_rebuf_time * play_duration_weight:
                    self.buffer_size -= cdn_rebuf_time * play_duration_weight
                    self.play_time += cdn_rebuf_time * play_duration_weight
                    rebuf = 0
                    play_len = cdn_rebuf_time * play_duration_weight 
                # not enough .let the client buffering
                else:
                    self.play_time += self.buffer_size
                    rebuf = cdn_rebuf_time  - (self.buffer_size / play_duration_weight)
                    play_len = self.buffer_size 
                    self.buffer_size = 0
                    self.buffer_status = True
               
            # calculate the play time , the real time ,the latency
                # the normal play the frame

                self.time = self.cdn_arrive_time[quality][self.video_chunk_counter[quality]]
                self.latency = sum(self.frame_time_len[quality][ self.video_chunk_counter[quality]:self.newest_frame[quality]])   + self.buffer_size
            else:
                rebuf = duration
                play_len = 0
                self.time = self.cdn_arrive_time[quality][self.video_chunk_counter[quality]]
                self.latency = sum(self.frame_time_len[quality][ self.video_chunk_counter[quality] : self.newest_frame[quality]])   + self.buffer_size
            # Debug info 
            if self.Debug:
                print("%.4f"%self.time ,
                  "  cdn rebuf %4f"%cdn_rebuf_time, 
                  "~rebuf~~ %.3f"%rebuf,
                  "~dur~~%.4f"%duration,
                  "~id! ", self.video_chunk_counter,
                  "~newest ", self.newest_frame,
                  "~~buffer~~ %4f"%self.buffer_size, 
                  "~~play_time~~%4f"%self.play_time ,
                  "~~quality~ ",quality,
                  "~~down_time~%.4f"%self.down_time,
                  "~~latency~~%4f"%self.latency,"000")
                self.log_file.write("real_time %.4f\t"%self.time +  
                                  "cdn_rebuf%.4f\t"%cdn_rebuf_time + 
                                  "client_rebuf %.3f\t"%rebuf  + 
                                  "download_duration %.4f\t"%duration + 
                                  "download_id %s\t"%str(self.video_chunk_counter) + 
                                  "cdn_newest_frame %s\t"%str(self.newest_frame) + 
                                  "frame_size %.4f\t"%video_frame_size + 
                                  "video_time  %.4f\t"% (self.down_time) +
                                  "client_buffer %.4f\t"%self.buffer_size  +
                                  "play_time %.4f\t"%self.play_time + 
                                  "latency %.4f\t"%self.latency + "000\n")
            cdn_has_frame = []
            if self.newest_frame[quality] - self.video_chunk_counter[quality] > 50:
                 for bitrate in range(BITRATE_LEVELS):
                      cdn_has_frame.append(sum(self.video_size[bitrate][self.video_chunk_counter[bitrate] : self.newest_frame[bitrate]]))
            # Return the loop
            return  [self.time,                             # physical time
                    duration,                               # this loop duration, means the download time
                    0,                                      # frame data size
                    0,                                      # frame time len
                    rebuf,                                  # rebuf len
                    self.buffer_size,                       # client buffer
                    self.latency ,                          # latency
                    self.newest_frame,                      # cdn the newest frame id
                    self.video_chunk_counter,               # download_id
                    cdn_has_frame,                          # CDN_has_frame
                    self.decision,                          # Is_I frame edge
                    quality,                                # real quality
                    self.buffer_status,                     # Is the buffer is buffering
                    0,                                      # calcuate the switch
                    True,                                   # Is the CDN has no frame
                    end_of_video]                           # Is the end of video
        else:
            the_newst_frame = copy.copy(self.video_chunk_counter)
            while(1):
                if self.cdn_arrive_time[quality][the_newst_frame[quality]] < self.time:
                    the_newst_frame[quality] += 1
                if self.cdn_arrive_time[other_quality][the_newst_frame[other_quality]] < self.time:
                    the_newst_frame[other_quality] += 1
                if (self.cdn_arrive_time[quality][the_newst_frame[quality]] >= self.time) and (self.cdn_arrive_time[other_quality][the_newst_frame[other_quality]] >= self.time) :
                    break
                #print(the_newst_frame, self.cdn_arrive_time[quality][the_newst_frame[quality]], self.cdn_arrive_time[other_quality][the_newst_frame[other_quality]], self.time)
            self.newest_frame = the_newst_frame
        # If the CDN can get the frame:
        if int(self.time / 0.5) >= len(self.cooked_bw):
            end_of_video = True
        else:
            throughput = self.cooked_bw[int(self.time / 0.5)]  * B_IN_MB
            #rtt        = self.cooked_rtt[int(self.time / 0.5)]
            duration = float(video_frame_size / throughput)
        # If the the frame is the Gop end ,next will be the next I frame
        if self.gop_flag[quality][self.video_chunk_counter[quality] + 1] == 1:
            self.decision = True 
        #print(self.video_chunk_counter, self.down_time, 
                             #sum(self.frame_time_len[0][:self.video_chunk_counter[0]]),sum( self.frame_time_len[1][:self.video_chunk_counter[1]]), self.last_bit_rate,bit_rate, quality,) 
                             #self.gop_flag[0][self.video_chunk_counter[0] + 1],  self.gop_flag[1][self.video_chunk_counter[1] + 1])
        # If the buffering
        if self.buffer_status and not end_of_video:
            # let the buffer_size to expand to the target_buffer
            self.buffer_size += self.frame_time_len[quality][self.video_chunk_counter[quality]]
            rebuf = duration
            self.time += duration
            self.down_time += self.frame_time_len[quality][self.video_chunk_counter[quality]]
            if self.buffer_size >= target_buffer:
                self.buffer_status = False
            # if the buffer is enough
            #else:
            #    self.down_time += self.frame_time_len[quality][self.video_chunk_counter[quality]]
            # calculate the play time , the real time ,the latency
            self.latency = sum(self.frame_time_len[quality][ self.video_chunk_counter[quality] : self.newest_frame[quality]])   + self.buffer_size
            # Debug Info
            if self.latency > latency_threshold :
                self.skip_time_frame = self.video_chunk_counter[quality]
                if self.newest_frame[quality] >  skip_add_frame + self.video_chunk_counter[quality]:
                    #ADD_FRAME = self.play_time_counter + skip_add_frame - self.video_chunk_counter 
                    ADD_FRAME =  skip_add_frame
                    ADD_FRAME_TIME = sum(self.frame_time_len[quality][self.video_chunk_counter[quality]: self.video_chunk_counter[quality] + ADD_FRAME +1])
                    self.down_time += ADD_FRAME_TIME
                else:
                    ADD_FRAME =  self.newest_frame[quality] - self.video_chunk_counter[quality]
                    ADD_FRAME_TIME = sum(self.frame_time_len[quality][self.video_chunk_counter[quality]: self.video_chunk_counter[quality] + ADD_FRAME +1])
                    self.down_time += ADD_FRAME_TIME
                rebuf += ADD_FRAME * FRAME_TIME_LEN  

                if self.Debug:
                    print("skip events: skip_download_frame, play_frame, new_download_frame, ADD_frame" + str(self.skip_time_frame) + "  " + str(self.video_chunk_counter[quality]) +" " +str(ADD_FRAME) + "\n")
                    self.log_file.write("skip events: skip_download_frame, play_frame, new_download_frame, ADD_frame" + str(self.skip_time_frame) + " " + str(self.video_chunk_counter) +" " +str(ADD_FRAME) + "\n")
            #else:
            #    self.video_chunk_counter[quality] += 1
            if self.Debug:
                 print("%.4f"%self.time ,
                      "  cdn %4f"%cdn_rebuf_time, 
                      "~rebuf~~ %.3f"%rebuf,
                      "~dur~~%.4f"%duration,
                      "~id! ", self.video_chunk_counter,
                      "~newest ", self.newest_frame,
                      "~~buffer~~ %4f"%self.buffer_size, 
                      "~~play_time~~%4f"%self.play_time ,
                      "~~quality~",quality,
                      "~~down_time%.4f"%self.down_time,
                      "~~latency~~%4f"%self.latency,"111")
                 self.log_file.write("real_time %.4f\t"%self.time +  
                                      "cdn_rebuf%.4f\t"%cdn_rebuf_time + 
                                      "client_rebuf %.3f\t"%rebuf  + 
                                      "download_duration %.4f\t"%duration + 
                                      "download_id %s\t"% str(self.video_chunk_counter) + 
                                      "cdn_newest_frame %s\t"% str(self.newest_frame) + 
                                      "frame_size %.4f\t"%video_frame_size + 
                                      "video_time  %.4f\t"% (self.down_time) +
                                      "client_buffer %.4f\t"%self.buffer_size  +
                                      "play_time %.4f\t"%self.play_time + 
                                      "latency %.4f\t"%self.latency + "111\n")
            cdn_has_frame = []
            if self.newest_frame[quality] - self.video_chunk_counter[quality] > 50:
                 for bitrate in range(BITRATE_LEVELS):
                      cdn_has_frame.append(sum(self.video_size[bitrate][self.video_chunk_counter[bitrate] : self.newest_frame[bitrate]]))
            # Return the loop
            return      [self.time,                             # physical time
                        duration,                               # this loop duration, means the download time
                        video_frame_size,                       # frame data size
                        FRAME_TIME_LEN,                         # frame time len
                        rebuf,                                  # rebuf len
                        self.buffer_size,                       # client buffer
                        self.latency ,                          # latency
                        self.newest_frame,                      # cdn the newest frame id
                        (self.video_chunk_counter),             # download_id
                        cdn_has_frame,                          # CDN_has_frame
                        self.decision,                          # Is_I frame edge
                        quality,                                # real quality
                        self.buffer_status,                     # Is the buffer is buffering
                        switch,                                 # switch
                        False,                                  # Is the CDN has no frame
                        end_of_video]                           # Is the end of video
        # If not buffering
        elif not end_of_video: 
            # the normal loop
            # the buffer is enough
            if self.buffer_size > duration * play_duration_weight:
                self.buffer_size -= duration * play_duration_weight
                self.play_time += duration * play_duration_weight 
                rebuf = 0
            # the buffer not enough
            else:
                self.play_time += self.buffer_size
                rebuf = duration  - (self.buffer_size / play_duration_weight)
                self.buffer_size = 0
                self.buffer_status = True
            # the normal play the frame
            #play add the time
            self.buffer_size += self.frame_time_len[quality][self.video_chunk_counter[quality]]
            self.time += duration
            self.down_time += self.frame_time_len[quality][self.video_chunk_counter[quality]]
            self.latency = sum(self.frame_time_len[quality][ self.video_chunk_counter[quality] : self.newest_frame[quality]])   + self.buffer_size
            if self.latency > latency_threshold :
                self.skip_time_frame = self.video_chunk_counter[quality]
                if self.video_chunk_counter[quality] + skip_add_frame < self.newest_frame[quality]:
                    #ADD_FRAME = self.play_time_counter + skip_add_frame - self.video_chunk_counter
                    ADD_FRAME = skip_add_frame
                    ADD_FRAME_TIME = sum(self.frame_time_len[quality][self.video_chunk_counter[quality]: self.video_chunk_counter[quality] + ADD_FRAME +1])
                    self.down_time += ADD_FRAME_TIME
                else:
                    ADD_FRAME = self.newest_frame[quality] - self.video_chunk_counter[quality]
                    ADD_FRAME_TIME = sum(self.frame_time_len[quality][self.video_chunk_counter[quality]: self.video_chunk_counter[quality] + ADD_FRAME +1])
                    self.down_time += ADD_FRAME_TIME
                rebuf += ADD_FRAME * FRAME_TIME_LEN  
                if self.Debug:
                    print("skip events: skip_download_frame, play_frame, new_download_frame, ADD_frame" + str(self.skip_time_frame) + "  " + str(self.video_chunk_counter[quality]) +" " +str(ADD_FRAME) + "\n")
                    self.log_file.write("skip events: skip_download_frame, play_frame, new_download_frame, ADD_frame" + str(self.skip_time_frame) + "  " + str(self.video_chunk_counter[quality]) +" " +str(ADD_FRAME) + "\n")
            #else:
            #    self.video_chunk_counter[quality] += 1
            if self.Debug:
                 print("%.4f"%self.time ,
                      "~cdn_rebuf~%.4f"%cdn_rebuf_time,
                      "~rebuf~~ %.3f"%rebuf,
                      "~dur~~%.4f"%duration,
                      "~id! ", self.video_chunk_counter,
                      "~newest ", self.newest_frame,
                      "~~buffer~~ %4f"%self.buffer_size, 
                      "~~play_time~~%4f"%self.play_time ,
                      "~~quality~",quality,
                      "~~down_time~%.4f"%self.down_time,
                      "~~latency~~%4f"%self.latency,"222")
                 self.log_file.write("real_time %.4f\t"%self.time +  
                                      "cdn_rebuf%.4f\t"%cdn_rebuf_time + 
                                      "client_rebuf %.3f\t"%rebuf  + 
                                      "download_duration %.4f\t"%duration + 
                                      "download_id %s\t"%str(self.video_chunk_counter) + 
                                      "cdn_newest_frame %s\t"%str(self.newest_frame) + 
                                      "frame_size %.4f\t"%video_frame_size + 
                                      "video_time %.4f\t"% self.down_time +
                                      "client_buffer %.4f\t"%self.buffer_size  +
                                      "play_time %.4f\t"%self.play_time + 
                                      "latency %.4f\t"%self.latency + "222\n")
            cdn_has_frame = []
            if self.newest_frame[quality] - self.video_chunk_counter[quality] > 50:
                 for bitrate in range(BITRATE_LEVELS):
                      cdn_has_frame.append(sum(self.video_size[bitrate][self.video_chunk_counter[bitrate] : self.newest_frame[bitrate]]))
            #return loop
            return      [self.time,                             # physical time
                        duration,                               # this loop duration, means the download time
                        video_frame_size,                       # frame data size
                        FRAME_TIME_LEN,                         # frame time len
                        rebuf,                                  # rebuf len
                        self.buffer_size,                       # client buffer
                        self.latency ,                          # latency
                        self.newest_frame ,                     # cdn the newest frame id
                        self.video_chunk_counter,               # download_id
                        cdn_has_frame,                          # CDN_has_frame
                        self.decision,                          # Is_I frame edge
                        quality,                                # real quality
                        self.buffer_status,                     # Is the buffer is buffering
                        switch,                                 # switch
                        False,                                  # Is the CDN has no frame
                        end_of_video]                           # Is the end of video
        # if the video is end
        if end_of_video:

            self.time = 0
            self.play_time = 0
            self.down_time = 0
            self.all_time0 = 0
            self.all_time1 = 0
            self.last_video_chunk0 = 0
            self.last_video_chunk1 = 0

            self.newest_frame = [0,0]
            self.video_chunk_counter = [0,0]

            self.last_bit_rate = 0
            self.buffer_size = 0

            # pick a random trace file
            #self.trace_idx = np.random.randint(len(self.all_cooked_time))
            self.trace_idx += 1
            if self.trace_idx >= len(self.all_cooked_time):
                self.trace_idx = 0 
            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]
       
            self.decision = False
            self.buffer_status = True
            self.skip_time_frame = 100000000

            ADD_FRAME = 0

            self.video_size = {}  # in bytes
            self.cdn_arrive_time = {}
            self.frame_time_len = {}
            self.gop_flag = {}

            for bitrate in range(BITRATE_LEVELS):
                self.video_size[bitrate] = []
                self.cdn_arrive_time[bitrate] = []
                self.frame_time_len[bitrate] = []
                self.gop_flag[bitrate] = []
                cnt = 0
                with open(self.video_size_file + str(bitrate)) as f:
                    for line in f:
                        self.video_size[bitrate].append(float(line.split()[1]))
                        self.frame_time_len[bitrate].append(1.0/float(line.split()[3]))
                        self.gop_flag[bitrate].append(int(float(line.split()[2])))
                        self.cdn_arrive_time[bitrate].append(float(line.split()[0]))
                        cnt += 1
            self.latency = self.frame_time_len[0][0]

            cdn_has_frame = []

            return      [self.time,                             # physical time
                        duration,                               # this loop duration, means the download time
                        video_frame_size,                       # frame data size
                        FRAME_TIME_LEN,                         # frame time len
                        rebuf,                                  # rebuf len
                        self.buffer_size,                       # client buffer
                        self.latency ,                          # latency
                        self.newest_frame ,                     # cdn the newest frame id
                        self.video_chunk_counter ,         # download_id
                        cdn_has_frame,                          # CDN_has_frame
                        self.decision,                          # Is_I frame edge
                        quality,                                # real quality
                        self.buffer_status,                     # Is the buffer is buffering
                        switch,                                 # play time len
                        False,                                  # Is the CDN has no frame
                        True]                                   # Is the end of video
             
