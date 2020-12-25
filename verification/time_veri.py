import torch,time

class Time_veri():
    def __init__(self):
        self.t_start = 0
        self.t_end = 0
        self.t_temp = 0
        self.t_point0 = {"start":[] , "end":[], "time":[]}
        self.t_point1 = {"start":[] , "end":[], "time":[]}
        self.t_point2 = {"start":[] , "end":[], "time":[]}
        self.t_point3 = {"start":[] , "end":[], "time":[]}
        self.t_point4 = {"start":[] , "end":[], "time":[]}
        self.t_point5 = {"start":[] , "end":[], "time":[]}
        self.t_point6 = {"start":[] , "end":[], "time":[]}

    # time.timeを返すのが基本で、pytorchとgpu使ってるときは同期を含む
    #time.time同士の引き算じゃないと秒として機能しない(片方が0の時は意味のない数字になってしまう)
    def time_start(self):
        # pytorch-accurate time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.t_start =  time.time()

    def time_end(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.t_end =  time.time()

    def start_point(self, point_num):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        point_num["start"].append(time.time())

    def end_point(self, point_num):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        point_num["end"].append(time.time())

    def drop_first(self, point_num):
        point_num["start"].pop(-1)
        point_num["end"].pop(0)


    def time_sum_list(self,return_all=False):
        self.t_point0["time"] = [x - y for (x, y) in zip(self.t_point0["end"], self.t_point0["start"])]
        self.t_point1["time"] = [x - y for (x, y) in zip(self.t_point1["end"], self.t_point1["start"])]
        self.t_point2["time"] = [x - y for (x, y) in zip(self.t_point2["end"], self.t_point2["start"])]
        self.t_point3["time"] = [x - y for (x, y) in zip(self.t_point3["end"], self.t_point3["start"])]
        self.t_point4["time"] = [x - y for (x, y) in zip(self.t_point4["end"], self.t_point4["start"])]
        self.t_point5["time"] = [x - y for (x, y) in zip(self.t_point5["end"], self.t_point5["start"])]
        self.t_point6["time"] = [x - y for (x, y) in zip(self.t_point6["end"], self.t_point6["start"])]

        li = [sum(self.t_point0["time"]), sum(self.t_point1["time"]), sum(self.t_point2["time"]), sum(self.t_point3["time"]),
         sum(self.t_point4["time"]), sum(self.t_point5["time"]), sum(self.t_point6["time"])]
        self.all = sum(li)

        if return_all == False:
          return li
        return  self.all

    @staticmethod
    def time_oneshot():
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return time.time()

# time_veri = Time_veri()
# time_veri.time_temp()
# time_veri.t_point(time_veri.t_point0)
# time_veri.time_sum_list()