#0：waist left front 1: waist right front 2: wrist left back 3: wrist right back
#4: back top 5: chest 6: back left 7: back right 8: head top 9: head front 
#10: head side 11: left shoulder back 12: left shoulder top 13: left elbow out
#14: LUARMHigh 15: left hand out 16: left wrist out 17: left wrist in
#18: right shoulder back 19: right shoulder top 20: right elbow out
#21: RUARMHigh 22: right hand out 23: right wrist out 24: right wrist in
#25: left knee out 26: LThigh 27: left ankle out 28: leftshin 29: LToeOut
#30: LToeIn 31: right knee out 32: RThigh 33: right ankle out 34: rightshin
#35: right toe out 36 right toe in

import numpy as np

marker_lines = np.asarray([[10,9],[9,8],[8,4],[4,6],[4,7],[6,11],[11,12],[11,13],[12,13],
                            [13,16],[16,15],[16,17],[7,18],[18,19],[18,20],[19,20],
                            [20,23],[23,24],[23,22],[12,5],[19,5],[0,1],[1,2],
                            [2,3],[3,0],[1,32],[32,31],[3,31],[31,33],[33,35],[35,36],
                            [0,26],[26,25],[2,25],[25,27],[27,29],[29,30],
                            [5,0],[5,1],[6,2],[7,3]])