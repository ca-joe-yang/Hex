import  threading, time, random, sys  
count =  0    
stop = 0
thread_id = []
class  Counter(threading.Thread):    
    def  __init__( self , lock, threadName):    
        super(Counter,  self ).__init__(name = threadName)
        self.lock = lock    
        
    def run( self ):    
        global  count, stop, thread_id
        thread_id.append(threading.get_ident())
        if threading.get_ident() == thread_id[0]:
            # self.lock.acquire()
            print ('123')
            while True:    
                print (stop, end='\b')
                if stop != 0:
                    break
            count = count + 100000
            # self .lock.release() 
        elif threading.get_ident() == thread_id[1]:
            # self.lock.acquire()
            print ('456')
            string = input('input anything!')
            print ('789')
            stop = 1
            # self .lock.release() 


lock = threading.Lock()    
for  i  in  range(2):     
    Counter(lock,  "thread-"  + str(i)).start()
while stop == 0:
    time.sleep(1)
print("Count={0}!".format(count))   
