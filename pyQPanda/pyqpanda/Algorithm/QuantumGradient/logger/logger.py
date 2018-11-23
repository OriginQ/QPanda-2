import datetime
import time
import os
class logger:
    
    filename='default_logger.txt'
    fp = None
    def __init__(self):
        pass

    @staticmethod
    def init(foldername, prefix='rawdata', add_time=True):
        
        if not os.path.exists(foldername):
            os.mkdir(foldername)

        if add_time==True:
            logger.filename = '{}/{}-{}.txt'.format(foldername,prefix,time.strftime('%Y%m%d-%H%M%S',time.localtime(time.time())))             
        else:
            logger.filename= '{}/{}.txt'.format(foldername,prefix)      

        logger.fp=open(logger.filename,'w+')
        logger.fp.write('{}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
        logger.fp.flush()

    @staticmethod
    def title(set_title):
        logger.fp.write('\n- {}\n\n'.format(set_title))
        logger.fp.flush()

    @staticmethod
    def log(data_tuple):
        for data in data_tuple:
            logger.fp.write('{:15.4f}'.format(data))            

        logger.fp.write('\n')
        logger.fp.flush()

    @staticmethod
    def finalize():
        logger.fp.close()

if __name__=='__main__':
    logger.init(foldername='test',prefix='scan2d')
    logger.title('scan_data')
    logger.log((2,3,5))
    logger.log((2,3,5))
    logger.log((2,3,5))
    logger.log((2,3,5))
    logger.log((2,3,5))