#log test
import logging

def main():
	logger = logging.getLogger()
	hdlr = logging.FileHandler(r"K:\nasa_data\test.log")
	#logging.basicConfig(filename=, filemode='w',level=logging.INFO)
	formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
	hdlr.setFormatter(formatter)
	logger.addHandler(hdlr) 
	logger.setLevel(logging.INFO)
	logger.info('started')
	logger.info('finished')
if __name__ == '__main__':
	main()
	