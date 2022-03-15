sim : 
	nvcc -arch sm_75 main.cu utilities.cu Logger.cu boundary.cu lbm.cu -o sim

clean :
	rm -rf sim *.csv